"""
CivicFlow environment — Phase 1 implementation.

Orchestration only. All transition logic + scoring lives in `verifier.py`;
all world data lives in `state.py`; tasks load from `data/tasks/*.json`.

Episode lifecycle per session:
  reset()
    -> pick a task (env var CIVICFLOW_TASK_ID or random) with a deterministic
       per-instance seed; build a fresh WorldState; fire pre-step curveball
       check (no-op since step_count=0); return briefing + zero metrics.
  step(action)
    -> apply action via verifier (legal moves mutate state, illegal ones
       leave state untouched and stamp metrics.illegal_action=1);
       fire curveball if due; if past curveball, record any block touched;
       compute fresh metrics + decomposed reward components; check timeout
       / terminal-success done conditions.

Anti-hacking guardrails (per the participant guide):
  - per-task max_episode_steps cap (timeout terminates the episode cleanly)
  - illegal action penalised but never crashes the server
  - no module-level mutable state between episodes (each instance owns its
    WorldState; SUPPORTS_CONCURRENT_SESSIONS=True keeps sessions isolated)
  - deterministic per-instance seed for reproducibility
"""

from __future__ import annotations

import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CivicflowAction, CivicflowObservation, ACTION_TYPES
    from ..tasks import pick_task
    from . import verifier
except ImportError:
    from models import CivicflowAction, CivicflowObservation, ACTION_TYPES
    from tasks import pick_task
    import verifier  # type: ignore


# Re-export for callers that want the canonical key sets.
METRIC_KEYS = verifier.METRIC_KEYS
REWARD_COMPONENT_KEYS = verifier.REWARD_COMPONENT_KEYS


# Set of action types whose successful application mutates a specific block.
# Used to track curveball-window touches for affected-set scoring.
_BLOCK_MUTATORS = {"set_zoning", "develop", "reserve_open_space",
                   "assign_amenity", "redevelop", "defer"}


def _instance_seed() -> int:
    base = os.environ.get("CIVICFLOW_SEED")
    if base is not None:
        try:
            return int(base)
        except ValueError:
            pass
    return random.SystemRandom().randint(0, 2**31 - 1)


class CivicflowEnvironment(Environment):
    """CivicFlow planner environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(_instance_seed())
        self._world = None
        self._prev_metrics = {}
        self._done = False

    # ------------------------------------------------------------------ reset
    def reset(self) -> CivicflowObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._world = pick_task(self._rng)
        self._done = False

        metrics = verifier.compute_metrics(
            self._world, last_action_legal=True, last_action_was_mutator=False, timeout=False,
        )
        self._prev_metrics = dict(metrics)
        components = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        phase = verifier.current_phase_info(self._world)

        return CivicflowObservation(
            briefing=self._world.briefing,
            planning_summary=verifier.planning_summary(self._world),
            legal_actions_summary=verifier.legal_actions_summary(self._world),
            active_constraints=verifier.active_constraints(self._world),
            current_phase=phase["name"],
            phase_objective=phase["objective"],
            last_metrics=metrics,
            last_reward_components=components,
            curveball_active=False,
            curveball_text=None,
            task_id=self._world.task_id,
            step_index=0,
            timeout=False,
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------- step
    def step(self, action: CivicflowAction) -> CivicflowObservation:  # type: ignore[override]
        if self._world is None:
            # Defensive: clients should call reset() first, but a step before
            # reset shouldn't crash the server.
            self.reset()

        w = self._world
        self._state.step_count += 1
        w.step_count = self._state.step_count

        legal = False
        message = ""
        if action.action_type not in ACTION_TYPES:
            message = f"unknown action_type '{action.action_type}'"
        else:
            legal, message = verifier.apply_action(w, action)

        # Curveball mechanic: any curveballs whose fire_at_step has been
        # reached fire now and mutate the world.
        cb_fired_indices = verifier.maybe_fire_curveballs(w)

        # Track block touches AFTER the FIRST curveball announcement. Actions
        # submitted on the firing step were sent blind (the agent only sees
        # the curveball in *this* step's observation), so they don't count.
        if (legal and w.first_fire_step is not None
                and self._state.step_count > w.first_fire_step
                and action.action_type in _BLOCK_MUTATORS and action.block_id):
            w.blocks_touched_after_curveball.add(action.block_id)

        w.action_history.append(action.model_dump())

        timeout = self._state.step_count >= w.targets.max_episode_steps
        metrics = verifier.compute_metrics(
            w, last_action_legal=legal, last_action_was_mutator=False, timeout=timeout,
        )
        terminal_success = bool(metrics["final_valid_plan"])
        self._done = bool(terminal_success or timeout)

        components = verifier.compute_reward_components(
            metrics, prev_metrics=self._prev_metrics, done=self._done,
        )
        self._prev_metrics = dict(metrics)
        reward = round(sum(components.values()), 4)

        briefing_lines = [
            f"step {self._state.step_count}: " + ("OK — " if legal else "REJECTED — ") + message,
        ]
        for i in cb_fired_indices:
            briefing_lines.append(f"CURVEBALL #{i+1}: {w.curveballs[i].description}")
        if self._done:
            if terminal_success:
                briefing_lines.append("TERMINAL: plan valid and complete.")
            elif timeout:
                briefing_lines.append("TERMINAL: timeout reached.")

        any_fired = any(w.curveballs_fired)
        active_text = " | ".join(
            f"#{i+1}: {w.curveballs[i].description}"
            for i, f in enumerate(w.curveballs_fired) if f
        ) or None
        phase = verifier.current_phase_info(w)

        return CivicflowObservation(
            briefing=" | ".join(briefing_lines),
            planning_summary=verifier.planning_summary(w),
            legal_actions_summary=verifier.legal_actions_summary(w),
            active_constraints=verifier.active_constraints(w),
            current_phase=phase["name"],
            phase_objective=phase["objective"],
            last_metrics=metrics,
            last_reward_components=components,
            curveball_active=any_fired,
            curveball_text=active_text,
            task_id=w.task_id,
            step_index=self._state.step_count,
            timeout=timeout,
            done=self._done,
            reward=reward,
            metadata={
                "history_len": len(w.action_history),
                "blocks_touched_after_curveball": sorted(w.blocks_touched_after_curveball),
            },
        )

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> State:
        return self._state
