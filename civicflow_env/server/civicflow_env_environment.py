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


def _maybe_advance_phase(w) -> str | None:
    """If current phase targets are met, reveal next batch of blocks and update targets."""
    if not w.pending_phases:
        return None
    if verifier._progress(w) < 0.999 or verifier._violations(w) != 0:
        return None
    next_phase = w.pending_phases.pop(0)
    w.phase_idx += 1
    # Reveal new blocks (constraints already baked into block definitions in the task JSON).
    for block_id in next_phase.get("new_block_ids", []):
        b = w._hidden_blocks.pop(block_id, None)
        if b is not None:
            w.blocks[block_id] = b
    # Rebuild adjacency for all currently revealed blocks.
    adj = {bid: [] for bid in w.blocks}
    for edge in w._all_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        a, b_id = str(edge[0]), str(edge[1])
        if a in adj and b_id in adj:
            adj[a].append(b_id)
            adj[b_id].append(a)
    for bid in adj:
        adj[bid] = sorted(set(adj[bid]))
    w.adjacency = adj
    # Update cumulative targets from next phase.
    pt = next_phase["targets"]
    t = w.targets
    t.blocks_by_use = dict(pt["blocks_by_use"])
    t.required_amenities = list(pt["required_amenities"])
    t.min_greenery_ratio = float(pt["min_greenery_ratio"])
    if "district_targets" in pt:
        t.district_targets = dict(pt["district_targets"])
    if "service_radius" in pt:
        t.service_radius = {k: int(v) for k, v in pt["service_radius"].items()}
    if "max_population" in pt:
        t.max_population = int(pt["max_population"])
    return next_phase["name"]


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

        w.action_history.append(action.model_dump())

        timeout = self._state.step_count >= w.targets.max_episode_steps
        metrics = verifier.compute_metrics(
            w, last_action_legal=legal, last_action_was_mutator=False, timeout=timeout,
            prev_metrics=self._prev_metrics,
        )
        terminal_success = bool(metrics["final_valid_plan"])
        self._done = bool(terminal_success or timeout)

        components = verifier.compute_reward_components(
            metrics, prev_metrics=self._prev_metrics, done=self._done,
            last_action_type=action.action_type,
        )
        self._prev_metrics = dict(metrics)
        reward = round(sum(components.values()), 4)

        # Check if current phase is complete; if so, reveal next batch of blocks.
        advanced_phase = _maybe_advance_phase(w)
        if advanced_phase:
            # Recompute metrics with newly revealed blocks.
            metrics = verifier.compute_metrics(
                w, last_action_legal=legal, last_action_was_mutator=False, timeout=timeout,
                prev_metrics=self._prev_metrics,
            )
            terminal_success = bool(metrics["final_valid_plan"])
            self._done = bool(terminal_success or timeout)
            components = verifier.compute_reward_components(
                metrics, prev_metrics=self._prev_metrics, done=self._done,
                last_action_type=action.action_type,
            )
            self._prev_metrics = dict(metrics)
            reward = round(sum(components.values()), 4)

        briefing_lines = [
            f"step {self._state.step_count}: " + ("OK — " if legal else "REJECTED — ") + message,
        ]
        if advanced_phase:
            briefing_lines.append(
                f"PHASE ADVANCE → '{advanced_phase}': {len(w.blocks)} blocks now revealed."
            )
        if self._done:
            if terminal_success:
                briefing_lines.append("TERMINAL: plan valid and complete.")
            elif timeout:
                briefing_lines.append("TERMINAL: timeout reached.")

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
            curveball_active=False,
            curveball_text=None,
            task_id=w.task_id,
            step_index=self._state.step_count,
            timeout=timeout,
            done=self._done,
            reward=reward,
            metadata={"history_len": len(w.action_history)},
        )

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> State:
        return self._state
