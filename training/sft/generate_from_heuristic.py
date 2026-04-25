"""Generate SFT training examples from heuristic rollouts.

Runs the greedy heuristic on every task, records each (observation → action)
step, and formats the result into the standard messages schema used by the
SFT trainer.  Produces heuristic_rollouts.jsonl alongside this script.
"""

import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from civicflow_env.tasks import list_task_ids
from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.models import CivicflowAction
from training.baselines.heuristic import plan, replan_after_curveball

SYSTEM_PROMPT = (
    "You are a municipal planner. Given the current city state, emit exactly one "
    "structured planning action as JSON. Legal actions are listed in legal_actions_summary. "
    "Respond with only the JSON object — no explanation, no markdown fences."
)

OUT_PATH = Path(__file__).parent / "heuristic_rollouts.jsonl"


def generate():
    examples = []
    task_ids = list_task_ids()

    for task_id in task_ids:
        os.environ["CIVICFLOW_TASK_ID"] = task_id
        env = CivicflowEnvironment()
        obs = env.reset()
        world = env._world
        actions = plan(world)

        fires_seen = 0
        i = 0
        while i < len(actions) and not env._done:
            a = actions[i]
            user_payload = {
                "task_briefing": obs.briefing,
                "current_phase": obs.current_phase,
                "phase_objective": obs.phase_objective,
                "observation_summary": obs.planning_summary,
                "active_constraints": obs.active_constraints,
                "legal_actions_summary": obs.legal_actions_summary,
            }
            expert = a.model_dump(exclude_none=True)
            difficulty = task_id.split("_")[0]
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(user_payload, separators=(",", ":"))},
                    {"role": "assistant", "content": json.dumps(expert, separators=(",", ":"))},
                ],
                "meta": {"task_id": task_id, "step": i, "difficulty": difficulty},
            })

            obs = env.step(a)

            # Detect newly-fired curveballs and insert reactive actions
            fired = sum(1 for f in world.curveballs_fired if f)
            if fired > fires_seen:
                fires_seen = fired
                extra = replan_after_curveball(world)
                actions = actions[:i + 1] + extra + actions[i + 1:]
            i += 1

        print(f"  {task_id}: {i} steps, valid={env._done}")

    with open(OUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nGenerated {len(examples)} examples from {len(task_ids)} tasks -> {OUT_PATH}")
    return examples


if __name__ == "__main__":
    generate()
