"""Generate SFT training examples from heuristic rollouts.

Runs the updated heuristic (designation-aware, proactive infra upgrade,
loop-safe fallback) on every phased task, records each (observation → action)
step in the Meta-RL supply-chain format, and writes heuristic_rollouts.jsonl.

The payload mirrors run_inference.py: simplified block_states + pre-computed
recommended_action so the model learns to confirm the correct action, not
reverse-engineer it from a raw observation blob.

Usage:
    python training/sft/generate_from_heuristic.py
    python training/sft/generate_from_heuristic.py --tasks tiny_a medium_a
    python training/sft/generate_from_heuristic.py --out custom_rollouts.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment

# Re-use the heuristic and payload builder from run_inference.py so the
# training data exactly matches what the model sees at inference time.
from training.eval.run_inference import heuristic_action, _build_model_payload, SYSTEM_PROMPT

OUT_PATH = Path(__file__).parent / "heuristic_rollouts.jsonl"

# Only phased tasks — nohint variants are for the no-hint baseline runner
DEFAULT_TASKS = ["tiny_a", "medium_a", "medium_b", "hard_a"]


def generate(task_ids=None, out_path=OUT_PATH):
    task_ids = task_ids or DEFAULT_TASKS
    examples = []

    for task_id in task_ids:
        os.environ["CIVICFLOW_TASK_ID"] = task_id
        env = CivicflowEnvironment()
        obs = env.reset()
        action_history = []
        step = 0

        while not env._done:
            # Build the same payload the model sees at inference time
            payload = _build_model_payload(obs, action_history, env._world)

            # Expert action = heuristic (designation-aware, proactive infra, loop-safe)
            action = heuristic_action(env._world)
            expert = action.model_dump(exclude_none=True)

            examples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": json.dumps(payload, separators=(",", ":"))},
                    {"role": "assistant", "content": json.dumps(expert,  separators=(",", ":"))},
                ],
                "meta": {
                    "task_id":   task_id,
                    "step":      step,
                    "difficulty": task_id.split("_")[0],
                },
            })

            obs = env.step(action)
            action_history.append(expert)
            step += 1

        valid = obs.last_metrics.get("final_valid_plan", 0)
        prog  = obs.last_metrics.get("progress_score",  0)
        print(f"  {task_id}: {step} steps  progress={prog:.3f}  valid={valid}")

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nGenerated {len(examples)} examples from {len(task_ids)} tasks → {out_path}")
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()
    generate(task_ids=args.tasks, out_path=Path(args.out))


if __name__ == "__main__":
    main()
