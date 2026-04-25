"""Evaluate zero-shot, SFT, and GRPO models against the heuristic baseline.

Usage:
    # Heuristic only (no GPU needed)
    python training/eval/eval_model.py --mode heuristic

    # Zero-shot Qwen2.5-3B
    python training/eval/eval_model.py --mode zero_shot

    # SFT model
    python training/eval/eval_model.py --mode sft --model-id Aaryan369/civicflow-sft-qwen2.5-3b

    # GRPO model
    python training/eval/eval_model.py --mode grpo --model-id Aaryan369/civicflow-grpo-qwen2.5-3b

    # All (prints comparison table)
    python training/eval/eval_model.py --mode all
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.models import CivicflowAction
from civicflow_env.server import verifier
from training.baselines.heuristic import run_episode as heuristic_episode

EVAL_TASKS = ["tiny_a", "tiny_b", "tiny_c"]

SYSTEM_PROMPT = (
    "You are a municipal planner. Given the current city state, emit exactly one "
    "structured planning action as JSON. Legal actions are listed in legal_actions_summary. "
    "Respond with only the JSON object — no explanation, no markdown fences."
)


def _build_user_payload(obs) -> str:
    payload = {
        "task_briefing": obs.briefing,
        "current_phase": obs.current_phase,
        "phase_objective": obs.phase_objective,
        "observation_summary": obs.planning_summary,
        "active_constraints": obs.active_constraints,
        "legal_actions_summary": obs.legal_actions_summary,
    }
    return json.dumps(payload, separators=(",", ":"))


def eval_heuristic(tasks: List[str] = EVAL_TASKS) -> List[Dict]:
    results = []
    for t in tasks:
        r = heuristic_episode(t)
        results.append({
            "model": "Heuristic",
            "task_id": t,
            "valid_plan": r["final_valid_plan"],
            "progress": r["progress_final"],
            "illegal_actions": r["illegal_action_count"],
            "constraint_violations": r["constraint_violations_final"],
            "curveball_f1": _f1(r),
        })
    return results


def eval_llm(
    model_id: str,
    label: str,
    tasks: List[str] = EVAL_TASKS,
    max_steps: int = 60,
) -> List[Dict]:
    """Run one episode per task using a HuggingFace model via transformers pipeline."""
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        print("transformers not installed — skipping LLM eval")
        return []

    device = 0 if __import__("torch").cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=__import__("torch").float16 if device == 0 else __import__("torch").float32,
        max_new_tokens=128,
    )

    results = []
    for task_id in tasks:
        os.environ["CIVICFLOW_TASK_ID"] = task_id
        env = CivicflowEnvironment()
        obs = env.reset()

        illegal = 0
        step = 0
        while not env._done and step < max_steps:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_payload(obs)},
            ]
            out = pipe(messages)[0]["generated_text"]
            # Extract just the assistant reply
            raw = out[-1]["content"] if isinstance(out, list) else out
            try:
                action = CivicflowAction(**json.loads(raw.strip()))
            except Exception:
                # Illegal parse counts as illegal action; use a safe fallback
                illegal += 1
                # Emit a no-op reserve to advance without crashing
                from civicflow_env.tasks import list_task_ids
                first_block = list(env._world.blocks.keys())[0]
                action = CivicflowAction(action_type="reserve_open_space", block_id=first_block)

            obs = env.step(action)
            if obs.last_metrics.get("illegal_action"):
                illegal += 1
            step += 1

        results.append({
            "model": label,
            "task_id": task_id,
            "valid_plan": int(obs.last_metrics.get("final_valid_plan", 0)),
            "progress": float(obs.last_metrics.get("progress_score", 0.0)),
            "illegal_actions": illegal,
            "constraint_violations": obs.last_metrics.get("constraint_violations", 0),
            "curveball_f1": _f1({"affected_set_precision": obs.last_metrics.get("affected_set_precision", 0),
                                   "affected_set_recall":    obs.last_metrics.get("affected_set_recall", 0)}),
        })
    return results


def _f1(r: Dict) -> Optional[float]:
    p = r.get("affected_set_precision", 0)
    rec = r.get("affected_set_recall", 0)
    if (p + rec) == 0:
        return None
    return round(2 * p * rec / (p + rec), 3)


def _avg(rows: List[Dict], key: str) -> str:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return "?"
    return f"{sum(vals)/len(vals):.2f}"


def print_table(all_results: List[Dict]) -> None:
    from itertools import groupby
    grouped = {}
    for r in all_results:
        grouped.setdefault(r["model"], []).append(r)

    header = f"{'Model':<28} {'Valid%':>7} {'Illegal%':>9} {'Progress':>9} {'Curv F1':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for label, rows in grouped.items():
        valid_pct = f"{sum(r['valid_plan'] for r in rows)/len(rows)*100:.0f}%"
        illegal_pct = f"{sum(r['illegal_actions'] for r in rows)/max(1, len(rows)):.1f}"
        prog = _avg(rows, "progress")
        f1 = _avg(rows, "curveball_f1")
        print(f"{label:<28} {valid_pct:>7} {illegal_pct:>9} {prog:>9} {f1:>8}")
    print("=" * len(header) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="heuristic",
                   choices=["heuristic", "zero_shot", "sft", "grpo", "all"])
    p.add_argument("--model-id", default=None, help="HF model id for sft/grpo modes")
    p.add_argument("--tasks", nargs="+", default=EVAL_TASKS)
    p.add_argument("--out", default=None, help="write results JSON here")
    args = p.parse_args()

    all_results: List[Dict] = []

    if args.mode in ("heuristic", "all"):
        print("Evaluating heuristic baseline...")
        all_results += eval_heuristic(args.tasks)

    if args.mode in ("zero_shot", "all"):
        print("Evaluating zero-shot Qwen2.5-3B-Instruct...")
        all_results += eval_llm("Qwen/Qwen2.5-3B-Instruct", "Zero-shot Qwen2.5-3B", args.tasks)

    if args.mode in ("sft", "all"):
        mid = args.model_id or "Aaryan369/civicflow-sft-qwen2.5-3b"
        print(f"Evaluating SFT model ({mid})...")
        all_results += eval_llm(mid, "SFT Qwen2.5-3B", args.tasks)

    if args.mode in ("grpo", "all"):
        mid = args.model_id or "Aaryan369/civicflow-grpo-qwen2.5-3b"
        print(f"Evaluating GRPO model ({mid})...")
        all_results += eval_llm(mid, "SFT+GRPO Qwen2.5-3B", args.tasks)

    print_table(all_results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
