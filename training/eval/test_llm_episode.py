"""
Quick LLM episode tester for CivicFlow.

Tests a Qwen model (via HF Inference API — free tier, no training credits used)
against the CivicFlow environment server.

Usage:
    # Default: Qwen2.5-3B-Instruct via HF Inference API (requires HF token)
    python training/eval/test_llm_episode.py --task tiny_a

    # Smaller model (even fewer credits)
    python training/eval/test_llm_episode.py --model Qwen/Qwen2.5-1.5B-Instruct --task tiny_a

    # Against the live HF Space instead of local server
    python training/eval/test_llm_episode.py --env-url https://aaryan369-civicflow-env.hf.space

    # Run all tiny tasks, save results
    python training/eval/test_llm_episode.py --task all --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.models import CivicflowAction

SYSTEM_PROMPT = (
    "You are a municipal planner. Given the current city state, emit exactly one "
    "structured planning action as JSON. Legal actions are listed in legal_actions_summary. "
    "Respond with only the JSON object — no explanation, no markdown fences.\n\n"
    "Example valid responses:\n"
    '  {"action_type":"set_zoning","block_id":"B1","zone":"residential"}\n'
    '  {"action_type":"develop","block_id":"B1","use":"housing"}\n'
    '  {"action_type":"reserve_open_space","block_id":"B3"}\n'
    '  {"action_type":"assign_amenity","block_id":"B2","amenity_type":"school"}'
)

EVAL_TASKS = ["tiny_a", "tiny_b", "tiny_c"]


def call_groq(messages: list, model: str = "qwen/qwen3-32b") -> str:
    """Call Qwen via Groq — uses OpenAI client pointed at Groq endpoint (no groq SDK needed)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,  # enough for <think> block + JSON
        temperature=0.3,
    )
    return resp.choices[0].message.content


def run_episode(task_id: str, call_fn, max_steps: int = 30, verbose: bool = True) -> dict:
    os.environ["CIVICFLOW_TASK_ID"] = task_id
    env = CivicflowEnvironment()
    obs = env.reset()

    total_reward = 0.0
    illegal = 0
    parse_errors = 0
    step = 0
    history = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Briefing: {obs.briefing[:120]}...")
        print(f"{'='*60}")

    while not env._done and step < max_steps:
        user_payload = {
            "task_briefing": obs.briefing if step == 0 else f"step {step}",
            "current_phase": obs.current_phase,
            "phase_objective": obs.phase_objective,
            "observation_summary": obs.planning_summary,
            "active_constraints": obs.active_constraints,
            "legal_actions_summary": obs.legal_actions_summary,
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(user_payload, separators=(",", ":"))},
        ]

        raw = call_fn(messages)

        # Extract just the JSON object from any model output format.
        # Handles: <think>...</think> blocks, markdown fences, plain JSON.
        import re
        raw_clean = raw.strip()

        # Remove <think>...</think> (with or without closing tag)
        raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
        raw_clean = re.sub(r"<think>.*$", "", raw_clean, flags=re.DOTALL)

        # Strip markdown fences
        raw_clean = re.sub(r"```[a-z]*\n?", "", raw_clean).strip()

        # Find the first {...} JSON object in whatever remains
        m = re.search(r"\{[^{}]*\}", raw_clean, re.DOTALL)
        if m:
            raw_clean = m.group(0)
        else:
            raw_clean = raw_clean.strip()

        try:
            action = CivicflowAction(**json.loads(raw_clean))
        except Exception as e:
            parse_errors += 1
            if verbose:
                print(f"  step {step:>2}: PARSE ERROR ({e}) | raw: {raw_clean[:80]}")
            # Safe fallback
            first_unzoned = next(
                (bid for bid, b in env._world.blocks.items() if b.zone == "unzoned" and b.has_road_access),
                list(env._world.blocks.keys())[0]
            )
            action = CivicflowAction(action_type="reserve_open_space", block_id=first_unzoned)

        obs = env.step(action)
        total_reward += obs.reward

        if obs.last_metrics.get("illegal_action"):
            illegal += 1

        history.append({
            "step": step,
            "action": action.model_dump(exclude_none=True),
            "reward": obs.reward,
            "progress": obs.last_metrics.get("progress_score"),
            "illegal": bool(obs.last_metrics.get("illegal_action")),
            "briefing": obs.briefing[:120],
        })

        if verbose:
            flag = "ILLEGAL" if obs.last_metrics.get("illegal_action") else "ok"
            print(f"  step {step:>2} [{flag:>7}] r={obs.reward:+.3f} prog={obs.last_metrics.get('progress_score', 0):.2f} | "
                  f"{action.action_type}({action.block_id or action.infra_zone or ''})")

        step += 1

    final_metrics = obs.last_metrics
    result = {
        "task_id": task_id,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "final_valid_plan": int(final_metrics.get("final_valid_plan", 0)),
        "progress_final": float(final_metrics.get("progress_score", 0)),
        "illegal_actions": illegal,
        "parse_errors": parse_errors,
        "constraint_violations": final_metrics.get("constraint_violations", 0),
        "done": env._done,
    }

    if verbose:
        print(f"\nResult: valid={result['final_valid_plan']} progress={result['progress_final']:.2f} "
              f"reward={result['total_reward']:+.3f} illegal={illegal} parse_errors={parse_errors}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Groq model id (default: qwen-qwq-32b)")
    p.add_argument("--task", default="tiny_a", help="Task id or 'all'")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--env-url", default="local", help="'local' or HF Space URL")
    p.add_argument("--out", default=None, help="JSON output path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.env_url != "local":
        os.environ["ENV_BASE_URL"] = args.env_url

    model = args.model or "qwen/qwen3-32b"
    call_fn = lambda msgs: call_groq(msgs, model)
    print(f"Provider: Groq / {model}")

    tasks = EVAL_TASKS if args.task == "all" else [args.task]
    results = []
    for tid in tasks:
        results.append(run_episode(tid, call_fn, args.max_steps, verbose=not args.quiet))

    # Summary table
    print(f"\n{'task':<12} {'valid':>5} {'progress':>9} {'reward':>9} {'illegal':>8} {'steps':>6}")
    for r in results:
        print(f"{r['task_id']:<12} {r['final_valid_plan']:>5} {r['progress_final']:>9.2f} "
              f"{r['total_reward']:>+9.3f} {r['illegal_actions']:>8} {r['steps']:>6}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
