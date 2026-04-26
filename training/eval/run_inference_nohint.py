"""
No-hint inference runner for CivicFlow.

Baseline that dumps the full raw observation at the model with no pre-computed
recommended_action. Demonstrates how a zero-shot LLM behaves when it must
reason through the planning problem from scratch — no guided hints.

Contrast with run_inference.py which uses the Meta-RL supply-chain pattern
(phased revelation + recommended_action pre-filled by the heuristic).

Usage:
    python training/eval/run_inference_nohint.py --model ./models/qwen2.5-3b --task tiny_a_nohint
    python training/eval/run_inference_nohint.py --model ./models/qwen2.5-3b --task all
    python training/eval/run_inference_nohint.py --heuristic --task tiny_a_nohint
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.models import CivicflowAction

TASKS = ["tiny_a_nohint", "medium_a_nohint", "medium_b_nohint", "hard_a_nohint"]

# ---------------------------------------------------------------------------
# No-hint system prompt — full instructions, no recommended action
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a municipal planner. Given the current city state, emit exactly one "
    "structured planning action as JSON. Legal actions are listed in legal_actions_summary. "
    "Treat the task as a dependency graph: targets require prerequisite actions "
    "(zone before develop), and each action should reduce planner_support.remaining_use_targets, "
    "planner_support.remaining_amenities, or planner_support.green_blocks_needed. "
    "Do not repeat an action on the same block. Do not reserve extra open space while "
    "housing/retail/institutional targets are still unmet unless the block is protected, "
    "has no road access, has a future_designation, or is needed for greenery. "
    "Respond with only the JSON object — no explanation, no markdown fences.\n\n"
    "Example valid responses:\n"
    '  {"action_type":"set_zoning","block_id":"B1","zone":"residential"}\n'
    '  {"action_type":"develop","block_id":"B1","use":"housing"}\n'
    '  {"action_type":"reserve_open_space","block_id":"B3"}\n'
    '  {"action_type":"assign_amenity","block_id":"B2","amenity_type":"school"}'
)


def _build_nohint_payload(obs, action_history: list) -> dict:
    """Full raw observation — no pre-computed recommended action."""
    ps = obs.planning_summary
    return {
        "task_briefing": obs.briefing if not action_history else None,
        "current_phase": obs.current_phase,
        "phase_objective": obs.phase_objective,
        "observation_summary": ps,
        "active_constraints": obs.active_constraints,
        "legal_actions_summary": obs.legal_actions_summary,
        "last_action": action_history[-1] if action_history else None,
    }


def load_model(model_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True).to(device)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model, device


def infer(tokenizer, model, device, messages: list) -> str:
    import torch
    from transformers import GenerationConfig
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    cfg = GenerationConfig(max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=cfg)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_action(raw: str) -> Optional[CivicflowAction]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    start = raw.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        return CivicflowAction(**json.loads(raw[start:end]))
    except Exception:
        return None


def _fallback_action(world) -> CivicflowAction:
    """Minimal safe fallback — zone the first unzoned accessible block."""
    for bid, b in world.blocks.items():
        if b.zone == "unzoned" and b.has_road_access and not b.is_protected:
            return CivicflowAction(action_type="set_zoning", block_id=bid, zone="residential")
    for bid, b in world.blocks.items():
        if not b.is_developed and b.phase == 0:
            return CivicflowAction(action_type="defer", block_id=bid, phase_id=2)
    first_iz = list(world.infra_zones.keys())[0]
    return CivicflowAction(action_type="upgrade_infrastructure", infra_zone=first_iz, infra_type="water", capacity=1.0)


def _heuristic_action(world) -> CivicflowAction:
    """Trivial heuristic for nohint baseline — just zone+develop in order."""
    from civicflow_env.server.state import ZONE_USE_COMPAT
    t = world.targets
    counts = {u: sum(1 for b in world.blocks.values() if b.use == u)
              for u in ["housing", "retail", "office", "workshop", "institutional", "park"]}

    for bid, b in world.blocks.items():
        if not b.is_developed and not b.reserved_open_space:
            if not b.has_road_access or b.is_protected:
                return CivicflowAction(action_type="reserve_open_space", block_id=bid)

    use_priority = ["housing", "retail", "office", "workshop", "institutional", "park"]
    for use in use_priority:
        remaining = t.blocks_by_use.get(use, 0) - counts.get(use, 0)
        if remaining <= 0:
            continue
        zone_map = {"housing": "residential", "retail": "commercial", "office": "commercial",
                    "workshop": "industrial", "institutional": "civic", "park": "open_space"}
        target_zone = zone_map[use]
        for bid, b in world.blocks.items():
            if b.is_developed or not b.has_road_access or b.is_protected:
                continue
            if b.zone == target_zone and use in ZONE_USE_COMPAT.get(b.zone, set()):
                if use == "park":
                    return CivicflowAction(action_type="assign_amenity", block_id=bid, amenity_type="park")
                return CivicflowAction(action_type="develop", block_id=bid, use=use)
        for bid, b in world.blocks.items():
            if b.is_developed or not b.has_road_access or b.is_protected:
                continue
            if b.zone == "unzoned":
                return CivicflowAction(action_type="set_zoning", block_id=bid, zone=target_zone)

    for bid, b in world.blocks.items():
        if not b.is_developed and b.phase == 0:
            return CivicflowAction(action_type="defer", block_id=bid, phase_id=2)
    first_iz = list(world.infra_zones.keys())[0]
    return CivicflowAction(action_type="upgrade_infrastructure", infra_zone=first_iz, infra_type="water", capacity=1.0)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    task_id: str,
    policy_fn: Callable,
    *,
    tokenizer=None,
    model=None,
    device=None,
    verbose: bool = True,
) -> dict:
    os.environ["CIVICFLOW_TASK_ID"] = task_id
    env = CivicflowEnvironment()
    obs = env.reset()

    total_reward = 0.0
    illegal = 0
    parse_errors = 0
    action_history = []
    step = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Briefing: {obs.briefing[:100]}...")
        print(f"{'='*60}")

    while not env._done:
        if tokenizer is not None:
            payload = _build_nohint_payload(obs, action_history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
            ]
            raw = infer(tokenizer, model, device, messages)
            action = parse_action(raw)
            if action is None:
                parse_errors += 1
                action = _fallback_action(env._world)
        else:
            action = policy_fn(env._world)

        obs = env.step(action)
        total_reward += obs.reward
        action_history.append(action.model_dump(exclude_none=True))

        if obs.last_metrics.get("illegal_action"):
            illegal += 1

        if verbose:
            flag = "ILLEGAL" if obs.last_metrics.get("illegal_action") else "ok"
            desc = action.action_type + f"({action.block_id or action.infra_zone or ''})"
            if action.zone:
                desc += f" zone={action.zone}"
            if action.use:
                desc += f" use={action.use}"
            if action.amenity_type:
                desc += f" amenity={action.amenity_type}"
            print(f"  step {step:>2} [{flag:>7}]  r={obs.reward:+.4f}  cumul={total_reward:+.4f}"
                  f"  prog={obs.last_metrics.get('progress_score', 0):.3f}  | {desc}")
        step += 1

    if verbose:
        valid = obs.last_metrics.get("final_valid_plan", 0)
        print(f"\n  DONE — steps={step}  total_r={total_reward:+.4f}  "
              f"avg_r={total_reward/max(step,1):+.4f}  "
              f"progress={obs.last_metrics.get('progress_score', 0):.3f}  "
              f"valid={valid}  illegal={illegal}  parse_errors={parse_errors}\n")

    return {
        "task_id": task_id,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "avg_reward": round(total_reward / max(step, 1), 4),
        "progress": obs.last_metrics.get("progress_score", 0),
        "valid_plan": obs.last_metrics.get("final_valid_plan", 0),
        "illegal_actions": illegal,
        "parse_errors": parse_errors,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Path to local model directory")
    p.add_argument("--heuristic", action="store_true", help="Run simple heuristic (no hints)")
    p.add_argument("--task", default="tiny_a_nohint", help="Task id or 'all'")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not args.model and not args.heuristic:
        p.error("Provide --model <path> or --heuristic")

    tasks = TASKS if args.task == "all" else [args.task]

    tokenizer = model = device = None
    if args.model:
        tokenizer, model, device = load_model(args.model)

    results = []
    for tid in tasks:
        r = run_episode(
            tid,
            policy_fn=_heuristic_action,
            tokenizer=tokenizer, model=model, device=device,
        )
        results.append(r)

    print(f"\n{'task':<20} {'avg_r':>8} {'total_r':>9} {'progress':>9} {'valid':>6} {'illegal':>8} {'parse_err':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['task_id']:<20} {r['avg_reward']:>+8.4f} {r['total_reward']:>+9.4f} "
              f"{r['progress']:>9.3f} {r['valid_plan']:>6} {r['illegal_actions']:>8} {r['parse_errors']:>10}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
