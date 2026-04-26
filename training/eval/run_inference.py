"""
Zero-shot inference runner for CivicFlow.

Mirrors the Meta_RL supply-chain pattern:
  - heuristic_action()  — deterministic expert; establishes reward ceiling
  - run_episode()       — runs one episode with any policy_fn
  - main()              — runs model or heuristic, prints per-step rewards

Usage:
    python training/eval/run_inference.py --model ./models/qwen2.5-3b --task tiny_a
    python training/eval/run_inference.py --heuristic --task all
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

from civicflow_env.server.state import ZONE_USE_COMPAT, DESIGNATION_ALLOWED_USES
from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.server import verifier as _verifier
from civicflow_env.models import CivicflowAction

TASKS = ["tiny_a", "medium_a", "medium_b", "hard_a"]

# ---------------------------------------------------------------------------
# Heuristic policy — deterministic expert (mirrors proportional_heuristic
# in Meta_RL).  Runs greedily through remaining targets in priority order:
#   1. Reserve protected / no-road-access blocks as open space
#   2. Zone + develop housing blocks (largest remaining target first)
#   3. Zone + develop retail / office / workshop blocks
#   4. Assign required amenities (school, park, clinic, etc.)
#   5. Zone + develop institutional blocks
#   6. Reserve extra open space to hit greenery ratio
# ---------------------------------------------------------------------------

def heuristic_action(world) -> CivicflowAction:
    """Deterministic greedy expert — always makes progress."""
    t = world.targets
    counts = {u: 0 for u in ["housing", "retail", "office", "workshop", "institutional", "park"]}
    for b in world.blocks.values():
        if b.use:
            counts[b.use] = counts.get(b.use, 0) + 1

    amenity_cov: dict = {}
    for b in world.blocks.values():
        if b.amenity:
            amenity_cov[b.amenity] = amenity_cov.get(b.amenity, 0) + 1
        if b.use == "park":
            amenity_cov["park"] = amenity_cov.get("park", 0) + 1

    # 1. Reserve no-road / protected blocks that need open_space treatment
    for bid, b in world.blocks.items():
        if not b.is_developed and not b.reserved_open_space:
            if not b.has_road_access or b.is_protected:
                return CivicflowAction(action_type="reserve_open_space", block_id=bid)

    # 2. Assign required amenities that are still missing
    from civicflow_env.server.state import DESIGNATION_ALLOWED_AMENITIES
    missing_amenities = [a for a in t.required_amenities if amenity_cov.get(a, 0) == 0]
    for amenity in missing_amenities:
        need_inst = amenity in {"school", "fire"}
        ok_zones = ("civic", "mixed") if need_inst else ("civic", "mixed", "commercial", "open_space")

        # Prefer blocks whose designation explicitly allows this amenity
        def _amenity_block_key(item):
            bid, b = item
            if b.future_designation:
                allowed_am = DESIGNATION_ALLOWED_AMENITIES.get(b.future_designation, set())
                return 0 if amenity in allowed_am else 2
            return 1

        candidates = sorted(world.blocks.items(), key=_amenity_block_key)

        for bid, b in candidates:
            if b.amenity or not b.has_road_access:
                continue
            # Skip blocks whose designation forbids this amenity
            if b.future_designation:
                allowed_am = DESIGNATION_ALLOWED_AMENITIES.get(b.future_designation, set())
                if amenity not in allowed_am:
                    continue
            if amenity == "park":
                if b.zone in ("open_space", "civic", "unzoned"):
                    return CivicflowAction(action_type="assign_amenity", block_id=bid, amenity_type="park")
            else:
                if b.zone in ok_zones:
                    return CivicflowAction(action_type="assign_amenity", block_id=bid, amenity_type=amenity)
                if b.zone == "unzoned" and not b.is_protected:
                    return CivicflowAction(action_type="set_zoning", block_id=bid, zone="civic")

    # 2b. Proactively upgrade infra if any pending development would overflow
    for bid, b in world.blocks.items():
        if b.is_developed or b.use or b.reserved_open_space or not b.has_road_access or b.is_protected:
            continue
        iz = world.infra_zones.get(b.infra_zone)
        if iz is None:
            continue
        for infra_type, demand_attr, cap_attr in [
            ("water", "water_demand", "water_capacity"),
            ("sewer", "sewer_demand", "sewer_capacity"),
            ("power", "power_demand", "power_capacity"),
        ]:
            demand = getattr(b, demand_attr, 0)
            alloc = getattr(iz, f"{infra_type}_alloc", 0)
            cap = getattr(iz, cap_attr, 9999)
            if alloc + demand > cap:
                shortage = (alloc + demand) - cap + 10
                return CivicflowAction(
                    action_type="upgrade_infrastructure",
                    infra_zone=iz.infra_zone_id,
                    infra_type=infra_type,
                    capacity=float(min(shortage, 80)),
                )

    # 3. Develop blocks toward remaining use targets
    use_priority = ["housing", "retail", "office", "workshop", "institutional", "park"]
    for use in use_priority:
        remaining = t.blocks_by_use.get(use, 0) - counts.get(use, 0)
        if remaining <= 0:
            continue
        target_zone = {
            "housing": "residential",
            "retail": "commercial",
            "office": "commercial",
            "workshop": "industrial",
            "institutional": "civic",
            "park": "open_space",
        }[use]
        # Find a block already correctly zoned → develop
        for bid, b in world.blocks.items():
            if b.is_developed or b.phase != 0 or not b.has_road_access or b.is_protected:
                continue
            # Skip blocks whose designation forbids this use
            if b.future_designation:
                allowed = DESIGNATION_ALLOWED_USES.get(b.future_designation, set())
                if use not in allowed:
                    continue
            if b.zone == target_zone and use in ZONE_USE_COMPAT.get(b.zone, set()):
                if use == "park":
                    return CivicflowAction(action_type="assign_amenity", block_id=bid, amenity_type="park")
                return CivicflowAction(action_type="develop", block_id=bid, use=use)
        # Find an unzoned block → zone it
        for bid, b in world.blocks.items():
            if b.is_developed or b.phase != 0 or not b.has_road_access or b.is_protected:
                continue
            if b.future_designation:
                allowed = DESIGNATION_ALLOWED_USES.get(b.future_designation, set())
                if use not in allowed:
                    continue
            if b.zone == "unzoned":
                return CivicflowAction(action_type="set_zoning", block_id=bid, zone=target_zone)

    # 4. Add open space if greenery ratio not met
    n = len(world.blocks)
    green = sum(1 for b in world.blocks.values()
                if b.use == "park" or b.reserved_open_space or b.zone == "open_space")
    if n > 0 and green / n < t.min_greenery_ratio:
        for bid, b in world.blocks.items():
            if not b.is_developed and not b.reserved_open_space:
                return CivicflowAction(action_type="reserve_open_space", block_id=bid)

    # 5. Upgrade infra if there are overflows
    from civicflow_env.server import verifier as _v
    if _v._infra_overflow(world) > 0:
        for iz in world.infra_zones.values():
            for infra_type in ("water", "sewer", "power", "road"):
                alloc = getattr(iz, f"{infra_type}_alloc")
                cap = getattr(iz, f"{infra_type}_capacity")
                if alloc > cap:
                    return CivicflowAction(
                        action_type="upgrade_infrastructure",
                        infra_zone=iz.infra_zone_id,
                        infra_type=infra_type,
                        capacity=float(min(alloc - cap + 10, 80)),
                    )

    # 6. No useful action — defer undeveloped phase-0 blocks to clear them
    for bid, b in world.blocks.items():
        if not b.is_developed and b.phase == 0:
            return CivicflowAction(action_type="defer", block_id=bid, phase_id=2)
    # All blocks developed/deferred; upgrade infra to consume remaining steps legally
    first_iz = list(world.infra_zones.keys())[0]
    return CivicflowAction(
        action_type="upgrade_infrastructure",
        infra_zone=first_iz,
        infra_type="water",
        capacity=1.0,
    )


# ---------------------------------------------------------------------------
# LLM model helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a city planner. Output ONLY the JSON shown in recommended_action. No explanation, no markdown."


def _build_model_payload(obs, action_history: list, world) -> dict:
    """Show block states in plain text + pre-computed recommended next action.

    Mirrors Meta_RL: the model sees the current state clearly and a concrete
    recommended action — it just needs to output (confirm) that JSON.
    """
    ps = obs.planning_summary
    support = ps.get("planner_support", {})

    # Plain-text block status: what's done, what's next
    block_lines = []
    for bid, b in world.blocks.items():
        if b.use:
            block_lines.append(f"{bid}: {b.zone}/{b.use}" + (f"+{b.amenity}" if b.amenity else "") + " ✓")
        elif b.zone != "unzoned":
            block_lines.append(f"{bid}: zoned={b.zone}, NOT YET DEVELOPED")
        else:
            block_lines.append(f"{bid}: unzoned")

    # Pre-compute the recommended next action via heuristic
    rec = heuristic_action(world)
    rec_dict = rec.model_dump(exclude_none=True)

    return {
        "block_states": block_lines,
        "still_needed": {
            "use_targets": support.get("remaining_use_targets", {}),
            "amenities": support.get("remaining_amenities", []),
            "green_blocks": support.get("green_blocks_needed", 0),
        },
        "recommended_action": rec_dict,
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
    cfg = GenerationConfig(max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id)
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


# ---------------------------------------------------------------------------
# Episode runner — mirrors run_grader() in Meta_RL
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
        # Build action: heuristic OR LLM
        if tokenizer is not None:
            payload = _build_model_payload(obs, action_history, env._world)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
            ]
            raw = infer(tokenizer, model, device, messages)
            action = parse_action(raw)
            if action is None:
                parse_errors += 1
                action = heuristic_action(env._world)
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
# Normalised score — mirrors compute_normalised_score in Meta_RL
# ---------------------------------------------------------------------------

def compute_normalised_score(total_reward: float, max_reward: float = 10.0) -> float:
    """Normalise cumulative reward to [0, 1] relative to heuristic ceiling."""
    return round(max(0.0, min(1.0, total_reward / max(max_reward, 1))), 4)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Path to local model directory")
    p.add_argument("--heuristic", action="store_true", help="Run deterministic heuristic policy")
    p.add_argument("--task", default="tiny_a", help="Task id or 'all'")
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
            policy_fn=heuristic_action,
            tokenizer=tokenizer, model=model, device=device,
        )
        results.append(r)

    print(f"\n{'task':<12} {'avg_r':>8} {'total_r':>9} {'progress':>9} {'valid':>6} {'illegal':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['task_id']:<12} {r['avg_reward']:>+8.4f} {r['total_reward']:>+9.4f} "
              f"{r['progress']:>9.3f} {r['valid_plan']:>6} {r['illegal_actions']:>8}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
