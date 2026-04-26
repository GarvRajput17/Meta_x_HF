"""
LLM semantic judge for CivicFlow — evaluates planning quality, not correctness.

The verifier already checks hard constraints (legality, infra caps, amenity counts).
This judge asks an LLM: "Does this plan make urban planning sense?"

Usage:
    # Judge a single completed episode (pass episode dict from test_llm_episode.py)
    python training/eval/llm_judge.py --results results.json

    # Run an episode and immediately judge it
    python training/eval/llm_judge.py --task tiny_a --run-episode

    # Judge multiple tasks, print comparison table
    python training/eval/llm_judge.py --task all --run-episode --out judged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.models import CivicflowAction

EVAL_TASKS = ["tiny_a", "medium_a", "medium_b"]

# --------------------------------------------------------------------------- #
# Judge prompt
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """\
You are an expert urban planning evaluator. You will receive a completed city planning episode:
- The task briefing (goals, constraints)
- The final block layout (what was built where)
- Key planning metrics (greenery, service gaps, etc.)
- Any curveball events that fired mid-episode

Score the plan on four dimensions, each 0–10:

1. spatial_coherence  — Are land uses grouped sensibly? (housing clusters together, not scattered between industrial blocks; parks near residential, not industrial dead-ends)
2. amenity_placement  — Are schools/clinics near housing? Is grocery accessible to residential areas? Are fire stations centrally placed?
3. curveball_response — Was replanning surgical (only affected blocks changed) or chaotic (unrelated blocks disturbed)? Score N/A (null) if no curveball fired.
4. overall_quality    — Holistic urban planning quality: transitions make sense, density fits infrastructure, the plan reads as intentional.

Respond ONLY with a JSON object in this exact schema:
{
  "spatial_coherence": <int 0-10>,
  "amenity_placement": <int 0-10>,
  "curveball_response": <int 0-10 or null>,
  "overall_quality": <int 0-10>,
  "reasoning": "<2-3 sentences explaining the scores>"
}
"""

JUDGE_USER_TEMPLATE = """\
TASK BRIEFING:
{briefing}

FINAL BLOCK LAYOUT:
{block_layout}

KEY METRICS:
{metrics}

CURVEBALL EVENTS:
{curveball_info}
"""


# --------------------------------------------------------------------------- #
# Context builders
# --------------------------------------------------------------------------- #

def _build_judge_context(world, final_obs, action_history: List[dict]) -> str:
    """Build the user message for the judge from world state + final observation."""
    blocks = world.blocks
    layout_lines = []
    for bid, b in sorted(blocks.items()):
        parts = [bid]
        parts.append(f"zone={b.zone}")
        if b.use:
            parts.append(f"use={b.use}")
        if b.amenity:
            parts.append(f"amenity={b.amenity}")
        if b.reserved_open_space:
            parts.append("open_space")
        if b.future_designation:
            parts.append(f"designation={b.future_designation}")
        if not b.has_road_access:
            parts.append("no_road")
        layout_lines.append("  " + ", ".join(parts))
    block_layout = "\n".join(layout_lines)

    m = final_obs.last_metrics
    metrics_summary = json.dumps({
        "final_valid_plan": m.get("final_valid_plan"),
        "progress_score": round(float(m.get("progress_score", 0)), 3),
        "greenery_shortfall": m.get("greenery_shortfall"),
        "amenity_shortfall_count": m.get("amenity_shortfall_count"),
        "infra_overflow_count": m.get("infra_overflow_count"),
        "spatial_service_gap_count": m.get("spatial_service_gap_count"),
        "designation_violation_count": m.get("designation_violation_count"),
        "constraint_violations": m.get("constraint_violations"),
        "use_target_shortfall_count": m.get("use_target_shortfall_count"),
        "budget_remaining_ratio": round(float(m.get("budget_remaining_ratio", 1.0)), 3),
    }, indent=2)

    curveball_lines = []
    for cb in world.curveballs:
        fired = any(
            h.get("briefing", "").startswith("Curveball")
            for h in action_history
        )
        curveball_lines.append(
            f"  step {cb.fire_at_step}: [{cb.mutation}] {cb.description}"
        )
    curveball_info = "\n".join(curveball_lines) if curveball_lines else "  None"

    return JUDGE_USER_TEMPLATE.format(
        briefing=world.briefing,
        block_layout=block_layout,
        metrics=metrics_summary,
        curveball_info=curveball_info,
    )


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #

def call_groq(messages: list, model: str = "qwen/qwen3-32b") -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
    return resp.choices[0].message.content


def judge_episode(
    world,
    final_obs,
    action_history: List[dict],
    call_fn=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the LLM judge on a completed episode. Returns score dict."""
    if call_fn is None:
        call_fn = lambda msgs: call_groq(msgs)

    context = _build_judge_context(world, final_obs, action_history)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": context},
    ]

    raw = call_fn(messages)

    import re
    raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw_clean = re.sub(r"```[a-z]*\n?", "", raw_clean).strip()
    start = raw_clean.find("{")
    if start != -1:
        depth, end = 0, -1
        for idx, ch in enumerate(raw_clean[start:], start=start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        raw_clean = raw_clean[start:end] if end != -1 else raw_clean[start:]

    try:
        scores = json.loads(raw_clean)
    except Exception as e:
        scores = {"error": str(e), "raw": raw_clean[:200]}

    if verbose and "error" not in scores:
        print(f"\n  LLM Judge scores:")
        print(f"    spatial_coherence : {scores.get('spatial_coherence')}/10")
        print(f"    amenity_placement : {scores.get('amenity_placement')}/10")
        print(f"    curveball_response: {scores.get('curveball_response')}")
        print(f"    overall_quality   : {scores.get('overall_quality')}/10")
        print(f"    reasoning: {scores.get('reasoning', '')}")

    return scores


# --------------------------------------------------------------------------- #
# Episode runner (minimal, for --run-episode flag)
# --------------------------------------------------------------------------- #

def run_episode_with_judge(
    task_id: str,
    call_fn,
    judge_call_fn=None,
    max_steps: int = 35,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a heuristic episode (no LLM planner needed) and then judge the result."""
    # Use the heuristic baseline to get a valid plan quickly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from baselines.heuristic import run_episode as heuristic_run

    os.environ["CIVICFLOW_TASK_ID"] = task_id
    env = CivicflowEnvironment()
    obs = env.reset()

    action_history: List[dict] = []
    step = 0
    while not env._done and step < max_steps:
        # Use heuristic to pick action without re-importing (call verifier planner hints directly)
        ps = obs.planning_summary.get("planner_support", {})
        las = obs.legal_actions_summary

        action = _heuristic_action(obs, env._world)
        if action is None:
            break

        obs = env.step(action)
        action_history.append({
            "step": step,
            "action": action.model_dump(exclude_none=True),
            "briefing": obs.briefing[:80],
        })
        if verbose:
            flag = "ILLEGAL" if obs.last_metrics.get("illegal_action") else "ok"
            print(f"  step {step:>2} [{flag}] {action.action_type}({action.block_id or action.infra_zone or ''})")
        step += 1

    if verbose:
        print(f"\n  Final: valid={obs.last_metrics.get('final_valid_plan')} "
              f"progress={obs.last_metrics.get('progress_score', 0):.2f}")

    judge_scores = judge_episode(env._world, obs, action_history, call_fn=judge_call_fn, verbose=verbose)

    return {
        "task_id": task_id,
        "steps": step,
        "final_valid_plan": int(obs.last_metrics.get("final_valid_plan", 0)),
        "progress_score": float(obs.last_metrics.get("progress_score", 0)),
        "constraint_violations": obs.last_metrics.get("constraint_violations", 0),
        "judge_scores": judge_scores,
        "llm_quality_score": _aggregate_judge_score(judge_scores),
    }


def _aggregate_judge_score(scores: Dict) -> Optional[float]:
    """Weighted average of judge dimensions → [0, 1]."""
    if "error" in scores:
        return None
    weights = {"spatial_coherence": 0.3, "amenity_placement": 0.3, "overall_quality": 0.4}
    total, w_sum = 0.0, 0.0
    for k, w in weights.items():
        v = scores.get(k)
        if v is not None:
            total += float(v) * w
            w_sum += w
    cb = scores.get("curveball_response")
    if cb is not None:
        total += float(cb) * 0.2
        w_sum += 0.2
    return round(total / w_sum / 10.0, 3) if w_sum > 0 else None


def _heuristic_action(obs, world) -> Optional[CivicflowAction]:
    """Minimal greedy action picker for demo episodes."""
    las = obs.legal_actions_summary
    ps = obs.planning_summary.get("planner_support", {})

    # 1. Zone unzoned blocks toward remaining use targets
    zoneable = las.get("set_zoning", {}).get("unzoned_blocks", [])
    remaining = ps.get("remaining_use_targets", {})
    use_to_zone = {"housing": "residential", "retail": "commercial", "office": "commercial",
                   "institutional": "civic", "park": "civic"}
    for use, count in remaining.items():
        if count > 0 and zoneable:
            zone = use_to_zone.get(use, "mixed")
            return CivicflowAction(action_type="set_zoning", block_id=zoneable[0], zone=zone)

    # 2. Develop blocks that are zoned but not yet developed
    for use, blocks in las.get("develop", {}).items():
        if blocks and remaining.get(use, 0) > 0:
            return CivicflowAction(action_type="develop", block_id=blocks[0], use=use)

    # 3. Assign missing amenities
    for amenity in ps.get("remaining_amenities", []):
        candidates = las.get("assign_amenity", {}).get(amenity, [])
        if candidates:
            return CivicflowAction(action_type="assign_amenity", block_id=candidates[0], amenity_type=amenity)

    # 4. Reserve open space for greenery if needed
    if ps.get("green_blocks_needed", 0) > 0:
        candidates = las.get("reserve_open_space", {}).get("eligible_blocks", [])
        if candidates:
            return CivicflowAction(action_type="reserve_open_space", block_id=candidates[0])

    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="tiny_a", help="Task id or 'all'")
    p.add_argument("--results", default=None, help="JSON file with pre-run episode results to judge")
    p.add_argument("--run-episode", action="store_true", help="Run a heuristic episode then judge it")
    p.add_argument("--model", default="qwen/qwen3-32b", help="Groq model for the judge")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    judge_fn = lambda msgs: call_groq(msgs, args.model)
    tasks = EVAL_TASKS if args.task == "all" else [args.task]
    results = []

    if args.run_episode:
        for tid in tasks:
            if not args.quiet:
                print(f"\n{'='*60}\nTask: {tid}\n{'='*60}")
            r = run_episode_with_judge(tid, call_fn=None, judge_call_fn=judge_fn, verbose=not args.quiet)
            results.append(r)
    elif args.results:
        with open(args.results) as f:
            episode_data = json.load(f)
        print("--results mode requires a running env; use --run-episode instead")
        return
    else:
        print("Specify --run-episode or --results <file>")
        return

    # Summary table
    print(f"\n{'task':<12} {'valid':>5} {'progress':>9} {'llm_score':>10}  reasoning")
    print("-" * 80)
    for r in results:
        js = r.get("judge_scores", {})
        sc = r.get("llm_quality_score")
        sc_str = f"{sc:.3f}" if sc is not None else "err"
        reasoning = js.get("reasoning", "")[:60]
        print(f"{r['task_id']:<12} {r['final_valid_plan']:>5} {r['progress_score']:>9.2f} {sc_str:>10}  {reasoning}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
