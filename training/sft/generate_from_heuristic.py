"""Generate SFT data from the greedy planner in training/baselines/heuristic.py.

Uses ``plan()`` and ``replan_after_curveball()`` — the same expert trace as
``heuristic.run_episode``, not ``run_inference.heuristic_action`` (different
ordering and infra heuristics).

Writes chat-formatted JSONL for Qwen/Unsloth and an optional plain-text dump
with ### STATE / ### ACTION sections.

Usage:
    python training/sft/generate_from_heuristic.py
    python training/sft/generate_from_heuristic.py --tasks tiny_a medium_a hard_a
    python training/sft/generate_from_heuristic.py --out-jsonl training/sft/sft_final.jsonl --out-txt training/sft/sft_final.txt
    python training/sft/generate_from_heuristic.py --repeat 3 --partial-prob 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.server.state import WorldState

from training.baselines.heuristic import plan, replan_after_curveball

OUT_JSONL = Path(__file__).parent / "sft_final.jsonl"
OUT_TXT = Path(__file__).parent / "sft_final.txt"

DEFAULT_TASKS = ["tiny_a", "medium_a", "medium_b", "hard_a"]

_SYSTEM_PROMPTS = [
    (
        "You are a CivicFlow city planner. Read the state and reply with exactly one JSON "
        "object: the next action. Keys may include action_type, block_id, zone, use, "
        "amenity_type, infra_zone, infra_type, capacity, phase_id — include only what applies. "
        "No markdown fences, no commentary."
    ),
    (
        "Role: municipal planner for CivicFlow. Given the state below, output a single JSON "
        "object describing your next legal action. Omit unused fields. JSON only."
    ),
    (
        "CivicFlow planning assistant. Respond with one compact JSON action (next step only). "
        "Allowed keys: action_type, block_id, zone, use, amenity_type, infra_zone, infra_type, "
        "capacity, phase_id."
    ),
]

_CLOSINGS = [
    "Next move: one JSON object only.",
    "Output only the JSON for the next action.",
    "Reply with the single next action as JSON.",
    "Return the next action as a JSON object, nothing else.",
]


def _block_sentence(b) -> str:
    parts = [f"{b.block_id}:"]
    if b.use:
        parts.append(
            f"zoned {b.zone}, built as {b.use}"
            + (f", amenity {b.amenity}" if b.amenity else "")
            + " (complete)"
        )
    elif b.zone != "unzoned":
        parts.append(f"zoned {b.zone}, not yet built")
    else:
        parts.append("unzoned, not built")

    flags = []
    if not b.has_road_access:
        flags.append("no road access")
    if b.is_protected:
        flags.append("protected land")
    if b.reserved_open_space:
        flags.append("reserved open space")
    if b.future_designation:
        flags.append(f"designation={b.future_designation}")
    if flags:
        parts.append("— " + ", ".join(flags))

    parts.append(
        f"infra {b.infra_zone}; demands water={b.water_demand}, "
        f"sewer={b.sewer_demand}, power={b.power_demand}"
    )
    return " ".join(parts)


def _legal_hints_compact(legal: Dict[str, Any], max_ids: int = 6) -> str:
    lines = []
    for k, v in legal.items():
        if k == "assign_amenity_by_type":
            continue
        if isinstance(v, list) and v and isinstance(v[0], str):
            shown = v[:max_ids]
            extra = len(v) - len(shown)
            suf = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"  {k}: {', '.join(shown)}{suf}")
        elif isinstance(v, list) and not v:
            continue
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "  (none)"


def _stable_mix(*parts: Any) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big")


def _prompt_variant_key(task_id: str, step: int, env_seed: Optional[int]) -> int:
    s = env_seed if env_seed is not None else 0
    return _stable_mix(task_id, s, step) % 1000000


def _system_prompt_for_row(task_id: str, step: int, env_seed: Optional[int]) -> str:
    k = _prompt_variant_key(task_id, step, env_seed)
    return _SYSTEM_PROMPTS[k % len(_SYSTEM_PROMPTS)]


def _closing_for_row(task_id: str, step: int, env_seed: Optional[int]) -> str:
    k = _prompt_variant_key(task_id, step + 17, env_seed)
    return _CLOSINGS[k % len(_CLOSINGS)]


def _blocks_display_order(
    world: WorldState,
    task_id: str,
    step: int,
    env_seed: Optional[int],
) -> List:
    blocks = list(world.blocks.values())
    rng = random.Random(_stable_mix(task_id, env_seed or 0, step, "blocks"))
    rng.shuffle(blocks)
    return blocks


def build_nl_user_prompt(
    world: WorldState,
    obs,
    action_history: List[dict],
    *,
    task_id: str,
    env_seed: Optional[int],
    compact_phase: bool = True,
) -> str:
    ps = obs.planning_summary
    support = ps.get("planner_support", {})
    phase = ps.get("current_phase") or {}
    legal = obs.legal_actions_summary or {}
    pname = phase.get("name", obs.current_phase)
    pobj = phase.get("objective", obs.phase_objective)
    if compact_phase and obs.step_index > 0:
        phase_line = f"Step: {obs.step_index} | Phase: {pname} (same episode; details were in step 0)"
    else:
        phase_line = f"Step: {obs.step_index} | Phase: {pname} — {pobj}"

    lines: List[str] = [
        f"Task: {world.task_id}",
        phase_line,
        "",
        "Briefing:",
        obs.briefing.strip(),
        "",
        "Targets still to satisfy:",
    ]
    rem_u = support.get("remaining_use_targets") or {}
    if rem_u:
        lines.append("  Uses: " + ", ".join(f"{k}={v}" for k, v in sorted(rem_u.items()) if v > 0))
    else:
        lines.append("  Uses: (none — phase uses satisfied or N/A)")
    am = support.get("remaining_amenities") or []
    lines.append("  Amenities: " + (", ".join(am) if am else "(none)"))
    gn = support.get("green_blocks_needed", 0)
    lines.append(f"  Extra green / open-space blocks needed (approx): {gn}")
    lines.append(f"  City greenery ratio now: {ps.get('greenery_ratio', 0)} (floor {world.targets.min_greenery_ratio})")
    lines.append("")
    lines.append("Blocks (order is arbitrary; use block_id):")
    for b in _blocks_display_order(world, task_id, obs.step_index, env_seed):
        lines.append("  - " + _block_sentence(b))
    lines.append("")
    lines.append("Infrastructure (alloc / capacity per zone):")
    for zid, caps in (ps.get("infra_zones") or {}).items():
        lines.append(f"  {zid}: {caps}")
    lines.append("")
    lines.append("Hint: candidate block ids by action family (not exhaustive):")
    lines.append(_legal_hints_compact(legal))
    lines.append("")
    if action_history:
        lines.append("Last action JSON:")
        lines.append(json.dumps(action_history[-1], separators=(",", ":")))
    else:
        lines.append("Last action: (episode start)")
    lines.append("")
    lines.append(_closing_for_row(task_id, obs.step_index, env_seed))
    return "\n".join(lines)


def action_to_dict(action) -> Dict[str, Any]:
    return {k: v for k, v in action.model_dump().items() if v is not None}


def sample_to_text(user: str, action_json: str) -> str:
    return (
        "### STATE\n"
        f"{user}\n\n"
        "### TASK\n"
        "Generate the next optimal action as JSON.\n\n"
        "### ACTION\n"
        f"{action_json}\n"
    )


def run_episode_examples(
    task_id: str,
    env_seed: Optional[int],
    partial_prob: float,
    compact_phase: bool = True,
) -> List[Dict[str, Any]]:
    os.environ["CIVICFLOW_TASK_ID"] = task_id
    if env_seed is not None:
        os.environ["CIVICFLOW_SEED"] = str(env_seed)
    else:
        os.environ.pop("CIVICFLOW_SEED", None)

    env = CivicflowEnvironment()
    obs = env.reset()
    world = env._world
    actions = plan(world)

    action_history: List[dict] = []
    examples: List[Dict[str, Any]] = []
    fires_seen = 0
    i = 0

    while i < len(actions) and not env._done:
        a = actions[i]
        user_nl = build_nl_user_prompt(
            world,
            obs,
            action_history,
            task_id=task_id,
            env_seed=env_seed,
            compact_phase=compact_phase,
        )
        expert = action_to_dict(a)
        sys_p = _system_prompt_for_row(task_id, obs.step_index, env_seed)

        examples.append({
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_nl},
                {"role": "assistant", "content": json.dumps(expert, separators=(",", ":"))},
            ],
            "meta": {
                "task_id": task_id,
                "step": obs.step_index,
                "difficulty": task_id.split("_")[0],
                "planner": "heuristic.plan",
                "env_seed": env_seed,
            },
        })

        obs = env.step(a)
        action_history.append(expert)

        currently_fired = sum(1 for f in world.curveballs_fired if f)
        if currently_fired > fires_seen:
            fires_seen = currently_fired
            extra = replan_after_curveball(world)
            actions = actions[: i + 1] + extra + actions[i + 1 :]
        i += 1

    if partial_prob > 0.0 and len(examples) > 2 and random.random() < partial_prob:
        cut = random.randint(1, len(examples) - 1)
        examples = examples[:cut]

    return examples


def _assistant_canonical(assistant_raw: str) -> str:
    try:
        return json.dumps(json.loads(assistant_raw), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return assistant_raw


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop rows that repeat the same (task, env step, action).

    With ``CIVICFLOW_TASK_ID`` set, the initial world does not depend on
    ``CIVICFLOW_SEED`` (see ``tasks.pick_task``). Running ``--repeat`` therefore
    replays identical trajectories — only prompt shims differ — which blows up
    the JSONL with near-duplicates.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta") or {}
        tid = meta.get("task_id", "")
        step = meta.get("step", -1)
        asst = _assistant_canonical(row["messages"][2]["content"])
        key = (tid, step, asst)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def generate(
    task_ids: List[str],
    out_jsonl: Path,
    out_txt: Optional[Path],
    repeat: int,
    partial_prob: float,
    seed: Optional[int],
    compact_phase: bool = True,
    dedupe: bool = True,
) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    if repeat > 1:
        print(
            "Note: With a fixed task id, each repeat uses the same initial world "
            "(RNG does not re-sample the task). Without --no-dedupe you get one "
            "row per (task, step, action); use --repeat only for partial-prob aug "
            "or add more task files.",
            flush=True,
        )

    all_rows: List[Dict[str, Any]] = []
    for tid in task_ids:
        for r in range(max(1, repeat)):
            env_seed = random.randint(0, 2**31 - 1) if repeat > 1 else None
            rows = run_episode_examples(
                tid,
                env_seed=env_seed,
                partial_prob=partial_prob,
                compact_phase=compact_phase,
            )
            all_rows.extend(rows)
            last_step = rows[-1]["meta"]["step"] if rows else -1
            print(f"  {tid} repeat={r}: {len(rows)} examples (last step index {last_step})")

    raw_n = len(all_rows)
    if dedupe:
        all_rows = dedupe_rows(all_rows)
        print(f"  Deduped: {raw_n} -> {len(all_rows)} rows (by task_id + step + action)")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    if out_txt:
        with open(out_txt, "w") as f:
            for row in all_rows:
                msgs = row["messages"]
                user = msgs[1]["content"]
                act = msgs[2]["content"]
                f.write(sample_to_text(user, act))
                f.write("\n---\n")

    print(f"\nWrote {len(all_rows)} examples → {out_jsonl}")
    if out_txt:
        print(f"Plain text → {out_txt}")
    return all_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Heuristic.plan → NL SFT JSONL + optional TXT")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--all-tasks", action="store_true", help="Use every task JSON under civicflow_env/data/tasks")
    p.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    p.add_argument("--out-txt", type=Path, default=OUT_TXT,
                   help="Plain-text ### STATE / ### ACTION dump")
    p.add_argument("--no-txt", action="store_true", help="Do not write the .txt file")
    p.add_argument("--repeat", type=int, default=1, help="Roll each task this many times with random env seeds")
    p.add_argument("--partial-prob", type=float, default=0.0,
                   help="Probability to truncate each episode to a random prefix (data aug)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for partial cuts / repeat draws")
    p.add_argument(
        "--full-phase-text-every-step",
        action="store_true",
        help="Repeat full phase objective on every step (default: shorten after step 0)",
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep repeated (task, step, action) rows e.g. for block-order paraphrase aug",
    )
    p.add_argument(
        "--dedupe-only",
        type=Path,
        default=None,
        metavar="PATH",
        help="Only read this JSONL, apply dedupe, write to --out-jsonl (skip env rollouts)",
    )
    args = p.parse_args()

    if args.dedupe_only:
        rows = []
        with open(args.dedupe_only) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        n0 = len(rows)
        rows = dedupe_rows(rows)
        out_jsonl = args.out_jsonl
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Deduped {args.dedupe_only}: {n0} -> {len(rows)} rows -> {out_jsonl}")
        return

    if args.all_tasks:
        from civicflow_env.tasks import list_task_ids
        task_ids = list_task_ids()
    else:
        task_ids = args.tasks

    out_txt: Optional[Path] = None if args.no_txt else args.out_txt

    generate(
        task_ids=task_ids,
        out_jsonl=args.out_jsonl,
        out_txt=out_txt,
        repeat=args.repeat,
        partial_prob=args.partial_prob,
        seed=args.seed,
        compact_phase=not args.full_phase_text_every_step,
        dedupe=not args.no_dedupe,
    )


if __name__ == "__main__":
    main()
