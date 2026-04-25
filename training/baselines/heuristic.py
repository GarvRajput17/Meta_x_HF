"""
Greedy heuristic baseline for CivicFlow.

Two roles:
  1. Pre-LLM reference baseline in the comparison table ("non-ML expert").
  2. Source of SFT warm-start trajectories — these action sequences are
     correct by construction (the verifier accepts them), so we can train
     the LLM to imitate them without manual labelling.

This planner is privileged: it reads `WorldState` directly (per-block
is_protected, is_floodrisk, infra demand, etc.). LLM baselines see only
the observation. That's intentional — the heuristic is the "expert with
ground truth", not a peer model.

Strategy (single-pass greedy with one reactive replan after curveball):

  1. Triage blocks into:
       - must_be_open : protected/floodrisk but reachable -> open_space
       - must_defer   : no road access -> defer to a later phase
       - available    : everything else, sorted by infra cost ascending
  2. Allocate roles in this order, draining `available`:
       a. parks (use=park) for park-use targets and the park amenity
       b. civic blocks for non-park amenities + institutional target
       c. housing -> retail -> office -> workshop targets
       d. extra open_space until greenery floor is met
       e. leftover available -> deferred to phase 2
  3. Emit actions in dependency-respecting order:
       defer -> set_zoning -> assign_amenity / develop / reserve_open_space
  4. If a curveball fires mid-episode, replan: redevelop affected blocks
     to open_space (frees infra) and re-allocate the lost role onto a
     fresh available block.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from civicflow_env.models import CivicflowAction
from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.server.state import WorldState


USE_TO_ZONE: Dict[str, str] = {
    "housing": "residential",
    "retail": "commercial",
    "office": "commercial",
    "workshop": "industrial",
    "institutional": "civic",
    "park": "open_space",
}


@dataclass
class Slot:
    """The role this block will play in the final plan."""
    block_id: str
    zone: Optional[str]                # None when the slot is a defer
    use: Optional[str]
    amenity: Optional[str]
    defer_phase: Optional[int]


# ---------------------------------------------------------------------- planner

# Amenity co-location preferences: try these existing-use blocks first
# (matches the relaxed verifier rule). Empty list = needs its own block.
AMENITY_HOSTS: Dict[str, List[str]] = {
    "school":  ["institutional"],
    "fire":    ["institutional"],
    "clinic":  ["retail", "office", "institutional"],
    "grocery": ["retail", "office", "institutional"],
    "transit": ["retail", "office", "institutional"],
    "park":    [],  # park needs its own open_space block
}


def _detect_invalidated(world: WorldState) -> Tuple[List, List]:
    """Find blocks whose committed state conflicts with their current flags.

    Returns (developed_violations, zoned_only_violations). Curveballs that
    flip a block to floodrisk/protected after it was already committed land
    here and need to be cleared before re-planning.
    """
    dev_viol, zoned_viol = [], []
    for b in world.blocks.values():
        if b.is_developed:
            if b.is_floodrisk and not b.reserved_open_space:
                dev_viol.append(b)
            elif b.is_protected and b.amenity != "park":
                dev_viol.append(b)
        elif (b.zone not in ("unzoned", "open_space")) and (b.is_floodrisk or b.is_protected):
            # zoned for development but flags now block any compatible use
            zoned_viol.append(b)
    return dev_viol, zoned_viol


def _triage(world: WorldState, skip_ids: set) -> Tuple[list, list, list]:
    """Bucket eligible blocks. Skips already-committed and invalidated blocks."""
    must_be_open, no_road, available = [], [], []
    for b in world.blocks.values():
        if b.block_id in skip_ids:
            continue
        if b.is_developed or b.zone == "open_space" or b.reserved_open_space:
            continue  # already committed
        if b.is_protected or b.is_floodrisk:
            (must_be_open if b.has_road_access else no_road).append(b)
        elif not b.has_road_access:
            no_road.append(b)
        else:
            available.append(b)
    available.sort(key=lambda b: b.water_demand + b.sewer_demand + b.power_demand)
    return must_be_open, no_road, available


def _allocate(world: WorldState, available: list, must_be_open: list,
              no_road: list,
              remaining_uses: Optional[Dict[str, int]] = None,
              pending_amenities: Optional[List[str]] = None,
              preexisting_use_block_ids: Optional[Dict[str, List[str]]] = None,
              already_committed_green: int = 0) -> Dict[str, Slot]:
    """Greedy allocation that exploits amenity co-location.

    `remaining_uses` / `pending_amenities` / `preexisting_use_block_ids` /
    `already_committed_green` let the same routine drive both first-pass
    planning and reactive replanning.
    """
    targets = world.targets
    slots: Dict[str, Slot] = {}
    use_block_ids: Dict[str, List[str]] = {u: [] for u in USE_TO_ZONE}
    if preexisting_use_block_ids:
        for u, ids in preexisting_use_block_ids.items():
            use_block_ids.setdefault(u, []).extend(ids)
    if remaining_uses is None:
        remaining_uses = dict(targets.blocks_by_use)
    if pending_amenities is None:
        pending_amenities = list(targets.required_amenities)

    # (a) park use + park amenity
    needs_park_amenity = "park" in pending_amenities
    park_count = max(remaining_uses.get("park", 0), 1 if needs_park_amenity else 0)
    park_amenity_done = False
    for _ in range(park_count):
        if not available:
            break
        b = available.pop(0)
        slots[b.block_id] = Slot(
            b.block_id, "open_space", "park",
            "park" if needs_park_amenity and not park_amenity_done else None,
            None,
        )
        use_block_ids["park"].append(b.block_id)
        park_amenity_done = needs_park_amenity or park_amenity_done

    # (b) institutional + co-locate school amenity if required
    institutional_count = remaining_uses.get("institutional", 0)
    needs_school = "school" in pending_amenities
    school_done = False
    for _ in range(institutional_count):
        if not available:
            break
        b = available.pop(0)
        slots[b.block_id] = Slot(
            b.block_id, "civic", "institutional",
            "school" if needs_school and not school_done else None,
            None,
        )
        use_block_ids["institutional"].append(b.block_id)
        school_done = needs_school or school_done

    # (c) housing / retail / office / workshop
    for use in ("housing", "retail", "office", "workshop"):
        n = remaining_uses.get(use, 0)
        for _ in range(n):
            if not available:
                break
            b = available.pop(0)
            slots[b.block_id] = Slot(
                b.block_id, USE_TO_ZONE[use], use, None, None,
            )
            use_block_ids[use].append(b.block_id)

    # (d) site remaining amenities — co-locate first, then fall back to a fresh civic block
    pending: List[str] = []
    for am in pending_amenities:
        if am == "park" and park_amenity_done:
            continue
        if am == "school" and school_done:
            continue
        pending.append(am)

    for am in pending:
        placed = False
        for host_use in AMENITY_HOSTS.get(am, []):
            for bid in use_block_ids.get(host_use, []):
                if slots[bid].amenity is None:
                    slots[bid].amenity = am
                    placed = True
                    break
            if placed:
                break
        if placed:
            continue
        if available:
            b = available.pop(0)
            slots[b.block_id] = Slot(
                b.block_id, "civic", "institutional", am, None,
            )
            use_block_ids["institutional"].append(b.block_id)

    # (e) greenery floor — fill from must_be_open, then no_road, then leftover available
    needed_green = math.ceil(targets.min_greenery_ratio * len(world.blocks))
    current_green = sum(1 for s in slots.values()
                        if s.zone == "open_space" or s.use == "park")
    extra = max(0, needed_green - current_green - already_committed_green)

    def _take_for_green(bucket):
        nonlocal extra
        for b in bucket:
            if extra <= 0:
                return
            if b.block_id in slots:
                continue
            slots[b.block_id] = Slot(b.block_id, "open_space", None, None, None)
            extra -= 1

    _take_for_green(must_be_open)
    _take_for_green(no_road)
    while extra > 0 and available:
        b = available.pop(0)
        slots[b.block_id] = Slot(b.block_id, "open_space", None, None, None)
        extra -= 1

    # (f) leftover handling
    for b in must_be_open:
        if b.block_id not in slots:
            slots[b.block_id] = Slot(b.block_id, "open_space", None, None, None)
    for b in no_road:
        if b.block_id not in slots:
            slots[b.block_id] = Slot(b.block_id, None, None, None, defer_phase=1)
    for b in available:
        if b.block_id not in slots:
            slots[b.block_id] = Slot(b.block_id, None, None, None, defer_phase=2)

    return slots


def _emit(slots: Dict[str, Slot]) -> List[CivicflowAction]:
    """Turn slots into a dependency-respecting action sequence."""
    actions: List[CivicflowAction] = []

    # 1. defers first (clear deferred blocks out of phase 0)
    for s in slots.values():
        if s.defer_phase is not None:
            actions.append(CivicflowAction(
                action_type="defer", block_id=s.block_id, phase_id=s.defer_phase,
            ))

    # 2. explicit zoning for blocks that need it (residential/commercial/industrial/civic/mixed)
    for s in slots.values():
        if s.defer_phase is not None:
            continue
        if s.zone in ("residential", "commercial", "industrial", "civic", "mixed"):
            actions.append(CivicflowAction(
                action_type="set_zoning", block_id=s.block_id, zone=s.zone,
            ))

    # 3. construction / amenity siting / open-space reservation
    for s in slots.values():
        if s.defer_phase is not None:
            continue
        if s.zone == "open_space":
            if s.amenity == "park":
                actions.append(CivicflowAction(
                    action_type="assign_amenity", block_id=s.block_id, amenity_type="park",
                ))
            elif s.use == "park":
                actions.append(CivicflowAction(
                    action_type="set_zoning", block_id=s.block_id, zone="open_space",
                ))
                actions.append(CivicflowAction(
                    action_type="develop", block_id=s.block_id, use="park",
                ))
            else:
                actions.append(CivicflowAction(
                    action_type="reserve_open_space", block_id=s.block_id,
                ))
        elif s.zone == "civic":
            if s.amenity:
                # assign_amenity develops as institutional internally
                actions.append(CivicflowAction(
                    action_type="assign_amenity", block_id=s.block_id, amenity_type=s.amenity,
                ))
            elif s.use == "institutional":
                actions.append(CivicflowAction(
                    action_type="develop", block_id=s.block_id, use="institutional",
                ))
        else:
            # residential / commercial / industrial / mixed: develop the use,
            # then co-locate the amenity if any (verifier doesn't re-allocate infra)
            if s.use:
                actions.append(CivicflowAction(
                    action_type="develop", block_id=s.block_id, use=s.use,
                ))
            if s.amenity and s.amenity != "park":
                actions.append(CivicflowAction(
                    action_type="assign_amenity", block_id=s.block_id, amenity_type=s.amenity,
                ))
    return actions


def plan(world: WorldState) -> List[CivicflowAction]:
    """Greedy plan from the current `WorldState` to a terminally valid plan."""
    must_be_open, must_defer, available = _triage(world, set())
    slots = _allocate(world, available, must_be_open, must_defer)
    return _emit(slots)


def replan_after_curveball(world: WorldState) -> List[CivicflowAction]:
    """React to a curveball by clearing now-invalid blocks and re-siting them."""
    actions: List[CivicflowAction] = []
    affected = []
    for b in world.blocks.values():
        if not b.is_developed:
            continue
        if b.is_floodrisk and not b.reserved_open_space:
            affected.append(b)
        elif b.is_protected and b.amenity != "park":
            affected.append(b)
    lost_uses: List[str] = []
    lost_amenities: List[str] = []
    for b in affected:
        if b.use:
            lost_uses.append(b.use)
        if b.amenity:
            lost_amenities.append(b.amenity)
        # redevelop with no `use` is legal exactly when the block became
        # floodrisk/protected — clears it and frees infra.
        actions.append(CivicflowAction(
            action_type="redevelop", block_id=b.block_id, use=None,
        ))

    # Find replacement blocks for what we lost
    available = [b for b in world.blocks.values()
                 if not b.is_developed and not b.is_protected and not b.is_floodrisk
                 and b.has_road_access and b.phase == 0]
    available.sort(key=lambda b: b.water_demand + b.sewer_demand + b.power_demand)

    for use in lost_uses:
        if not available:
            break
        b = available.pop(0)
        zone = USE_TO_ZONE.get(use)
        if zone in ("residential", "commercial", "industrial", "civic"):
            actions.append(CivicflowAction(
                action_type="set_zoning", block_id=b.block_id, zone=zone,
            ))
            actions.append(CivicflowAction(
                action_type="develop", block_id=b.block_id, use=use,
            ))
        elif zone == "open_space":
            actions.append(CivicflowAction(
                action_type="set_zoning", block_id=b.block_id, zone="open_space",
            ))
            actions.append(CivicflowAction(
                action_type="develop", block_id=b.block_id, use="park",
            ))
    for am in lost_amenities:
        if not available:
            break
        b = available.pop(0)
        if am == "park":
            actions.append(CivicflowAction(
                action_type="assign_amenity", block_id=b.block_id, amenity_type="park",
            ))
        else:
            actions.append(CivicflowAction(
                action_type="set_zoning", block_id=b.block_id, zone="civic",
            ))
            actions.append(CivicflowAction(
                action_type="assign_amenity", block_id=b.block_id, amenity_type=am,
            ))
    return actions


# ---------------------------------------------------------------------- runner

def run_episode(task_id: str, verbose: bool = False) -> Dict:
    os.environ["CIVICFLOW_TASK_ID"] = task_id
    env = CivicflowEnvironment()
    obs = env.reset()
    world = env._world
    actions = plan(world)

    total_reward = 0.0
    illegal_count = 0
    replans = 0
    fires_seen = 0
    last_obs = obs
    i = 0

    while i < len(actions) and not env._done:
        a = actions[i]
        last_obs = env.step(a)
        total_reward += last_obs.reward
        if last_obs.last_metrics["illegal_action"]:
            illegal_count += 1
        # Detect any newly-fired curveball (the env may fire several across
        # the episode; replan once per new fire).
        currently_fired = sum(1 for f in world.curveballs_fired if f)
        if currently_fired > fires_seen:
            fires_seen = currently_fired
            replans += 1
            extra = replan_after_curveball(world)
            actions = actions[:i + 1] + extra + actions[i + 1:]
        if verbose:
            payload = {k: v for k, v in a.model_dump(exclude_none=True).items()
                       if k != "action_type"}
            print(f"  s{last_obs.step_index:>2}: r={last_obs.reward:+.3f} "
                  f"prog={last_obs.last_metrics['progress_score']:.3f} "
                  f"viol={last_obs.last_metrics['constraint_violations']} | "
                  f"{a.action_type}({payload}) -> {last_obs.briefing}")
        i += 1

    progress_final = float(last_obs.last_metrics["progress_score"])
    steps_taken = int(last_obs.step_index)
    avg_step_reward = (total_reward / steps_taken) if steps_taken > 0 else 0.0

    return {
        "task_id": world.task_id,
        "steps_taken": steps_taken,
        "actions_planned": len(actions),
        "total_reward": round(total_reward, 4),
        "avg_step_reward": round(avg_step_reward, 6),  # print/debug metric only
        "final_valid_plan": int(last_obs.last_metrics["final_valid_plan"]),
        "progress_final": progress_final,
        "completion_percent": round(progress_final * 100.0, 2),  # print/debug metric only
        "constraint_violations_final": last_obs.last_metrics["constraint_violations"],
        "illegal_action_count": illegal_count,
        "affected_set_precision": last_obs.last_metrics["affected_set_precision"],
        "affected_set_recall": last_obs.last_metrics["affected_set_recall"],
        "unnecessary_change_count": last_obs.last_metrics["unnecessary_change_count"],
        "done": env._done,
        "curveballs_fired": fires_seen,
        "replans_run": replans,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", help="tiny_a | tiny_b | tiny_c | all")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    if args.task == "all":
        from civicflow_env.tasks import list_task_ids
        task_ids = list_task_ids()
    else:
        task_ids = [args.task]

    results = []
    for tid in task_ids:
        if args.verbose:
            print(f"\n=== {tid} ===")
        results.append(run_episode(tid, verbose=args.verbose))

    print(f"\n{'task':<10} {'steps':>5} {'reward':>9} {'avg_r/step':>11} {'valid':>5} "
          f"{'prog':>5} {'comp%':>7} {'viol':>4} {'illegal':>7} {'F1':>6}")
    for r in results:
        p_, rec = r["affected_set_precision"], r["affected_set_recall"]
        f1 = 2 * p_ * rec / (p_ + rec) if (p_ + rec) > 0 else 0.0
        f1_s = f"{f1:.2f}" if (p_ + rec) > 0 else "-"
        print(f"{r['task_id']:<10} {r['steps_taken']:>5} {r['total_reward']:>+9.3f} "
              f"{r['avg_step_reward']:>+11.4f} {r['final_valid_plan']:>5} {r['progress_final']:>5.2f} "
              f"{r['completion_percent']:>6.2f}% "
              f"{r['constraint_violations_final']:>4} {r['illegal_action_count']:>7} {f1_s:>6}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
