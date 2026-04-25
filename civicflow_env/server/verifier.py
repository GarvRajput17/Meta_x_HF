"""
Deterministic verifier for CivicFlow.

Three jobs:
1. `apply_action`: legality-check + (if legal) mutate WorldState; never crash.
2. `compute_metrics`: emit the canonical METRIC_KEYS dict from current state.
3. `compute_reward_components`: emit the canonical REWARD_COMPONENT_KEYS dict.

All logic is pure-deterministic — no time, no randomness, no globals.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .state import (
    AMENITIES,
    INFRA_TYPES,
    USES,
    ZONE_USE_COMPAT,
    ZONES,
    Block,
    InfraZone,
    WorldState,
)


# Canonical key sets — DO NOT add/remove without re-freezing the contract.
METRIC_KEYS = (
    "illegal_action",
    "constraint_violations",
    "infra_overflow_count",
    "amenity_shortfall_count",
    "greenery_shortfall",
    "use_target_shortfall_count",
    "district_service_gap_count",
    "district_green_gap_count",
    "district_mix_gap_count",
    "accessibility_score",
    "land_use_balance_score",
    "district_coverage_score",
    "phase_completion_score",
    "progress_score",
    "affected_set_precision",
    "affected_set_recall",
    "unnecessary_change_count",
    "final_valid_plan",
    "timeout",
)

REWARD_COMPONENT_KEYS = (
    "legality",
    "constraints",
    "accessibility",
    "land_use_balance",
    "district_quality",
    "phase_progress",
    "progress",
    "replanning",
    "revision_discipline",
    "terminal",
)

MAX_INFRA_UPGRADE = 80.0  # anti-spam cap on a single upgrade_infrastructure call


def current_phase_info(state: WorldState) -> Dict[str, str]:
    fired = any(state.curveballs_fired)
    if fired:
        for phase in state.planning_phases:
            if phase["name"] == "replanning":
                return phase
    progress = state.step_count / max(1, state.targets.max_episode_steps)
    ordered = [p for p in state.planning_phases if p["name"] != "replanning"]
    if not ordered:
        return {"name": "base_structure", "objective": ""}
    idx = min(int(progress * len(ordered)), len(ordered) - 1)
    return ordered[idx]


# -------------------------------------------------------------- legality + apply

def _err(msg: str) -> Tuple[bool, str]:
    return False, msg


def apply_action(state: WorldState, action) -> Tuple[bool, str]:
    """Validate `action` against `state` and apply it if legal.

    Returns (was_legal, message). Mutates `state` only on success.
    """
    a = action  # CivicflowAction (pydantic)
    t = a.action_type

    if t == "set_zoning":
        return _apply_set_zoning(state, a)
    if t == "develop":
        return _apply_develop(state, a)
    if t == "reserve_open_space":
        return _apply_reserve_open_space(state, a)
    if t == "upgrade_infrastructure":
        return _apply_upgrade_infrastructure(state, a)
    if t == "assign_amenity":
        return _apply_assign_amenity(state, a)
    if t == "redevelop":
        return _apply_redevelop(state, a)
    if t == "defer":
        return _apply_defer(state, a)
    return _err(f"unknown action_type '{t}'")


def _get_block(state: WorldState, block_id) -> Tuple[bool, Any]:
    if not block_id or block_id not in state.blocks:
        return False, f"unknown block_id '{block_id}'"
    return True, state.blocks[block_id]


def _apply_set_zoning(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if not a.zone or a.zone not in ZONES or a.zone == "unzoned":
        return _err(f"invalid zone '{a.zone}'")
    if b.is_protected and a.zone != "open_space":
        return _err(f"block {b.block_id} is protected; only open_space zoning allowed")
    if b.is_developed:
        return _err(f"block {b.block_id} is developed; use redevelop")
    b.zone = a.zone
    if a.zone == "open_space":
        b.reserved_open_space = True
    return True, f"zoned {b.block_id} as {a.zone}"


def _apply_develop(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if not a.use or a.use not in USES:
        return _err(f"invalid use '{a.use}'")
    if b.is_developed:
        return _err(f"{b.block_id} already developed; use redevelop")
    if b.is_protected:
        return _err(f"{b.block_id} is protected")
    if b.is_floodrisk:
        return _err(f"{b.block_id} is flood-risk; reserve_open_space instead")
    if not b.has_road_access:
        return _err(f"{b.block_id} has no road access")
    if b.zone == "unzoned":
        return _err(f"{b.block_id} unzoned; set_zoning first")
    if b.phase != 0:
        return _err(f"{b.block_id} deferred to phase {b.phase}")
    if a.use not in ZONE_USE_COMPAT.get(b.zone, set()):
        return _err(f"use '{a.use}' incompatible with zone '{b.zone}'")
    iz = state.infra_zones[b.infra_zone]
    if (iz.water_alloc + b.water_demand > iz.water_capacity
            or iz.sewer_alloc + b.sewer_demand > iz.sewer_capacity
            or iz.power_alloc + b.power_demand > iz.power_capacity):
        return _err(f"infra zone {iz.infra_zone_id} would overflow")
    iz.water_alloc += b.water_demand
    iz.sewer_alloc += b.sewer_demand
    iz.power_alloc += b.power_demand
    b.use = a.use
    return True, f"developed {b.block_id} as {a.use}"


def _apply_reserve_open_space(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if b.is_developed and b.use != "park":
        return _err(f"{b.block_id} developed; redevelop first")
    b.zone = "open_space"
    b.reserved_open_space = True
    b.use = None
    b.amenity = None
    return True, f"reserved {b.block_id} as open space"


def _apply_upgrade_infrastructure(state: WorldState, a) -> Tuple[bool, str]:
    if not a.infra_zone or a.infra_zone not in state.infra_zones:
        return _err(f"unknown infra_zone '{a.infra_zone}'")
    if a.infra_type not in INFRA_TYPES:
        return _err(f"invalid infra_type '{a.infra_type}'")
    if a.capacity is None or a.capacity <= 0 or a.capacity > MAX_INFRA_UPGRADE:
        return _err(f"capacity must be in (0, {MAX_INFRA_UPGRADE}]")
    iz = state.infra_zones[a.infra_zone]
    setattr(iz, f"{a.infra_type}_capacity",
            getattr(iz, f"{a.infra_type}_capacity") + float(a.capacity))
    return True, f"upgraded {iz.infra_zone_id} {a.infra_type} by +{a.capacity}"


# Amenity-to-zone compatibility:
# - park       : own block, open_space (or civic-as-park)
# - school/fire: institutional anchor — civic or mixed
# - clinic/grocery/transit: civic, mixed, OR commercial co-location
#   (a clinic on the corner of a retail block, a transit stop beside an office)
_AMENITY_INSTITUTIONAL_ONLY = {"school", "fire"}


def _apply_assign_amenity(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if a.amenity_type not in AMENITIES:
        return _err(f"invalid amenity_type '{a.amenity_type}'")
    if not b.has_road_access:
        return _err(f"{b.block_id} has no road access")
    if b.is_protected and a.amenity_type != "park":
        return _err(f"{b.block_id} protected; only 'park' amenity allowed")
    if b.amenity is not None:
        return _err(f"{b.block_id} already has amenity '{b.amenity}'")

    if a.amenity_type == "park":
        if b.zone not in ("open_space", "civic", "unzoned"):
            return _err("park requires open_space/civic zoning")
        if b.is_developed and b.use != "park":
            return _err(f"{b.block_id} developed with non-park use")
        b.zone = "open_space"
        b.use = "park"
        b.reserved_open_space = True
        b.amenity = "park"
        return True, f"assigned amenity park to {b.block_id}"

    institutional_only = a.amenity_type in _AMENITY_INSTITUTIONAL_ONLY
    allowed_zones = ("civic", "mixed") if institutional_only else ("civic", "mixed", "commercial")
    if b.zone not in allowed_zones:
        return _err(f"amenity '{a.amenity_type}' requires zoning in {allowed_zones}")

    if b.is_developed:
        # Co-locate with existing use; do not change use, do not re-allocate infra.
        if institutional_only and b.use != "institutional":
            return _err(f"school/fire need institutional anchor on {b.block_id}")
        b.amenity = a.amenity_type
        return True, f"co-located amenity {a.amenity_type} on {b.block_id}"

    # Greenfield: develop as institutional anchor and stamp the amenity.
    iz = state.infra_zones[b.infra_zone]
    if (iz.water_alloc + b.water_demand > iz.water_capacity
            or iz.sewer_alloc + b.sewer_demand > iz.sewer_capacity
            or iz.power_alloc + b.power_demand > iz.power_capacity):
        return _err(f"infra zone {iz.infra_zone_id} would overflow")
    iz.water_alloc += b.water_demand
    iz.sewer_alloc += b.sewer_demand
    iz.power_alloc += b.power_demand
    b.use = "institutional"
    b.amenity = a.amenity_type
    return True, f"assigned amenity {a.amenity_type} to {b.block_id}"


def _apply_redevelop(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if not b.is_developed:
        return _err(f"{b.block_id} not developed; use develop")
    if a.use is None and not b.is_floodrisk and not b.is_protected:
        return _err("redevelop requires new 'use' (or use reserve_open_space)")
    iz = state.infra_zones[b.infra_zone]
    iz.water_alloc -= b.water_demand
    iz.sewer_alloc -= b.sewer_demand
    iz.power_alloc -= b.power_demand
    b.use = None
    b.amenity = None
    if a.use is None or b.is_floodrisk or b.is_protected:
        b.zone = "open_space"
        b.reserved_open_space = True
        return True, f"cleared {b.block_id} (now open_space)"
    if a.use not in ZONE_USE_COMPAT.get(b.zone, set()):
        return _err(f"use '{a.use}' incompatible with zone '{b.zone}'")
    if (iz.water_alloc + b.water_demand > iz.water_capacity
            or iz.sewer_alloc + b.sewer_demand > iz.sewer_capacity
            or iz.power_alloc + b.power_demand > iz.power_capacity):
        return _err(f"infra zone {iz.infra_zone_id} would overflow on redevelop")
    iz.water_alloc += b.water_demand
    iz.sewer_alloc += b.sewer_demand
    iz.power_alloc += b.power_demand
    b.use = a.use
    return True, f"redeveloped {b.block_id} as {a.use}"


def _apply_defer(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if a.phase_id is None or a.phase_id not in (1, 2):
        return _err("phase_id must be 1 or 2")
    if b.is_developed:
        return _err(f"{b.block_id} already developed; cannot defer")
    b.phase = int(a.phase_id)
    return True, f"deferred {b.block_id} to phase {a.phase_id}"


# -------------------------------------------------------------- curveball

def maybe_fire_curveballs(state: WorldState) -> List[int]:
    """Fire any curveballs whose fire_at_step has been reached.

    Returns the list of indices fired this step (empty if none).
    """
    fired_now: List[int] = []
    for i, cb in enumerate(state.curveballs):
        if state.curveballs_fired[i]:
            continue
        if state.step_count < cb.fire_at_step:
            continue
        for block_id in cb.block_ids:
            b = state.blocks.get(block_id)
            if b is None:
                continue
            if cb.mutation == "protect":
                b.is_protected = True
            elif cb.mutation == "floodrisk":
                b.is_floodrisk = True
        if cb.mutation == "capacity_cut" and cb.infra_zone in state.infra_zones and cb.infra_type:
            iz = state.infra_zones[cb.infra_zone]
            attr = f"{cb.infra_type}_capacity"
            setattr(iz, attr, max(0.0, getattr(iz, attr) + cb.capacity_delta))
        if cb.target_overrides:
            if "min_greenery_ratio" in cb.target_overrides:
                state.targets.min_greenery_ratio = float(cb.target_overrides["min_greenery_ratio"])
            if "required_amenities" in cb.target_overrides:
                state.targets.required_amenities = list(cb.target_overrides["required_amenities"])
            if "district_targets" in cb.target_overrides:
                for district, patch in cb.target_overrides["district_targets"].items():
                    base = dict(state.targets.district_targets.get(district, {}))
                    base.update(patch)
                    state.targets.district_targets[district] = base
        state.curveballs_fired[i] = True
        fired_now.append(i)
        if state.first_fire_step is None:
            state.first_fire_step = state.step_count
    return fired_now


# Legacy single-curveball entry point (kept so older callers don't break).
def maybe_fire_curveball(state: WorldState) -> bool:
    return bool(maybe_fire_curveballs(state))


# -------------------------------------------------------------- metrics

def _count_uses(state: WorldState) -> Dict[str, int]:
    counts: Dict[str, int] = {u: 0 for u in USES}
    for b in state.blocks.values():
        if b.use:
            counts[b.use] = counts.get(b.use, 0) + 1
    return counts


def _amenity_coverage(state: WorldState) -> Dict[str, int]:
    cov: Dict[str, int] = {a: 0 for a in AMENITIES}
    for b in state.blocks.values():
        if b.amenity:
            cov[b.amenity] = cov.get(b.amenity, 0) + 1
    # parks tagged via use also count
    for b in state.blocks.values():
        if b.use == "park" and b.amenity != "park":
            cov["park"] = cov.get("park", 0) + 1
    return cov


def _greenery_ratio(state: WorldState) -> float:
    n = len(state.blocks)
    if n == 0:
        return 0.0
    green = sum(1 for b in state.blocks.values()
                if b.use == "park" or b.reserved_open_space or b.zone == "open_space")
    return green / n


def _district_blocks(state: WorldState) -> Dict[str, List[Block]]:
    out: Dict[str, List[Block]] = {}
    for b in state.blocks.values():
        out.setdefault(b.district, []).append(b)
    return out


def _district_stats(state: WorldState) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for district, blocks in _district_blocks(state).items():
        total = len(blocks)
        uses: Dict[str, int] = {}
        amenities: Dict[str, int] = {}
        green = 0
        economic = 0
        connected_dev = 0
        developed = 0
        for b in blocks:
            if b.use:
                uses[b.use] = uses.get(b.use, 0) + 1
            if b.amenity:
                amenities[b.amenity] = amenities.get(b.amenity, 0) + 1
            if b.use == "park" or b.reserved_open_space or b.zone == "open_space":
                green += 1
            if b.use in {"retail", "office", "workshop"} or b.zone in {"commercial", "mixed", "industrial"}:
                economic += 1
            if b.is_developed:
                developed += 1
                if b.has_road_access:
                    connected_dev += 1
        stats[district] = {
            "total": total,
            "uses": uses,
            "amenities": amenities,
            "greenery_ratio": green / max(1, total),
            "economic_ratio": economic / max(1, total),
            "connected_dev_ratio": connected_dev / max(1, developed) if developed else 1.0,
        }
    return stats


def _infra_overflow(state: WorldState) -> int:
    n = 0
    for iz in state.infra_zones.values():
        n += int(iz.water_alloc > iz.water_capacity)
        n += int(iz.sewer_alloc > iz.sewer_capacity)
        n += int(iz.power_alloc > iz.power_capacity)
    return n


def _violations(state: WorldState) -> int:
    """Number of binding hard-constraint violations right now."""
    v = 0
    v += _infra_overflow(state)
    for b in state.blocks.values():
        if b.is_developed and not b.has_road_access:
            v += 1
        if b.is_developed and b.is_protected and b.amenity != "park":
            v += 1
        if b.is_developed and b.is_floodrisk and not b.reserved_open_space:
            v += 1
        if b.use and b.zone in ZONE_USE_COMPAT and b.use not in ZONE_USE_COMPAT[b.zone]:
            # legality enforces this on entry, but a curveball can break it later
            v += 1
    return v


def _amenity_shortfall(state: WorldState) -> int:
    cov = _amenity_coverage(state)
    return sum(1 for a in state.targets.required_amenities if cov.get(a, 0) == 0)


def _use_target_shortfall(state: WorldState) -> int:
    counts = _count_uses(state)
    return sum(max(0, target - counts.get(use, 0)) for use, target in state.targets.blocks_by_use.items())


def _district_gap_counts(state: WorldState) -> Dict[str, int]:
    district_targets = state.targets.district_targets or {}
    stats = _district_stats(state)
    service_gaps = 0
    green_gaps = 0
    mix_gaps = 0
    for district, target in district_targets.items():
        s = stats.get(district, {})
        amenities = s.get("amenities", {})
        uses = s.get("uses", {})
        for amenity in target.get("required_amenities", []):
            if amenities.get(amenity, 0) == 0:
                service_gaps += 1
        min_green = target.get("min_greenery_ratio")
        if min_green is not None and s.get("greenery_ratio", 0.0) < float(min_green):
            green_gaps += 1
        min_economic = target.get("min_economic_ratio")
        if min_economic is not None and s.get("economic_ratio", 0.0) < float(min_economic):
            mix_gaps += 1
        required_uses = target.get("required_uses", {})
        for use, req in required_uses.items():
            if uses.get(use, 0) < int(req):
                mix_gaps += 1
    return {
        "district_service_gap_count": service_gaps,
        "district_green_gap_count": green_gaps,
        "district_mix_gap_count": mix_gaps,
    }


def _accessibility_score(state: WorldState) -> float:
    district_targets = state.targets.district_targets or {}
    if not district_targets:
        # Tiny tiers without district targets fall back to global amenity coverage.
        cov = _amenity_coverage(state)
        vals = [1.0 if cov.get(a, 0) > 0 else 0.0 for a in state.targets.required_amenities]
        return sum(vals) / max(1, len(vals))
    stats = _district_stats(state)
    achieved = 0.0
    total = 0
    for district, target in district_targets.items():
        amenities = stats.get(district, {}).get("amenities", {})
        for amenity in target.get("required_amenities", []):
            total += 1
            achieved += 1.0 if amenities.get(amenity, 0) > 0 else 0.0
    return achieved / max(1, total)


def _land_use_balance_score(state: WorldState) -> float:
    counts = _count_uses(state)
    vals: List[float] = []
    for use, target in state.targets.blocks_by_use.items():
        if target <= 0:
            continue
        vals.append(min(counts.get(use, 0), target) / target)
    return sum(vals) / max(1, len(vals))


def _district_coverage_score(state: WorldState) -> float:
    district_targets = state.targets.district_targets or {}
    if not district_targets:
        return 1.0
    stats = _district_stats(state)
    vals: List[float] = []
    for district, target in district_targets.items():
        s = stats.get(district, {})
        amenities = s.get("amenities", {})
        uses = s.get("uses", {})
        checks: List[float] = []
        for amenity in target.get("required_amenities", []):
            checks.append(1.0 if amenities.get(amenity, 0) > 0 else 0.0)
        min_green = target.get("min_greenery_ratio")
        if min_green is not None:
            checks.append(1.0 if s.get("greenery_ratio", 0.0) >= float(min_green) else 0.0)
        min_economic = target.get("min_economic_ratio")
        if min_economic is not None:
            checks.append(1.0 if s.get("economic_ratio", 0.0) >= float(min_economic) else 0.0)
        required_uses = target.get("required_uses", {})
        for use, req in required_uses.items():
            checks.append(1.0 if uses.get(use, 0) >= int(req) else 0.0)
        vals.append(sum(checks) / max(1, len(checks)))
    return sum(vals) / max(1, len(vals))


def _phase_completion_score(state: WorldState) -> float:
    phase = current_phase_info(state)["name"]
    if phase == "base_structure":
        return 0.5 * _land_use_balance_score(state) + 0.5 * (1.0 - min(1.0, _violations(state) / 5.0))
    if phase == "services_and_access":
        return 0.6 * _accessibility_score(state) + 0.4 * _district_coverage_score(state)
    if phase == "infrastructure_balance":
        infra_ok = 1.0 if _infra_overflow(state) == 0 else 0.0
        green_ok = 1.0 if _greenery_ratio(state) >= state.targets.min_greenery_ratio else 0.0
        return (infra_ok + green_ok + _district_coverage_score(state)) / 3.0
    if phase == "replanning":
        fired_indices = [i for i, f in enumerate(state.curveballs_fired) if f]
        gold: set = set()
        for i in fired_indices:
            gold.update(state.curveballs[i].gold_affected)
        touched = set(state.blocks_touched_after_curveball)
        precision = (len(touched & gold) / len(touched)) if touched else 0.0
        recall = (len(touched & gold) / len(gold)) if gold else 0.0
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall / (precision + recall))
    return 1.0 if _final_valid(state) else _progress(state)


def _progress(state: WorldState) -> float:
    """0..1 fraction of global + district planning objectives met."""
    t = state.targets
    counts = _count_uses(state)
    items: List[float] = []
    for use, target in t.blocks_by_use.items():
        if target == 0:
            items.append(1.0)
        else:
            items.append(min(counts.get(use, 0), target) / target)
    cov = _amenity_coverage(state)
    for a in t.required_amenities:
        items.append(1.0 if cov.get(a, 0) > 0 else 0.0)
    items.append(1.0 if _greenery_ratio(state) >= t.min_greenery_ratio else 0.0)
    if t.district_targets:
        items.append(_accessibility_score(state))
        items.append(_district_coverage_score(state))
        items.append(_land_use_balance_score(state))
    return sum(items) / max(1, len(items))


def _final_valid(state: WorldState) -> bool:
    return _progress(state) >= 0.999 and _violations(state) == 0


def compute_metrics(state: WorldState, last_action_legal: bool, last_action_was_mutator: bool,
                    timeout: bool) -> Dict[str, Any]:
    affected_p = 0.0
    affected_r = 0.0
    unnecessary = 0
    fired_indices = [i for i, f in enumerate(state.curveballs_fired) if f]
    if fired_indices:
        gold: set = set()
        for i in fired_indices:
            gold.update(state.curveballs[i].gold_affected)
        touched = set(state.blocks_touched_after_curveball)
        if touched:
            affected_p = len(touched & gold) / len(touched)
        if gold:
            affected_r = len(touched & gold) / len(gold)
        unnecessary = len(touched - gold)

    district_gaps = _district_gap_counts(state)
    accessibility_score = _accessibility_score(state)
    land_use_balance_score = _land_use_balance_score(state)
    district_coverage_score = _district_coverage_score(state)
    phase_completion_score = _phase_completion_score(state)

    return {
        "illegal_action": int(not last_action_legal),
        "constraint_violations": _violations(state),
        "infra_overflow_count": _infra_overflow(state),
        "amenity_shortfall_count": _amenity_shortfall(state),
        "greenery_shortfall": int(_greenery_ratio(state) < state.targets.min_greenery_ratio),
        "use_target_shortfall_count": _use_target_shortfall(state),
        "district_service_gap_count": district_gaps["district_service_gap_count"],
        "district_green_gap_count": district_gaps["district_green_gap_count"],
        "district_mix_gap_count": district_gaps["district_mix_gap_count"],
        "accessibility_score": round(accessibility_score, 4),
        "land_use_balance_score": round(land_use_balance_score, 4),
        "district_coverage_score": round(district_coverage_score, 4),
        "phase_completion_score": round(phase_completion_score, 4),
        "progress_score": round(_progress(state), 4),
        "affected_set_precision": round(affected_p, 4),
        "affected_set_recall": round(affected_r, 4),
        "unnecessary_change_count": unnecessary,
        "final_valid_plan": int(_final_valid(state)),
        "timeout": int(timeout),
    }


def compute_reward_components(metrics: Dict[str, Any], prev_metrics: Dict[str, float],
                              done: bool) -> Dict[str, float]:
    """Per-step decomposed reward.

    Shaping kept minimal and stable: the dominant signal is `terminal`. Other
    components nudge the policy toward legality, constraint satisfaction,
    measurable progress, scoped replanning, and revision discipline.
    """
    legality = -1.0 if metrics["illegal_action"] else 0.0
    constraints = -0.5 * metrics["constraint_violations"]
    accessibility = 0.5 * max(0.0, metrics["accessibility_score"] - prev_metrics.get("accessibility_score", 0.0))
    land_use_balance = 0.5 * max(0.0, metrics["land_use_balance_score"] - prev_metrics.get("land_use_balance_score", 0.0))
    district_quality = 0.5 * max(0.0, metrics["district_coverage_score"] - prev_metrics.get("district_coverage_score", 0.0))
    phase_progress = 0.5 * max(0.0, metrics["phase_completion_score"] - prev_metrics.get("phase_completion_score", 0.0))
    progress = max(0.0, metrics["progress_score"] - prev_metrics.get("progress_score", 0.0))
    f1 = 0.0
    p, r = metrics["affected_set_precision"], metrics["affected_set_recall"]
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    replanning = f1 if done else 0.0
    revision_discipline = -0.25 * metrics["unnecessary_change_count"] if done else 0.0
    terminal = 5.0 if (done and metrics["final_valid_plan"]) else 0.0
    return {
        "legality": legality,
        "constraints": constraints,
        "accessibility": round(accessibility, 4),
        "land_use_balance": round(land_use_balance, 4),
        "district_quality": round(district_quality, 4),
        "phase_progress": round(phase_progress, 4),
        "progress": round(progress, 4),
        "replanning": round(replanning, 4),
        "revision_discipline": revision_discipline,
        "terminal": terminal,
    }


# -------------------------------------------------------------- summaries

def planning_summary(state: WorldState) -> Dict[str, Any]:
    blocks_by_zone: Dict[str, int] = {}
    blocks_by_use: Dict[str, int] = {"undeveloped": 0}
    for b in state.blocks.values():
        blocks_by_zone[b.zone] = blocks_by_zone.get(b.zone, 0) + 1
        if b.use:
            blocks_by_use[b.use] = blocks_by_use.get(b.use, 0) + 1
        else:
            blocks_by_use["undeveloped"] += 1
    infra = {
        z.infra_zone_id: {
            "water": [round(z.water_alloc, 1), round(z.water_capacity, 1)],
            "sewer": [round(z.sewer_alloc, 1), round(z.sewer_capacity, 1)],
            "power": [round(z.power_alloc, 1), round(z.power_capacity, 1)],
            "road_capacity": round(z.road_capacity, 1),
        }
        for z in state.infra_zones.values()
    }
    return {
        "blocks_total": len(state.blocks),
        "blocks_by_zone": blocks_by_zone,
        "blocks_by_use": blocks_by_use,
        "infra_zones": infra,
        "amenity_coverage": _amenity_coverage(state),
        "district_stats": _district_stats(state),
        "greenery_ratio": round(_greenery_ratio(state), 3),
        "targets": {
            "blocks_by_use": dict(state.targets.blocks_by_use),
            "required_amenities": list(state.targets.required_amenities),
            "min_greenery_ratio": state.targets.min_greenery_ratio,
            "district_targets": dict(state.targets.district_targets),
        },
        "current_phase": current_phase_info(state),
    }


def legal_actions_summary(state: WorldState) -> Dict[str, Any]:
    """Compact hints — not exhaustive. Trainer code uses this for prompt context."""
    s = {
        "set_zoning": [], "develop": [], "reserve_open_space": [],
        "upgrade_infrastructure": list(state.infra_zones.keys()),
        "assign_amenity": [], "redevelop": [], "defer": [],
        # Backwards-compatible detail for amenity co-location hosts.
        "assign_amenity_by_type": {
            "park": [],
            "school": [],
            "clinic": [],
            "grocery": [],
            "fire": [],
            "transit": [],
        },
    }
    for b in state.blocks.values():
        if not b.is_developed and b.zone == "unzoned" and not b.is_protected:
            s["set_zoning"].append(b.block_id)
        if (not b.is_developed and b.has_road_access and not b.is_protected
                and not b.is_floodrisk and b.zone not in ("unzoned", "open_space") and b.phase == 0):
            s["develop"].append(b.block_id)
        if not b.is_developed:
            s["reserve_open_space"].append(b.block_id)
        if b.has_road_access and b.zone in ("civic", "mixed", "commercial", "open_space", "unzoned"):
            s["assign_amenity"].append(b.block_id)
            if b.zone in ("open_space", "civic", "unzoned"):
                s["assign_amenity_by_type"]["park"].append(b.block_id)
            if b.zone in ("civic", "mixed", "unzoned"):
                s["assign_amenity_by_type"]["school"].append(b.block_id)
                s["assign_amenity_by_type"]["fire"].append(b.block_id)
            if b.zone in ("civic", "mixed", "commercial", "unzoned"):
                s["assign_amenity_by_type"]["clinic"].append(b.block_id)
                s["assign_amenity_by_type"]["grocery"].append(b.block_id)
                s["assign_amenity_by_type"]["transit"].append(b.block_id)
        if b.is_developed:
            s["redevelop"].append(b.block_id)
        if not b.is_developed:
            s["defer"].append(b.block_id)
    return s


def active_constraints(state: WorldState) -> List[str]:
    out = ["road_access", "infra_capacity", "zone_use_compat", "protected_blocks",
           "floodrisk_blocks", "min_greenery_ratio", "required_amenities",
           "district_access", "district_mix", "district_greenery"]
    if state.curveballs:
        out.append("curveball_replanning")
    return out
