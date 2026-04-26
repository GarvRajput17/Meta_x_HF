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
from collections import deque
from typing import Any, Dict, List, Tuple

from .state import (
    AMENITIES,
    DEFAULT_ACTION_COSTS,
    DESIGNATION_ALLOWED_AMENITIES,
    DESIGNATION_ALLOWED_USES,
    FUTURE_DESIGNATIONS,
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
    "spatial_service_gap_count",
    "designation_violation_count",
    "repeated_action_count",
    "no_progress_step",
    "open_space_oversupply_count",
    "city_resource_overflow_count",
    "population_total",
    "population_served_ratio",
    "budget_remaining_ratio",
    "accessibility_score",
    "spatial_service_score",
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


def _budget_cost(state: WorldState, action_type: str) -> float:
    costs = state.city_resources.action_costs or DEFAULT_ACTION_COSTS
    return float(costs.get(action_type, DEFAULT_ACTION_COSTS.get(action_type, 0.0)))


def _check_budget(state: WorldState, action_type: str) -> Tuple[bool, str]:
    cr = state.city_resources
    if cr.construction_budget <= 0:
        return True, ""
    cost = _budget_cost(state, action_type)
    if cost > 0 and cr.construction_budget < cost:
        return False, f"budget exhausted: need {cost:.0f}, remaining {cr.construction_budget:.0f}"
    return True, ""


def _deduct_budget(state: WorldState, action_type: str) -> None:
    cr = state.city_resources
    if cr.construction_budget <= 0:
        return
    cr.construction_budget -= _budget_cost(state, action_type)


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
    if b.zone == a.zone:
        return _err(f"block {b.block_id} is already zoned as {a.zone}")
    if b.is_protected and a.zone != "open_space":
        return _err(f"block {b.block_id} is protected; only open_space zoning allowed")
    if b.is_developed:
        return _err(f"block {b.block_id} is developed; use redevelop")
    ok2, msg = _check_budget(state, "set_zoning")
    if not ok2:
        return _err(msg)
    b.zone = a.zone
    if a.zone == "open_space":
        b.reserved_open_space = True
    _deduct_budget(state, "set_zoning")
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
    if not b.has_road_access:
        return _err(f"{b.block_id} has no road access")
    if b.zone == "unzoned":
        return _err(f"{b.block_id} unzoned; set_zoning first")
    if b.phase != 0:
        return _err(f"{b.block_id} deferred to phase {b.phase}")
    if a.use not in ZONE_USE_COMPAT.get(b.zone, set()):
        return _err(f"use '{a.use}' incompatible with zone '{b.zone}'")
    if b.future_designation:
        allowed = DESIGNATION_ALLOWED_USES.get(b.future_designation, set())
        if a.use not in allowed:
            return _err(
                f"{b.block_id} is designated '{b.future_designation}'; "
                f"only uses {allowed or '{none}'} are permitted"
            )
    iz = state.infra_zones[b.infra_zone]
    if (iz.water_alloc + b.water_demand > iz.water_capacity
            or iz.sewer_alloc + b.sewer_demand > iz.sewer_capacity
            or iz.power_alloc + b.power_demand > iz.power_capacity
            or iz.road_alloc + b.road_demand > iz.road_capacity):
        return _err(f"infra zone {iz.infra_zone_id} would overflow")
    ok2, msg = _check_budget(state, "develop")
    if not ok2:
        return _err(msg)
    iz.water_alloc += b.water_demand
    iz.sewer_alloc += b.sewer_demand
    iz.power_alloc += b.power_demand
    iz.road_alloc += b.road_demand
    b.use = a.use
    _deduct_budget(state, "develop")
    return True, f"developed {b.block_id} as {a.use}"


def _apply_reserve_open_space(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if b.reserved_open_space:
        return _err(f"{b.block_id} already reserved as open space")
    if b.is_developed and b.use != "park":
        return _err(f"{b.block_id} developed; redevelop first")
    ok2, msg = _check_budget(state, "reserve_open_space")
    if not ok2:
        return _err(msg)
    b.zone = "open_space"
    b.reserved_open_space = True
    b.use = None
    b.amenity = None
    _deduct_budget(state, "reserve_open_space")
    return True, f"reserved {b.block_id} as open space"


def _apply_upgrade_infrastructure(state: WorldState, a) -> Tuple[bool, str]:
    if not a.infra_zone or a.infra_zone not in state.infra_zones:
        return _err(f"unknown infra_zone '{a.infra_zone}'")
    if a.infra_type not in INFRA_TYPES:
        return _err(f"invalid infra_type '{a.infra_type}'")
    if a.capacity is None or a.capacity <= 0 or a.capacity > MAX_INFRA_UPGRADE:
        return _err(f"capacity must be in (0, {MAX_INFRA_UPGRADE}]")
    ok2, msg = _check_budget(state, "upgrade_infrastructure")
    if not ok2:
        return _err(msg)
    iz = state.infra_zones[a.infra_zone]
    setattr(iz, f"{a.infra_type}_capacity",
            getattr(iz, f"{a.infra_type}_capacity") + float(a.capacity))
    _deduct_budget(state, "upgrade_infrastructure")
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
    if b.future_designation:
        allowed = DESIGNATION_ALLOWED_AMENITIES.get(b.future_designation, set())
        if a.amenity_type not in allowed:
            return _err(
                f"{b.block_id} is designated '{b.future_designation}'; "
                f"only amenities {allowed} are permitted"
            )
    ok2, msg = _check_budget(state, "assign_amenity")
    if not ok2:
        return _err(msg)

    if a.amenity_type == "park":
        if b.zone not in ("open_space", "civic", "unzoned"):
            return _err("park requires open_space/civic zoning")
        if b.is_developed and b.use != "park":
            return _err(f"{b.block_id} developed with non-park use")
        b.zone = "open_space"
        b.use = "park"
        b.reserved_open_space = True
        b.amenity = "park"
        _deduct_budget(state, "assign_amenity")
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
        _deduct_budget(state, "assign_amenity")
        return True, f"co-located amenity {a.amenity_type} on {b.block_id}"

    # Greenfield: develop as institutional anchor and stamp the amenity.
    iz = state.infra_zones[b.infra_zone]
    if (iz.water_alloc + b.water_demand > iz.water_capacity
            or iz.sewer_alloc + b.sewer_demand > iz.sewer_capacity
            or iz.power_alloc + b.power_demand > iz.power_capacity
            or iz.road_alloc + b.road_demand > iz.road_capacity):
        return _err(f"infra zone {iz.infra_zone_id} would overflow")
    iz.water_alloc += b.water_demand
    iz.sewer_alloc += b.sewer_demand
    iz.power_alloc += b.power_demand
    iz.road_alloc += b.road_demand
    b.use = "institutional"
    b.amenity = a.amenity_type
    _deduct_budget(state, "assign_amenity")
    return True, f"assigned amenity {a.amenity_type} to {b.block_id}"


def _apply_redevelop(state: WorldState, a) -> Tuple[bool, str]:
    ok, b = _get_block(state, a.block_id)
    if not ok:
        return _err(b)
    if not b.is_developed:
        return _err(f"{b.block_id} not developed; use develop")
    if a.use is None and not b.is_protected:
        return _err("redevelop requires new 'use' (or use reserve_open_space)")
    ok2, msg = _check_budget(state, "redevelop")
    if not ok2:
        return _err(msg)
    iz = state.infra_zones[b.infra_zone]
    iz.water_alloc -= b.water_demand
    iz.sewer_alloc -= b.sewer_demand
    iz.power_alloc -= b.power_demand
    iz.road_alloc -= b.road_demand
    b.use = None
    b.amenity = None
    if a.use is None or b.is_protected:
        b.zone = "open_space"
        b.reserved_open_space = True
        _deduct_budget(state, "redevelop")
        return True, f"cleared {b.block_id} (now open_space)"
    if a.use not in ZONE_USE_COMPAT.get(b.zone, set()):
        return _err(f"use '{a.use}' incompatible with zone '{b.zone}'")
    if (iz.water_alloc + b.water_demand > iz.water_capacity
            or iz.sewer_alloc + b.sewer_demand > iz.sewer_capacity
            or iz.power_alloc + b.power_demand > iz.power_capacity
            or iz.road_alloc + b.road_demand > iz.road_capacity):
        return _err(f"infra zone {iz.infra_zone_id} would overflow on redevelop")
    iz.water_alloc += b.water_demand
    iz.sewer_alloc += b.sewer_demand
    iz.power_alloc += b.power_demand
    iz.road_alloc += b.road_demand
    b.use = a.use
    _deduct_budget(state, "redevelop")
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
            elif cb.mutation == "designate" and cb.designation_type:
                b.future_designation = cb.designation_type
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


def _normalise_action(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in action.items()
        if k not in {"metadata", "justification"} and v is not None
    }


def _repeated_action_count(state: WorldState) -> int:
    if len(state.action_history) < 2:
        return 0
    return int(
        _normalise_action(state.action_history[-1])
        == _normalise_action(state.action_history[-2])
    )


def _shortest_hops(state: WorldState, src: str, dst: str, max_radius: int) -> int | None:
    if src == dst:
        return 0
    if not state.adjacency:
        return None
    seen = {src}
    q = deque([(src, 0)])
    while q:
        cur, dist = q.popleft()
        if dist >= max_radius:
            continue
        for nxt in state.adjacency.get(cur, []):
            if nxt in seen:
                continue
            if nxt == dst:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None


def _spatial_service_stats(state: WorldState) -> Dict[str, Any]:
    """Score graph-distance amenity coverage for housing blocks.

    Tasks opt in with targets.service_radius, e.g. {"school": 1, "clinic": 2}.
    Existing tasks without this field keep a perfect neutral score.
    """
    requirements = state.targets.service_radius or {}
    if not requirements:
        return {
            "spatial_service_score": 1.0,
            "spatial_service_gap_count": 0,
            "spatial_service_detail": {},
        }

    housing_blocks = [b for b in state.blocks.values() if b.use == "housing"]
    if not housing_blocks:
        return {
            "spatial_service_score": 0.0,
            "spatial_service_gap_count": len(requirements),
            "spatial_service_detail": {a: {"covered": 0, "total": 0, "radius": r}
                                       for a, r in requirements.items()},
        }

    detail: Dict[str, Any] = {}
    covered_pairs = 0
    total_pairs = 0
    gaps = 0
    for amenity, radius in requirements.items():
        hosts = [
            b.block_id for b in state.blocks.values()
            if b.amenity == amenity or (amenity == "park" and b.use == "park")
        ]
        covered = 0
        for h in housing_blocks:
            total_pairs += 1
            is_covered = any(
                _shortest_hops(state, h.block_id, host_id, int(radius)) is not None
                for host_id in hosts
            )
            if is_covered:
                covered += 1
                covered_pairs += 1
            else:
                gaps += 1
        detail[amenity] = {"covered": covered, "total": len(housing_blocks), "radius": int(radius)}
    score = covered_pairs / max(1, total_pairs)
    return {
        "spatial_service_score": score,
        "spatial_service_gap_count": gaps,
        "spatial_service_detail": detail,
    }


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
        n += int(iz.road_alloc > iz.road_capacity)
    return n


def _city_resource_overflow(state: WorldState) -> int:
    cr = state.city_resources
    n = 0
    total_water = sum(z.water_alloc for z in state.infra_zones.values())
    total_power = sum(z.power_alloc for z in state.infra_zones.values())
    total_sewer = sum(z.sewer_alloc for z in state.infra_zones.values())
    if cr.water_supply > 0 and total_water > cr.water_supply:
        n += 1
    if cr.power_grid > 0 and total_power > cr.power_grid:
        n += 1
    if cr.sewer_network > 0 and total_sewer > cr.sewer_network:
        n += 1
    return n


def _budget_remaining_ratio(state: WorldState) -> float:
    cr = state.city_resources
    if cr.construction_budget <= 0:
        return 1.0
    # Ratio relative to the original budget encoded in external_ledgers for display;
    # here we track remaining / (remaining + spent). We don't store original, so
    # we compare remaining vs the sum of all default costs * block count as a proxy.
    # Simpler: just return remaining / max(1, remaining + spent_estimate).
    # Since we mutate construction_budget in place, store the initial via external_ledger.
    initial = float(state.external_ledgers.get("initial_budget", cr.construction_budget))
    if initial <= 0:
        return 1.0
    return round(min(1.0, cr.construction_budget / initial), 4)


def _population_stats(state: WorldState) -> Dict[str, Any]:
    housing_blocks = [b for b in state.blocks.values() if b.use == "housing"]
    total_pop = sum(b.population_capacity for b in housing_blocks)
    requirements = state.targets.service_radius or {}
    if not requirements or total_pop == 0:
        return {
            "population_total": total_pop,
            "population_served_ratio": 1.0 if total_pop == 0 else 0.0,
        }
    served_pop = 0
    for h in housing_blocks:
        fully_served = all(
            any(
                _shortest_hops(state, h.block_id, host_id, int(radius)) is not None
                for host_id in [
                    b.block_id for b in state.blocks.values()
                    if b.amenity == amenity or (amenity == "park" and b.use == "park")
                ]
            )
            for amenity, radius in requirements.items()
        )
        if fully_served:
            served_pop += h.population_capacity
    return {
        "population_total": total_pop,
        "population_served_ratio": round(served_pop / max(1, total_pop), 4),
    }


def _designation_violations(state: WorldState) -> int:
    """Count blocks whose current use/amenity violates their future_designation."""
    v = 0
    for b in state.blocks.values():
        if not b.future_designation:
            continue
        if b.use is not None:
            allowed_uses = DESIGNATION_ALLOWED_USES.get(b.future_designation, set())
            if b.use not in allowed_uses and b.use != "park":
                v += 1
        if b.amenity is not None:
            allowed_am = DESIGNATION_ALLOWED_AMENITIES.get(b.future_designation, set())
            if b.amenity not in allowed_am:
                v += 1
    return v


def _violations(state: WorldState) -> int:
    """Number of binding hard-constraint violations right now."""
    v = 0
    v += _infra_overflow(state)
    v += _city_resource_overflow(state)
    v += _designation_violations(state)
    for b in state.blocks.values():
        if b.is_developed and not b.has_road_access:
            v += 1
        if b.is_developed and b.is_protected and b.amenity != "park":
            v += 1
        if b.use and b.zone in ZONE_USE_COMPAT and b.use not in ZONE_USE_COMPAT[b.zone]:
            v += 1
    t = state.targets
    if t.max_population > 0:
        total_pop = sum(b.population_capacity for b in state.blocks.values() if b.use == "housing")
        if total_pop > t.max_population:
            v += 1
    return v


def _amenity_shortfall(state: WorldState) -> int:
    cov = _amenity_coverage(state)
    return sum(1 for a in state.targets.required_amenities if cov.get(a, 0) == 0)


def _use_target_shortfall(state: WorldState) -> int:
    counts = _count_uses(state)
    return sum(max(0, target - counts.get(use, 0)) for use, target in state.targets.blocks_by_use.items())


def _remaining_use_targets(state: WorldState) -> Dict[str, int]:
    counts = _count_uses(state)
    return {
        use: max(0, target - counts.get(use, 0))
        for use, target in state.targets.blocks_by_use.items()
        if max(0, target - counts.get(use, 0)) > 0
    }


def _remaining_amenities(state: WorldState) -> List[str]:
    cov = _amenity_coverage(state)
    return [a for a in state.targets.required_amenities if cov.get(a, 0) == 0]


def _candidate_blocks(state: WorldState) -> Dict[str, List[str]]:
    candidates: Dict[str, List[str]] = {
        "housing": [],
        "retail": [],
        "office": [],
        "workshop": [],
        "institutional": [],
        "park_or_open_space": [],
    }
    for b in state.blocks.values():
        if b.is_developed or b.phase != 0:
            continue
        if not b.has_road_access:
            candidates["park_or_open_space"].append(b.block_id)
            continue
        if b.is_protected:
            candidates["park_or_open_space"].append(b.block_id)
            continue
        if b.zone in ("unzoned", "residential"):
            candidates["housing"].append(b.block_id)
        if b.zone in ("unzoned", "commercial", "mixed"):
            candidates["retail"].append(b.block_id)
            candidates["office"].append(b.block_id)
        if b.zone in ("unzoned", "industrial"):
            candidates["workshop"].append(b.block_id)
        if b.zone in ("unzoned", "civic", "mixed"):
            candidates["institutional"].append(b.block_id)
        if b.zone in ("unzoned", "open_space", "civic"):
            candidates["park_or_open_space"].append(b.block_id)
    return {k: v for k, v in candidates.items() if v}


def _planner_support(state: WorldState) -> Dict[str, Any]:
    remaining_uses = _remaining_use_targets(state)
    remaining_amenities = _remaining_amenities(state)
    green_needed = int((state.targets.min_greenery_ratio * len(state.blocks)) + 0.9999)
    green_now = sum(
        1 for b in state.blocks.values()
        if b.use == "park" or b.reserved_open_space or b.zone == "open_space"
    )
    return {
        "solve_as": "small dependency graph / constraint-satisfaction plan",
        "remaining_use_targets": remaining_uses,
        "remaining_amenities": remaining_amenities,
        "green_blocks_needed": max(0, green_needed - green_now),
        "candidate_blocks_by_role": _candidate_blocks(state),
        "dependency_rules": [
            "set_zoning before develop unless assigning a compatible amenity can develop the block",
            "housing needs residential zoning",
            "retail/office need commercial or mixed zoning",
            "institutional/school/fire need civic or mixed zoning",
            "park/open_space satisfies greenery but too much open_space can block housing/retail targets",
            "prefer actions that reduce remaining_use_targets, remaining_amenities, or green_blocks_needed",
            "do not repeat an already completed action on the same block",
        ],
        "high_value_next_action_order": [
            "zone blocks for remaining housing/retail/institutional targets",
            "assign required school/clinic/fire/transit/park amenities on compatible blocks",
            "develop zoned blocks into remaining target uses",
            "reserve only the open_space still needed for greenery/protected/flood/no-road blocks",
        ],
    }


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
    spatial = _spatial_service_stats(state)
    district_targets = state.targets.district_targets or {}
    if not district_targets:
        # Tiny tiers without district targets fall back to global amenity coverage.
        cov = _amenity_coverage(state)
        vals = [1.0 if cov.get(a, 0) > 0 else 0.0 for a in state.targets.required_amenities]
        global_score = sum(vals) / max(1, len(vals))
        return 0.5 * global_score + 0.5 * float(spatial["spatial_service_score"])
    stats = _district_stats(state)
    achieved = 0.0
    total = 0
    for district, target in district_targets.items():
        amenities = stats.get(district, {}).get("amenities", {})
        for amenity in target.get("required_amenities", []):
            total += 1
            achieved += 1.0 if amenities.get(amenity, 0) > 0 else 0.0
    district_score = achieved / max(1, total)
    return 0.5 * district_score + 0.5 * float(spatial["spatial_service_score"])


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
    if t.service_radius:
        items.append(_spatial_service_stats(state)["spatial_service_score"])
        if t.max_population > 0:
            items.append(_population_stats(state)["population_served_ratio"])
    if t.district_targets:
        items.append(_accessibility_score(state))
        items.append(_district_coverage_score(state))
        items.append(_land_use_balance_score(state))
    return sum(items) / max(1, len(items))


def _final_valid(state: WorldState) -> bool:
    if _progress(state) < 0.999 or _violations(state) != 0:
        return False
    if _spatial_service_stats(state)["spatial_service_gap_count"] != 0:
        return False
    t = state.targets
    if t.max_population > 0:
        pop = _population_stats(state)
        if pop["population_served_ratio"] < 0.999:
            return False
    return True


def compute_metrics(state: WorldState, last_action_legal: bool, last_action_was_mutator: bool,
                    timeout: bool, prev_metrics: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
    spatial = _spatial_service_stats(state)
    pop = _population_stats(state)
    accessibility_score = _accessibility_score(state)
    land_use_balance_score = _land_use_balance_score(state)
    district_coverage_score = _district_coverage_score(state)
    phase_completion_score = _phase_completion_score(state)
    progress_score = round(_progress(state), 4)

    no_progress_step = 0
    if prev_metrics is not None and last_action_legal and not timeout:
        prev_progress = prev_metrics.get("progress_score", 0.0)
        if (progress_score - prev_progress) <= 0.0001:
            no_progress_step = 1

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
        "spatial_service_gap_count": spatial["spatial_service_gap_count"],
        "designation_violation_count": _designation_violations(state),
        "repeated_action_count": _repeated_action_count(state),
        "no_progress_step": no_progress_step,
        "open_space_oversupply_count": max(
            0,
            sum(1 for b in state.blocks.values() if b.use == "park" or b.reserved_open_space or b.zone == "open_space")
            - int((state.targets.min_greenery_ratio * len(state.blocks)) + 0.9999),
        ),
        "city_resource_overflow_count": _city_resource_overflow(state),
        "population_total": pop["population_total"],
        "population_served_ratio": pop["population_served_ratio"],
        "budget_remaining_ratio": _budget_remaining_ratio(state),
        "accessibility_score": round(accessibility_score, 4),
        "spatial_service_score": round(float(spatial["spatial_service_score"]), 4),
        "land_use_balance_score": round(land_use_balance_score, 4),
        "district_coverage_score": round(district_coverage_score, 4),
        "phase_completion_score": round(phase_completion_score, 4),
        "progress_score": progress_score,
        "affected_set_precision": round(affected_p, 4),
        "affected_set_recall": round(affected_r, 4),
        "unnecessary_change_count": unnecessary,
        "final_valid_plan": int(_final_valid(state)),
        "timeout": int(timeout),
    }


def compute_reward_components(metrics: Dict[str, Any], prev_metrics: Dict[str, float],
                              done: bool, last_action_type: str = "") -> Dict[str, float]:
    """Per-step decomposed reward.

    Shaping kept minimal and stable: the dominant signal is `terminal`. Other
    components nudge the policy toward legality, constraint satisfaction,
    measurable progress, scoped replanning, and revision discipline.
    """
    repeated = int(metrics.get("repeated_action_count", 0))
    # Prerequisite actions (zoning, infra upgrades, defer) are required setup steps —
    # they don't advance progress_score directly, so never penalise them as no-progress.
    _PREREQ_ACTIONS = {"set_zoning", "upgrade_infrastructure", "defer"}
    no_progress = bool(metrics.get("no_progress_step", 0)) and last_action_type not in _PREREQ_ACTIONS
    progress_delta = metrics["progress_score"] - prev_metrics.get("progress_score", 0.0)

    legality = -1.0 if metrics["illegal_action"] else 0.0
    # Small positive shaping for productive setup: zoning unlocks develop, infra upgrades unblock capacity.
    if not metrics["illegal_action"] and last_action_type == "set_zoning":
        legality += 0.05
    if not metrics["illegal_action"] and last_action_type == "upgrade_infrastructure":
        legality += 0.03
    if repeated:
        legality -= 0.35

    constraints = -0.5 * metrics["constraint_violations"]
    if metrics["progress_score"] >= 0.50:
        constraints -= 0.05 * int(metrics.get("spatial_service_gap_count", 0))
    if metrics.get("use_target_shortfall_count", 0) or metrics.get("amenity_shortfall_count", 0):
        constraints -= 0.05 * int(metrics.get("open_space_oversupply_count", 0))
    constraints -= 0.3 * int(metrics.get("city_resource_overflow_count", 0))
    constraints -= 0.4 * int(metrics.get("designation_violation_count", 0))

    accessibility = 0.5 * max(0.0, metrics["accessibility_score"] - prev_metrics.get("accessibility_score", 0.0))
    land_use_balance = 0.5 * max(0.0, metrics["land_use_balance_score"] - prev_metrics.get("land_use_balance_score", 0.0))
    district_quality = 0.5 * max(0.0, metrics["district_coverage_score"] - prev_metrics.get("district_coverage_score", 0.0))
    phase_progress = 0.5 * max(0.0, metrics["phase_completion_score"] - prev_metrics.get("phase_completion_score", 0.0))
    progress = max(0.0, progress_delta)
    if no_progress:
        progress -= 0.03
    f1 = 0.0
    p, r = metrics["affected_set_precision"], metrics["affected_set_recall"]
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    replanning = f1 if done else 0.0
    revision_discipline = -0.25 * metrics["unnecessary_change_count"] if done else 0.0
    terminal = 8.0 if (done and metrics["final_valid_plan"]) else 0.0
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
            "road": [round(z.road_alloc, 1), round(z.road_capacity, 1)],
        }
        for z in state.infra_zones.values()
    }
    cr = state.city_resources
    city_totals = {
        "water": [round(sum(z.water_alloc for z in state.infra_zones.values()), 1),
                  round(cr.water_supply, 1) if cr.water_supply > 0 else "uncapped"],
        "power": [round(sum(z.power_alloc for z in state.infra_zones.values()), 1),
                  round(cr.power_grid, 1) if cr.power_grid > 0 else "uncapped"],
        "sewer": [round(sum(z.sewer_alloc for z in state.infra_zones.values()), 1),
                  round(cr.sewer_network, 1) if cr.sewer_network > 0 else "uncapped"],
        "budget_remaining": round(cr.construction_budget, 0) if cr.construction_budget > 0 else "uncapped",
    }
    pop = _population_stats(state)
    return {
        "blocks_total": len(state.blocks),
        "blocks_by_zone": blocks_by_zone,
        "blocks_by_use": blocks_by_use,
        "infra_zones": infra,
        "city_resources": city_totals,
        "population": {
            "total": pop["population_total"],
            "served_ratio": pop["population_served_ratio"],
            "max_allowed": state.targets.max_population if state.targets.max_population > 0 else "uncapped",
        },
        "designated_blocks": {
            b.block_id: b.future_designation
            for b in state.blocks.values()
            if b.future_designation
        },
        "designation_violations": _designation_violations(state),
        "amenity_coverage": _amenity_coverage(state),
        "spatial_service": _spatial_service_stats(state),
        "planner_support": _planner_support(state),
        "adjacency": dict(state.adjacency),
        "external_ledgers": dict(state.external_ledgers),
        "district_stats": _district_stats(state),
        "greenery_ratio": round(_greenery_ratio(state), 3),
        "targets": {
            "blocks_by_use": dict(state.targets.blocks_by_use),
            "required_amenities": list(state.targets.required_amenities),
            "min_greenery_ratio": state.targets.min_greenery_ratio,
            "max_population": state.targets.max_population,
            "district_targets": dict(state.targets.district_targets),
            "service_radius": dict(state.targets.service_radius),
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
                and b.zone not in ("unzoned", "open_space") and b.phase == 0):
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
           "min_greenery_ratio", "required_amenities",
           "district_access", "district_mix", "district_greenery"]
    if state.curveballs:
        out.append("curveball_replanning")
    if state.targets.service_radius:
        out.append("graph_service_coverage")
    if state.adjacency:
        out.append("spatial_adjacency")
    cr = state.city_resources
    if cr.water_supply > 0 or cr.power_grid > 0 or cr.sewer_network > 0:
        out.append("city_resource_cap")
    if cr.construction_budget > 0:
        out.append("budget_constraint")
    if state.targets.max_population > 0:
        out.append("population_cap")
    designated = [b for b in state.blocks.values() if b.future_designation]
    if designated:
        types = sorted({b.future_designation for b in designated})
        out.append(f"future_designations:{','.join(types)}")
    if state.external_ledgers:
        out.append("external_ledgers")
    return out
