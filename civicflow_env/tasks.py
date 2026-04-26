"""
Task loader for CivicFlow.

Tasks are JSON fixtures under `data/tasks/`. The schema is the contract
between Person A (server) and Person B (task authoring); do not change
field names without re-freezing.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from .server.state import Block, CityResources, CurveballSpec, InfraZone, Targets, WorldState


_DATA_DIR = Path(__file__).parent / "data" / "tasks"


def list_task_ids() -> List[str]:
    return sorted(p.stem for p in _DATA_DIR.glob("*.json"))


def load_task_dict(task_id: str) -> dict:
    path = _DATA_DIR / f"{task_id}.json"
    with open(path) as f:
        return json.load(f)


def build_world_state(task: dict) -> WorldState:
    """Inflate a JSON task dict into a fresh WorldState."""
    blocks: Dict[str, Block] = {}
    for b in task["blocks"]:
        blocks[b["block_id"]] = Block(
            block_id=b["block_id"],
            district=b.get("district", b["infra_zone"]),
            has_road_access=b.get("has_road_access", True),
            is_protected=b.get("is_protected", False),
            infra_zone=b["infra_zone"],
            water_demand=b.get("water_demand", 5.0),
            sewer_demand=b.get("sewer_demand", 4.0),
            power_demand=b.get("power_demand", 8.0),
            road_demand=float(b.get("road_demand", 0.0)),
            population_capacity=int(b.get("population_capacity", 0)),
            future_designation=b.get("future_designation") or None,
        )

    infra_zones: Dict[str, InfraZone] = {}
    for z in task["infra_zones"]:
        infra_zones[z["infra_zone_id"]] = InfraZone(
            infra_zone_id=z["infra_zone_id"],
            water_capacity=float(z["water_capacity"]),
            sewer_capacity=float(z["sewer_capacity"]),
            power_capacity=float(z["power_capacity"]),
            road_capacity=float(z.get("road_capacity", 100.0)),
        )

    t = task["targets"]
    targets = Targets(
        blocks_by_use=dict(t["blocks_by_use"]),
        required_amenities=list(t["required_amenities"]),
        min_greenery_ratio=float(t["min_greenery_ratio"]),
        max_episode_steps=int(t["max_episode_steps"]),
        district_targets=dict(t.get("district_targets", {})),
        service_radius={k: int(v) for k, v in dict(t.get("service_radius", {})).items()},
        max_population=int(t.get("max_population", 0)),
    )

    adjacency: Dict[str, List[str]] = {block_id: [] for block_id in blocks}
    for edge in task.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        a, b = str(edge[0]), str(edge[1])
        if a in adjacency and b in adjacency:
            adjacency[a].append(b)
            adjacency[b].append(a)
    for block_id in adjacency:
        adjacency[block_id] = sorted(set(adjacency[block_id]))

    # Accept either the legacy single `curveball` field or a `curveballs` list.
    raw_list = task.get("curveballs")
    if raw_list is None:
        single = task.get("curveball")
        raw_list = [single] if single else []
    curveballs: List[CurveballSpec] = []
    for cb in raw_list:
        block_ids = list(cb.get("block_ids", []))
        if not block_ids and cb.get("block_id"):
            block_ids = [cb["block_id"]]
        curveballs.append(CurveballSpec(
            fire_at_step=int(cb["fire_at_step"]),
            description=cb["description"],
            mutation=cb["mutation"],
            gold_affected=list(cb["gold_affected"]),
            block_ids=block_ids,
            infra_zone=cb.get("infra_zone"),
            infra_type=cb.get("infra_type"),
            capacity_delta=float(cb.get("capacity_delta", 0.0)),
            target_overrides=dict(cb.get("target_overrides", {})),
            designation_type=cb.get("designation_type") or None,
        ))
    curveballs.sort(key=lambda c: c.fire_at_step)

    planning_phases = task.get("planning_phases") or [
        {"name": "base_structure", "objective": "Establish land-use skeleton, open space, and district structure."},
        {"name": "services_and_access", "objective": "Place amenities and preserve neighbourhood accessibility."},
        {"name": "infrastructure_balance", "objective": "Keep infra, greenery, and district balance feasible."},
        {"name": "replanning", "objective": "Revise only the affected subgraph after curveballs."},
        {"name": "final_validation", "objective": "Reach a valid and complete district plan."},
    ]

    cr_raw = task.get("city_resources", {})
    city_resources = CityResources(
        water_supply=float(cr_raw.get("water_supply", 0.0)),
        power_grid=float(cr_raw.get("power_grid", 0.0)),
        sewer_network=float(cr_raw.get("sewer_network", 0.0)),
        construction_budget=float(cr_raw.get("construction_budget", 0.0)),
        action_costs=dict(cr_raw.get("action_costs", {})),
    )

    # Phased block revelation: if task has a "phases" key, only reveal phase 0 blocks.
    phases_raw = task.get("phases")
    hidden_blocks: Dict[str, Block] = {}
    pending_phases: List[dict] = []
    all_edges: List[List[str]] = list(task.get("edges", []))

    if phases_raw:
        phase0_ids = set(phases_raw[0]["block_ids"])
        hidden_blocks = {bid: b for bid, b in blocks.items() if bid not in phase0_ids}
        blocks = {bid: b for bid, b in blocks.items() if bid in phase0_ids}
        # Rebuild adjacency for phase 0 only
        adjacency = {bid: [] for bid in blocks}
        for edge in all_edges:
            if not isinstance(edge, list) or len(edge) != 2:
                continue
            a, b_id = str(edge[0]), str(edge[1])
            if a in adjacency and b_id in adjacency:
                adjacency[a].append(b_id)
                adjacency[b_id].append(a)
        for bid in adjacency:
            adjacency[bid] = sorted(set(adjacency[bid]))
        # Store subsequent phases with their new_block_ids and cumulative targets
        for ph in phases_raw[1:]:
            ph_targets = ph["targets"]
            pending_phases.append({
                "name": ph["name"],
                "objective": ph.get("objective", ""),
                "new_block_ids": list(ph["block_ids"]),
                "targets": ph_targets,
            })
        # Override initial targets with phase 0 targets
        pt = phases_raw[0]["targets"]
        targets = Targets(
            blocks_by_use=dict(pt["blocks_by_use"]),
            required_amenities=list(pt["required_amenities"]),
            min_greenery_ratio=float(pt["min_greenery_ratio"]),
            max_episode_steps=int(task["targets"]["max_episode_steps"]),
            district_targets=dict(pt.get("district_targets", {})),
            service_radius={k: int(v) for k, v in dict(pt.get("service_radius", {})).items()},
            max_population=int(pt.get("max_population", 0)),
        )

    return WorldState(
        task_id=task["task_id"],
        briefing=task["briefing"],
        blocks=blocks,
        infra_zones=infra_zones,
        targets=targets,
        adjacency=adjacency,
        city_resources=city_resources,
        external_ledgers=dict(task.get("external_ledgers", {})),
        curveballs=curveballs,
        planning_phases=planning_phases,
        curveballs_fired=[False] * len(curveballs),
        pending_phases=pending_phases,
        _hidden_blocks=hidden_blocks,
        _all_edges=all_edges,
    )


def pick_task(rng: random.Random, override: Optional[str] = None) -> WorldState:
    """Pick a task by id, by env var, or uniformly at random."""
    if override is None:
        override = os.environ.get("CIVICFLOW_TASK_ID")
    if override:
        return build_world_state(load_task_dict(override))
    task_id = rng.choice(list_task_ids())
    return build_world_state(load_task_dict(task_id))
