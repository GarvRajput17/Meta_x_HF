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

from .server.state import Block, CurveballSpec, InfraZone, Targets, WorldState


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
            is_floodrisk=b.get("is_floodrisk", False),
            infra_zone=b["infra_zone"],
            water_demand=b.get("water_demand", 5.0),
            sewer_demand=b.get("sewer_demand", 4.0),
            power_demand=b.get("power_demand", 8.0),
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
    )

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
        ))
    curveballs.sort(key=lambda c: c.fire_at_step)

    planning_phases = task.get("planning_phases") or [
        {"name": "base_structure", "objective": "Establish land-use skeleton, open space, and district structure."},
        {"name": "services_and_access", "objective": "Place amenities and preserve neighbourhood accessibility."},
        {"name": "infrastructure_balance", "objective": "Keep infra, greenery, and district balance feasible."},
        {"name": "replanning", "objective": "Revise only the affected subgraph after curveballs."},
        {"name": "final_validation", "objective": "Reach a valid and complete district plan."},
    ]

    return WorldState(
        task_id=task["task_id"],
        briefing=task["briefing"],
        blocks=blocks,
        infra_zones=infra_zones,
        targets=targets,
        curveballs=curveballs,
        planning_phases=planning_phases,
        curveballs_fired=[False] * len(curveballs),
    )


def pick_task(rng: random.Random, override: Optional[str] = None) -> WorldState:
    """Pick a task by id, by env var, or uniformly at random."""
    if override is None:
        override = os.environ.get("CIVICFLOW_TASK_ID")
    if override:
        return build_world_state(load_task_dict(override))
    task_id = rng.choice(list_task_ids())
    return build_world_state(load_task_dict(task_id))
