"""
World-state dataclasses for CivicFlow.

The environment instantiates a fresh `WorldState` from a `Task` on every
`reset()`. Mutations happen only through the verifier in step(); no code
outside the server package should ever construct or mutate these directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any


# Allowed enums. Kept as tuples so they're trivially serialisable.
ZONES = ("unzoned", "residential", "commercial", "industrial", "mixed", "civic", "open_space")
USES = ("housing", "retail", "office", "workshop", "institutional", "park")
AMENITIES = ("school", "clinic", "grocery", "park", "fire", "transit")
INFRA_TYPES = ("water", "sewer", "power", "road")

# Which uses are allowed in which zones.
ZONE_USE_COMPAT: Dict[str, Set[str]] = {
    "residential": {"housing"},
    "commercial": {"retail", "office"},
    "industrial": {"workshop"},
    "mixed": {"housing", "retail", "office", "institutional"},
    "civic": {"institutional", "park"},
    "open_space": {"park"},
    # "unzoned": nothing buildable
}


@dataclass
class Block:
    block_id: str
    district: str
    has_road_access: bool
    is_protected: bool
    is_floodrisk: bool
    infra_zone: str
    water_demand: float
    sewer_demand: float
    power_demand: float

    # Mutable fields set by actions:
    zone: str = "unzoned"
    use: Optional[str] = None
    amenity: Optional[str] = None
    phase: int = 0
    reserved_open_space: bool = False

    @property
    def is_developed(self) -> bool:
        return self.use is not None or self.amenity is not None


@dataclass
class InfraZone:
    infra_zone_id: str
    water_capacity: float
    sewer_capacity: float
    power_capacity: float
    road_capacity: float
    water_alloc: float = 0.0
    sewer_alloc: float = 0.0
    power_alloc: float = 0.0
    road_alloc: float = 0.0


@dataclass
class Targets:
    blocks_by_use: Dict[str, int]
    required_amenities: List[str]
    min_greenery_ratio: float
    max_episode_steps: int
    district_targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CurveballSpec:
    fire_at_step: int
    description: str
    mutation: str          # "protect" | "floodrisk" | "capacity_cut" | "target_override"
    gold_affected: List[str]
    block_ids: List[str] = field(default_factory=list)
    infra_zone: Optional[str] = None
    infra_type: Optional[str] = None
    capacity_delta: float = 0.0
    target_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def block_id(self) -> Optional[str]:
        return self.block_ids[0] if self.block_ids else None


@dataclass
class WorldState:
    task_id: str
    briefing: str
    blocks: Dict[str, Block]
    infra_zones: Dict[str, InfraZone]
    targets: Targets
    curveballs: List[CurveballSpec] = field(default_factory=list)
    planning_phases: List[Dict[str, str]] = field(default_factory=list)

    # Episode bookkeeping (mutated during stepping):
    step_count: int = 0
    curveballs_fired: List[bool] = field(default_factory=list)
    first_fire_step: Optional[int] = None
    blocks_touched_after_curveball: Set[str] = field(default_factory=set)
    action_history: List[dict] = field(default_factory=list)

    # Backward-compat shim so existing references to `state.curveball` (singular)
    # still resolve to the first curveball, and `curveball_fired` to its flag.
    @property
    def curveball(self) -> Optional[CurveballSpec]:
        return self.curveballs[0] if self.curveballs else None

    @property
    def curveball_fired(self) -> bool:
        return bool(self.curveballs_fired) and all(self.curveballs_fired)
