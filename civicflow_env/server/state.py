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

# Future designation types — city masterplan reservations on a block.
# A designated block has restricted allowed uses; the agent must honor the
# designation rather than freely develop it.
FUTURE_DESIGNATIONS = (
    "transit_corridor",    # earmarked for metro/BRT; only transit amenity or open_space
    "civic_anchor",        # public institution site; only institutional or school/clinic/fire
    "entertainment_zone",  # arena/stadium/convention; only institutional, no housing/workshop
    "green_corridor",      # citywide greenway; park/open_space only
)

# What uses are allowed on each designation type (empty set = no develop allowed)
DESIGNATION_ALLOWED_USES: Dict[str, Set[str]] = {
    "transit_corridor":  set(),                          # no develop; only assign transit or reserve
    "civic_anchor":      {"institutional"},
    "entertainment_zone": {"institutional"},
    "green_corridor":    {"park"},
}
# What amenities are allowed on each designation type
DESIGNATION_ALLOWED_AMENITIES: Dict[str, Set[str]] = {
    "transit_corridor":  {"transit", "park"},
    "civic_anchor":      {"school", "clinic", "fire", "park"},
    "entertainment_zone": {"park"},
    "green_corridor":    {"park"},
}

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

# Default construction costs per action type (monetary units).
# Tasks can override via city_resources.action_costs.
DEFAULT_ACTION_COSTS: Dict[str, float] = {
    "set_zoning": 2_000,
    "develop": 30_000,
    "assign_amenity": 20_000,
    "reserve_open_space": 5_000,
    "upgrade_infrastructure": 15_000,
    "redevelop": 25_000,
    "defer": 0,
}


@dataclass
class CityResources:
    """City-wide resource pools that cap total consumption across all infra zones."""
    water_supply: float = 0.0       # 0 = uncapped
    power_grid: float = 0.0         # 0 = uncapped
    sewer_network: float = 0.0      # 0 = uncapped
    construction_budget: float = 0.0  # 0 = uncapped
    action_costs: Dict[str, float] = field(default_factory=dict)


@dataclass
class Block:
    block_id: str
    district: str
    has_road_access: bool
    is_protected: bool
    infra_zone: str
    water_demand: float
    sewer_demand: float
    power_demand: float
    road_demand: float = 0.0
    population_capacity: int = 0    # residents when developed as housing
    future_designation: Optional[str] = None  # masterplan reservation (see FUTURE_DESIGNATIONS)

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
    # Optional graph-distance requirements, e.g. {"school": 1, "clinic": 2}
    # means every housing block should be within that many hops of the amenity.
    service_radius: Dict[str, int] = field(default_factory=dict)
    max_population: int = 0         # 0 = uncapped


@dataclass
class CurveballSpec:
    fire_at_step: int
    description: str
    mutation: str          # "protect" | "designate" | "capacity_cut" | "target_override"
    gold_affected: List[str]
    block_ids: List[str] = field(default_factory=list)
    infra_zone: Optional[str] = None
    infra_type: Optional[str] = None
    designation_type: Optional[str] = None  # for "designate" mutation
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
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    city_resources: CityResources = field(default_factory=CityResources)
    external_ledgers: Dict[str, Any] = field(default_factory=dict)
    curveballs: List[CurveballSpec] = field(default_factory=list)
    planning_phases: List[Dict[str, str]] = field(default_factory=list)

    # Episode bookkeeping (mutated during stepping):
    step_count: int = 0
    curveballs_fired: List[bool] = field(default_factory=list)
    first_fire_step: Optional[int] = None
    blocks_touched_after_curveball: Set[str] = field(default_factory=set)
    action_history: List[dict] = field(default_factory=list)

    # Phased block revelation (populated by tasks.py when task has "phases"):
    phase_idx: int = 0
    pending_phases: List[Dict] = field(default_factory=list)
    _hidden_blocks: Dict[str, Any] = field(default_factory=dict)
    _all_edges: List[List[str]] = field(default_factory=list)

    # Backward-compat shim so existing references to `state.curveball` (singular)
    # still resolve to the first curveball, and `curveball_fired` to its flag.
    @property
    def curveball(self) -> Optional[CurveballSpec]:
        return self.curveballs[0] if self.curveballs else None

    @property
    def curveball_fired(self) -> bool:
        return bool(self.curveballs_fired) and all(self.curveballs_fired)
