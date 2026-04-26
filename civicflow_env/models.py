"""
Data models for the CivicFlow environment.

Schemas frozen at Phase 0 bootstrap. Do not mutate field names or types
after this point — server, client, and trainer-side reward functions all
depend on this contract.

Vocabulary: a `block` is a contiguous developable unit (the standard
urban-planning term). Blocks sit on a road graph and draw on infrastructure
zones (water / sewer / power capacity pools).
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# Tiny-tier action vocabulary. Medium/Hard tiers add more verbs (see README
# tier roadmap) but never repurpose these names.
ACTION_TYPES = (
    "set_zoning",            # designate a block's land-use class
    "develop",               # commission construction (consumes capacity, requires access)
    "reserve_open_space",    # keep block undeveloped (greenery / floodplain / setback)
    "upgrade_infrastructure",  # add water / sewer / power / road capacity
    "assign_amenity",        # site a school / clinic / grocery / park / fire / transit
    "redevelop",             # change a developed block's use (replanning under curveball)
    "defer",                 # push decision to a later planning phase
)


class CivicflowAction(Action):
    """A single municipal-planning action.

    Only `action_type` is always required. Other fields are populated based
    on which action is being emitted; the verifier rejects actions whose
    payload does not match the action type.
    """

    action_type: str = Field(..., description=f"One of {ACTION_TYPES}")
    block_id: Optional[str] = Field(
        default=None,
        description="Target block id for set_zoning, develop, reserve_open_space, "
        "assign_amenity, redevelop, defer",
    )
    zone: Optional[str] = Field(
        default=None,
        description="Zoning class for set_zoning "
        "(residential|commercial|industrial|mixed|civic|open_space)",
    )
    use: Optional[str] = Field(
        default=None,
        description="Land-use type for develop / redevelop "
        "(housing|retail|office|workshop|institutional|park)",
    )
    amenity_type: Optional[str] = Field(
        default=None,
        description="Amenity for assign_amenity "
        "(school|clinic|grocery|park|fire|transit)",
    )
    infra_zone: Optional[str] = Field(
        default=None,
        description="Infrastructure zone id for upgrade_infrastructure",
    )
    infra_type: Optional[str] = Field(
        default=None,
        description="Infrastructure type for upgrade_infrastructure (water|sewer|power|road)",
    )
    capacity: Optional[float] = Field(
        default=None,
        description="Capacity to add for upgrade_infrastructure",
    )
    phase_id: Optional[int] = Field(
        default=None,
        description="Phase index for defer",
    )
    decision_id: Optional[str] = Field(
        default=None,
        description="Prior-decision id (Medium/Hard tier supersede_decision)",
    )
    justification: Optional[str] = Field(
        default=None,
        description="Optional free-text rationale (not scored in core reward)",
    )


class CivicflowObservation(Observation):
    """What the agent sees after `reset` or `step`.

    All structured fields are emitted by the deterministic verifier. Trainer
    rewards must read these — never recompute legality client-side.
    """

    briefing: str = Field(default="", description="Task briefing or step-result text")
    planning_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Counts of blocks by zone/use, infrastructure allocations, coverage stats",
    )
    legal_actions_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-action-type hints about which blocks/zones currently legal",
    )
    active_constraints: List[str] = Field(
        default_factory=list,
        description="Names of currently-binding hard constraints",
    )
    current_phase: str = Field(default="base_structure", description="Current planning phase name")
    phase_objective: str = Field(default="", description="Current planning objective for the agent")
    last_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verifier metrics: illegal_action, constraint_violations, "
        "infra_overflow_count, amenity_shortfall_count, greenery_shortfall, "
        "use_target_shortfall_count, district_service_gap_count, "
        "district_green_gap_count, district_mix_gap_count, spatial_service_gap_count, "
        "designation_violation_count, repeated_action_count, no_progress_step, open_space_oversupply_count, "
        "city_resource_overflow_count, population_total, population_served_ratio, "
        "budget_remaining_ratio, accessibility_score, spatial_service_score, "
        "land_use_balance_score, district_coverage_score, phase_completion_score, "
        "progress_score, affected_set_precision, affected_set_recall, "
        "unnecessary_change_count, final_valid_plan, timeout",
    )
    last_reward_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed reward components: legality, constraints, accessibility, "
        "land_use_balance, district_quality, phase_progress, progress, "
        "replanning, revision_discipline, terminal",
    )
    curveball_active: bool = Field(default=False, description="A policy curveball is in effect")
    curveball_text: Optional[str] = Field(default=None, description="Description of the curveball")
    task_id: Optional[str] = Field(default=None, description="Current task identifier")
    step_index: int = Field(default=0, description="Number of steps elapsed in episode")
    timeout: bool = Field(default=False, description="Episode terminated by timeout")
