"""CivicFlow environment client.

Maintains a persistent WebSocket session against a CivicFlow server, whether
running locally (`uv run server`) or on a Hugging Face Space. Trainer and
baseline code construct one client per parallel rollout.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CivicflowAction, CivicflowObservation


class CivicflowEnv(EnvClient[CivicflowAction, CivicflowObservation, State]):
    """Client for the CivicFlow environment."""

    def _step_payload(self, action: CivicflowAction) -> Dict:
        # Drop Nones so the server sees only fields the action actually uses.
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[CivicflowObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = CivicflowObservation(
            briefing=obs_data.get("briefing", ""),
            planning_summary=obs_data.get("planning_summary", {}) or {},
            legal_actions_summary=obs_data.get("legal_actions_summary", {}) or {},
            active_constraints=obs_data.get("active_constraints", []) or [],
            last_metrics=obs_data.get("last_metrics", {}) or {},
            last_reward_components=obs_data.get("last_reward_components", {}) or {},
            curveball_active=obs_data.get("curveball_active", False),
            curveball_text=obs_data.get("curveball_text"),
            task_id=obs_data.get("task_id"),
            step_index=obs_data.get("step_index", 0),
            timeout=obs_data.get("timeout", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}) or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
