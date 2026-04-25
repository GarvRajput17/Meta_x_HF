---
title: CivicFlow Environment Server
emoji: 🏙️
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - long-horizon-planning
  - rlvr
---

# CivicFlow

**Long-horizon municipal planning under evolving constraints — an OpenEnv environment.**

The agent acts as a municipal planner on a graph of **blocks** (developable
units), **roads**, and **infrastructure zones** (water/sewer/power capacity
pools). Episodes are phase-aware: base structure -> services/access ->
infrastructure balance -> replanning -> final validation. The deterministic
verifier scores legality, hard-constraint satisfaction, district-level amenity
coverage, greenery ratio, land-use balance, and — when a policy curveball
fires — how well the agent revises the affected districts while preserving
the rest of the plan.

Targets hackathon **Theme #2 (long-horizon planning & instruction following)**
with secondary fit to **#3.1 (professional tasks)**.

## Status

**Phase 0 (bootstrap)** — schemas frozen, `reset` / `step` / `state`
round-trip live, mock task wired so the HTTP + WebSocket contract can be
exercised before Phase 1 lands real graph state, transitions, and the full
verifier stack.

## Action vocabulary

### Tiny tier (MVP, implemented)

| Action | Effect |
|---|---|
| `set_zoning(block_id, zone)` | Designate a block's land-use class (residential, commercial, industrial, mixed, civic, open_space) |
| `develop(block_id, use)` | Commission construction. Consumes infra capacity; requires zoning + road access |
| `reserve_open_space(block_id)` | Keep a block undeveloped — counts toward greenery / floodplain / setback targets |
| `upgrade_infrastructure(infra_zone, infra_type, capacity)` | Add water/sewer/power/road capacity to an infra zone |
| `assign_amenity(block_id, amenity_type)` | Site a school / clinic / grocery / park / fire / transit on a block |
| `redevelop(block_id, use)` | Change a developed block's use — the replanning lever under curveballs |
| `defer(block_id, phase_id)` | Push a decision to a later planning phase |

### Medium-tier roadmap (post-MVP)

`add_road_segment` · `set_density` (FAR) · `set_setback` ·
`establish_easement` · `phase_budget` · `request_variance`

### Hard-tier roadmap

`supersede_decision` (with justification, scored against affected-set) ·
`acquire_land` · `redistrict` (bulk rezoning, tests scoped vs over-revision)

## Verifier metrics (canonical)

Every step emits these keys in `observation.last_metrics`. Trainer-side
reward functions consume them directly — never recompute legality client-side.

```
illegal_action            constraint_violations    infra_overflow_count
amenity_shortfall_count   greenery_shortfall       progress_score
use_target_shortfall_count district_service_gap_count district_green_gap_count
district_mix_gap_count    accessibility_score      land_use_balance_score
district_coverage_score   phase_completion_score
affected_set_precision    affected_set_recall      unnecessary_change_count
final_valid_plan          timeout
```

## Reward components

`observation.last_reward_components` decomposes per-step reward into:

```
legality   constraints   accessibility   land_use_balance   district_quality
phase_progress   progress   replanning   revision_discipline   terminal
```

## Quick start (local)

```bash
pip install -e .
server --port 8000
# in another shell
curl http://localhost:8000/health
```

Persistent WebSocket session via the client:

```python
from civicflow_env.client import CivicflowEnv
from civicflow_env.models import CivicflowAction

with CivicflowEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset().observation
    print(obs.briefing)
    obs = env.step(CivicflowAction(
        action_type="set_zoning", block_id="B1", zone="residential"
    )).observation
    print(obs.last_metrics, obs.last_reward_components)
```

## Endpoints

`/health` · `/docs` · `/schema` · `/reset` · `/step` · `/state` · `/ws`
(persistent session) · `/mcp`

## Repo layout

```
civicflow_env/
├── models.py                          # frozen Action/Observation schemas
├── client.py                          # CivicflowEnv WebSocket client
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml
└── server/
    ├── app.py                         # FastAPI app, max_concurrent_envs=8
    ├── civicflow_env_environment.py   # Environment + verifier (Phase 1)
    └── Dockerfile
```

## Deployment

```bash
# Hugging Face Space
openenv push --repo-id <hf-username>/civicflow-env
```

After Phase 1 the Space is the canonical demo target; trainer can switch
between local and remote via `ENV_BASE_URL`.
