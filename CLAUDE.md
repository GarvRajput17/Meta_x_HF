# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **Meta OpenEnv hackathon submission** (India, April 2026): **CivicFlow**, a long-horizon municipal planning environment for RL training. The agent acts as a city planner on a graph of blocks, roads, and infrastructure zones.

Reference docs in the repo root explain the rules and context:
- `Themes_and_judging_criteria.md` — judging rubric (Env Innovation 40%, Storytelling 30%, Reward 20%, Pipeline 10%) and non-negotiable submission requirements
- `pariticipant_help_guide.md` — RL loop design, reward design, anti-hacking, stack
- `HackathonFAQs.md` — RLVR, GRPO, reward hacking concepts
- `hackathon_resources.md` — canonical links

## Running the server

```bash
# From civicflow_env/ directory
uv sync                                    # install deps
uv run --project . server                  # run on :8000
uv run --project . server --port 8001      # custom port

# Or directly with uvicorn
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Dev check
curl http://localhost:8000/health
```

## Running the heuristic baseline

```bash
# From repo root; installs civicflow_env as a package first
pip install -e civicflow_env

python training/baselines/heuristic.py --task all --verbose
python training/baselines/heuristic.py --task tiny_a --out results.json
```

## Preparing SFT data

```bash
python training/sft/prepare_sft_data.py \
    --in syndata.json --in syndata1.json \
    --out-json training/sft/sft_merged_clean.json \
    --out-jsonl training/sft/sft_merged_clean.jsonl \
    --report training/sft/sft_merge_report.json
```

## Architecture

```
civicflow_env/            # OpenEnv environment package (server side)
├── models.py             # Frozen CivicflowAction / CivicflowObservation + ACTION_TYPES
├── client.py             # CivicflowEnv WebSocket client (never imports server internals)
├── tasks.py              # JSON task loader → WorldState; pick_task() selects by env var or random
├── data/tasks/*.json     # Task fixtures: tiny_a/b/c, medium_a, hard_a, hard1
├── openenv.yaml          # OpenEnv manifest (type: space, runtime: fastapi)
├── pyproject.toml
└── server/
    ├── app.py                        # FastAPI app via openenv.create_app(), max_concurrent_envs=8
    ├── civicflow_env_environment.py  # CivicflowEnvironment (reset/step/state); orchestrates only
    ├── state.py                      # WorldState, Block, InfraZone, Targets, CurveballSpec dataclasses
    └── verifier.py                   # All transition logic + scoring; pure-deterministic, no globals

training/
├── baselines/heuristic.py   # Greedy expert baseline; also generates SFT warm-start trajectories
└── sft/prepare_sft_data.py  # Merges/cleans syndata*.json → JSONL for trainer ingestion
```

**Key separation**: `environment.py` orchestrates episodes; `verifier.py` owns all legality checking and scoring; `state.py` owns world data; `tasks.py` loads fixtures. These must stay separated — the verifier is the ground truth; never recompute legality on the client side.

## Frozen contracts — do not change field names

These are consumed by trainer-side reward functions and must stay stable:

- `models.py` — `ACTION_TYPES`, `CivicflowAction` fields, `CivicflowObservation` fields
- `verifier.py` — `METRIC_KEYS` (19 keys), `REWARD_COMPONENT_KEYS` (10 keys)
- Task JSON schema — `blocks`, `infra_zones`, `targets`, `curveballs`, `planning_phases`

## Environment variables

| Variable | Effect |
|---|---|
| `CIVICFLOW_TASK_ID` | Pin the task loaded on `reset()` (e.g. `tiny_a`) |
| `CIVICFLOW_SEED` | Deterministic seed for task selection |
| `ENV_BASE_URL` | Used by trainer to switch between local server and HF Space |

## Task difficulty tiers

- **Tiny** (`tiny_a/b/c`): small grid, few constraints, no district targets — MVP/smoke tests
- **Medium** (`medium_a`): district targets, multi-amenity, curveball
- **Hard** (`hard_a`, `hard1`): multi-curveball, long-horizon, full district coverage + mix requirements

## Hard constraints from submission rules

- Build on **OpenEnv** — use `Environment` base class, standard Gym-style API (`reset`, `step`, `state`), valid `openenv.yaml`
- **Client/server separation** — `client.py` must never import from `server/`
- Do **not** use reserved MCP tool names (`reset`, `step`, `state`, `close`)
- Training must use **Unsloth or HuggingFace TRL**
- Deploy environment to a **Hugging Face Space** via `openenv push --repo-id <hf-username>/civicflow-env`
- Commit reward/loss plots as `.png`/`.jpg` to the Space README; do not commit large video files
