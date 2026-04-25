# CivicFlow Implementation Plan

## Summary

Build **CivicFlow** as a Tiny-first OpenEnv environment for long-horizon municipal planning on a parcel-road-utility graph, with a strictly deterministic verifier and a **`SFT -> GRPO`** training path. The required MVP is:

- one OpenEnv environment scaffolded first and kept stable before training,
- three Tiny task maps only,
- deterministic verifier and multiple reward components,
- local server + Docker + Hugging Face Space deployment,
- baseline runs, SFT warm start, then GRPO,
- no LLM judge in the reward loop,
- two-person split **only after** the initial bootstrap is complete.

Chosen defaults:
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Scope: Tiny only
- Judge model: none in reward or eval
- Training: SFT warm start, then GRPO
- Environment representation: symbolic graph, not GIS polygons
- Deployment target: local first, then Hugging Face Docker Space via `openenv push`

## Implementation Changes

### Phase 0 — Initial bootstrap (Person A only, blocking gate)
This phase must finish before the two-person split starts.

- Create a new environment repo with `openenv init civicflow_env`.
- Keep the standard OpenEnv scaffold intact: `openenv.yaml`, `pyproject.toml`, `client.py`, `models.py`, server app, Dockerfile, README, outputs folder.
- Install core tooling with `pip install openenv-core` and install the repo in editable mode.
- Define the public contract once and freeze it before splitting work:
  - `Action` schema: `action_type`, `parcel_id`, optional `target_zone`, optional `service_type`, optional `utility_zone`, optional `amount`, optional `phase_id`, optional `decision_id`, optional `justification`
  - `Observation` schema: current planning summary, legal actions summary, active constraints summary, latest verifier metrics, curveball status, done flag
  - `State` schema: full parcel graph, road graph, utility capacities, service/liveability stats, episode metadata, action history, current task id, current phase, curveball flags
- Implement empty but wired `reset()`, `step()`, and `state()` with deterministic mock data so the local loop works immediately.
- Add the training-facing environment wrapper contract:
  - a Python class used by TRL `environment_factory`
  - public tool methods matching the planning actions
  - `reset()` returning the initial user-visible observation string
  - `last_metrics` and `last_reward_components` stored on the wrapper for reward functions
- Turn on OpenEnv training concurrency from day one:
  - declare `SUPPORTS_CONCURRENT_SESSIONS = True`
  - set `max_concurrent_envs` to at least the planned `generation_batch_size`
- Add local run instructions and verify:
  - local server runs with `uv run server --host 0.0.0.0 --port 8000`
  - `/health`, `/docs`, and `/web` load successfully
  - one manual reset/step cycle succeeds

**Bootstrap handoff gate:** after this phase, Person B must be able to run the local server, connect a client, see one sample task, and call one action successfully.

### Phase 1 — Environment runtime and verifier (Person A)
Person A owns all server-side logic, all deterministic checking, all packaging, and all deployment. Person B must not edit runtime code after the split.

- Implement the Tiny task state model with three hand-authored scenarios:
  - `tiny_a`: 8–10 parcels, 1 utility zone, school + park target
  - `tiny_b`: 10–12 parcels, protected parcel, clinic/grocery coverage target
  - `tiny_c`: 12–15 parcels, 2 utility zones, one policy curveball
- Represent the world as typed symbolic data:
  - parcels with zoning, land use, frontage, developable/protected/flood flags, phase, permit status
  - road connectivity graph with adjacency to parcels
  - utility zones with water/sewer/power capacities and current allocations
  - service/liveability layer with school, clinic, grocery, park, greenery, transit accessibility counters
- Implement only the Tiny action set:
  - `rezone`
  - `approve_permit`
  - `deny_permit`
  - `upgrade_utility`
  - `reserve_service`
  - `defer`
- Keep all transitions deterministic and side-effect free outside environment state.
- Build the verifier as server-side modules that run on every step:
  - legality checker: invalid action, wrong parcel state, missing prerequisites
  - hard-constraint checker: road access, zoning compatibility, utility overflow, protected land, phase violations
  - liveability checker: amenity coverage, greenery/open-space minimum, daily-needs accessibility
  - terminal checker: target counts reached, all hard constraints satisfied, final plan valid
  - curveball affected-scope checker: score whether the right parcels/decisions were revised and unaffected ones preserved
- Emit machine-readable metrics after every step and at episode end:
  - `illegal_action`
  - `constraint_violations`
  - `utility_overflow_count`
  - `amenity_shortfall_count`
  - `greenery_shortfall`
  - `progress_score`
  - `affected_set_precision`
  - `affected_set_recall`
  - `unnecessary_change_count`
  - `final_valid_plan`
  - `done`
  - `timeout`
- Add anti-hacking protections required by the guide:
  - max episode length per task
  - illegal-action penalty without crashing the episode
  - no hidden mutable globals between episodes
  - deterministic seeding for reproducibility
  - clear timeout termination reason
- Package and deploy early:
  - build Docker image locally
  - verify local Docker run exposes the OpenEnv server
  - push to Hugging Face Space with `openenv push --repo-id <hf-username>/civicflow-env`
  - validate the deployed Space at `/health`, `/docs`, `/web`
  - document the exact install command from the Space “Use this Space” menu in the README

### Phase 2 — Tasks, prompts, baselines, SFT, GRPO, and demo assets (Person B)
Person B owns all task content, all trainer-side reward aggregation, all baselines, SFT/GRPO, and all training artifacts. Person A must not edit these after the split.

- Author the three Tiny tasks as static JSON/YAML fixtures matching the bootstrap schema.
- For each task, define:
  - initial state
  - target outputs
  - hard constraints
  - one curveball event and trigger step
  - gold affected-set labels
  - terminal success conditions
- Write the planner prompt format once and reuse it across baseline, SFT, and GRPO:
  - system prompt describes the planner role and the requirement to use only tool actions
  - user prompt contains task briefing, current observation, and explicit action-output expectations
  - all generated actions must be tool calls, not free-form plans
- Build baseline evaluation first, before any training:
  - zero-shot baseline on the Tiny task set
  - structured-prompt baseline using the same model and stronger formatting constraints
  - collect failure cases and keep them as demo examples
- Prepare a small SFT warm-start dataset from expert trajectories:
  - 8–12 good trajectories total across the 3 Tiny tasks
  - include legal action sequencing, one curveball revision example, and one recovery-from-near-failure example
  - format the SFT examples so the model learns the exact tool-calling style used later in GRPO
- Run SFT with Unsloth first:
  - use LoRA/QLoRA path for speed
  - save adapters cleanly
  - test inference immediately after saving
  - do not merge adapters naively into an upcast base model
- After SFT is verified, run GRPO with the OpenEnv wrapper:
  - use `GRPOTrainer`
  - use `environment_factory` so each generation gets its own environment instance
  - ensure `max_completion_length` is large enough for multi-turn episodes
  - set batch/concurrency to match the server concurrency limits
- Keep reward functions trainer-side and entirely deterministic from env metrics:
  - `reward_legality = -1 * illegal_action`
  - `reward_constraints = -1 * constraint_violations`
  - `reward_progress = normalized progress_score`
  - `reward_replanning = F1 from affected-set precision/recall`
  - `reward_revision_discipline = -1 * unnecessary_change_count`
  - `reward_terminal = +1 if final_valid_plan else 0`
- Make reward shaping minimal and stable:
  - primary success criterion is final valid plan
  - intermediate reward only supports learning, not replaces success
  - no free-form textual judge scores
- Log and monitor exactly what the guide asks for:
  - overall reward
  - each reward column separately
  - final valid-plan rate
  - timeout rate
  - illegal-action rate
  - sampled generations every few updates
- Build the final demo package:
  - one before/after episode on the same Tiny task
  - reward-verifier output side by side
  - one concise explanation of safeguards against reward hacking

### Phase 3 — Integration path and run order
This is the only allowed order; do not reorder it.

1. Install OpenEnv core and scaffold the repo.
2. Make local `reset/step/state` work with one dummy task.
3. Freeze schemas and handoff contract.
4. Person A implements the deterministic environment and verifiers.
5. Person B creates the three Tiny tasks and baseline prompts.
6. Run baseline evaluation locally.
7. Deploy the environment to a Hugging Face Docker Space.
8. Re-run the same baseline against the deployed environment.
9. Build and validate the SFT dataset.
10. Run SFT with Unsloth.
11. Verify post-SFT inference on all three Tiny tasks.
12. Run a tiny GRPO smoke run against the stable environment.
13. Inspect generations for hacking or malformed actions.
14. Only then increase steps/batch/concurrency modestly.
15. Save adapters/checkpoints, verify inference again, and build the demo.

### Phase 4 — Repo and interface structure
Use this layout exactly so both people can work without overlap.

- Environment package:
  - OpenEnv scaffold root with `models.py`, `client.py`, `openenv.yaml`, `server/`, `README.md`
- Task data:
  - `data/tasks/tiny_a.*`, `tiny_b.*`, `tiny_c.*`
  - `data/gold_labels/*`
- Training:
  - `training/baselines/`
  - `training/sft/`
  - `training/grpo/`
  - `training/eval/`
- Outputs:
  - `outputs/logs/`
  - `outputs/evals/`
  - `outputs/plots/`

Contract rules:
- Server owns state transitions and metric emission.
- Task files only define scenario data and gold labels.
- Training code never recomputes legality or constraint logic; it only consumes emitted metrics.
- No one changes action or observation schemas after bootstrap freeze.

### Phase 5 — Deployment and operational instructions
These are required and should be written into the repo README exactly.

- Local setup:
  - install Python environment
  - `pip install openenv-core`
  - `pip install -e .`
  - run server locally with `uv run server --host 0.0.0.0 --port 8000`
- Local validation:
  - confirm `http://localhost:8000/health`
  - confirm `http://localhost:8000/docs`
  - confirm `http://localhost:8000/web`
  - run one client reset and one legal step
- Docker:
  - build image from the OpenEnv server Dockerfile
  - run container locally and repeat the same health/docs/web checks
- Hugging Face Space:
  - login to Hugging Face CLI
  - from the environment root run `openenv push --repo-id <hf-username>/civicflow-env`
  - wait for Space build
  - validate remote `/health`, `/docs`, and `/web`
  - record the pip install command shown by the Space for reproducibility
- Training connectivity:
  - support `ENV_BASE_URL` so training can switch between local and Space deployment without code edits
  - use local env for development and the Space for reproducibility/demo
- Concurrency:
  - match `max_concurrent_envs` to or above GRPO generation batch size
  - test with a small parallel connection smoke test before real training

## Test Plan

- Bootstrap tests:
  - environment scaffolds successfully
  - `reset`, `step`, `state` return typed results
  - one sample action round trip works from client to server
- Verifier tests:
  - illegal action gets penalized but episode survives
  - utility overflow is detected
  - road-access violation is detected
  - greenery/amenity shortfall is detected
  - curveball affected-set metrics are correct on one hand-labeled example
- Deployment tests:
  - local server works
  - Docker image works
  - Hugging Face Space works
  - remote client can reset and step
- Training tests:
  - zero-shot baseline runs on all 3 Tiny tasks
  - SFT checkpoint loads and generates valid tool calls
  - GRPO smoke run completes without session-limit failures
  - reward columns are logged separately
  - sampled generations show no obvious hacking or malformed outputs
- Demo acceptance:
  - one task clearly shows baseline failure and post-training improvement
  - README links to Space, training notebook, and result plots
  - all final artifacts can be re-run by judges from the documented commands

## Guide Compliance (points 0–21)

- `0` Build order follows the official stack exactly: environment -> verifier/reward -> trainer -> Unsloth -> deployment.
- `1` Narrow, verifiable Tiny task only.
- `2` RL loop is explicit: prompt -> action -> env -> reward -> update.
- `3` SFT warm start is mandatory before GRPO.
- `4` Environment is designed before any trainer code.
- `5` OpenEnv is scaffolded first and kept as the primary interface.
- `6` Start with Tiny tasks only.
- `7` Use multiple independent reward components.
- `8` Add anti-hacking checks, time limits, no mutable globals, and inspect outputs.
- `9` Use lightweight process-aware signals via per-step verifier metrics.
- `10` Training stack is OpenEnv + TRL + Unsloth.
- `11` Use deterministic RLVR-style rewards and GRPO.
- `12` Keep inference fast by using Tiny tasks and a 3B model.
- `13` Deploy the environment early, before serious training.
- `14` Do not scale until local and remote envs are stable.
- `15` Monitor reward columns, success columns, timeouts, and generations.
- `16` Save LoRA/QLoRA outputs safely and verify inference immediately.
- `17` Team split is adapted to two people after bootstrap.
- `18` The implementation order matches the one-day execution plan.
- `19` The deliverables target what judges find compelling: objective verifier, improvement, safeguards, reproducibility, demo.
- `20` The PS remains a Theme 2 long-horizon planning environment.
- `21` The plan explicitly avoids the common mistakes list.

## Assumptions

- The project is a new greenfield OpenEnv environment repo, not a patch to an existing codebase.
- `Qwen/Qwen2.5-3B-Instruct` is available and acceptable for SFT/GRPO.
- Tiny-only scope is the required MVP; Medium is out of scope for implementation.
- No LLM judge is used anywhere in the core reward pipeline.
- Person A does the initial bootstrap alone and owns all server-side code permanently after the split.
- Person B owns all task fixtures, training, plots, and demo artifacts permanently after the split.
