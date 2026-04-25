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

**Long-horizon municipal planning under evolving constraints — an OpenEnv environment for RL training.**

| | |
|---|---|
| **HF Space** | https://huggingface.co/spaces/Aaryan369/civicflow-env |
| **SFT model** | https://huggingface.co/Aaryan369/civicflow-sft-qwen2.5-3b |
| **GRPO model** | https://huggingface.co/Aaryan369/civicflow-grpo-qwen2.5-3b |
| **Theme** | #2 Long-horizon planning + instruction following |

---

## What the agent does

The agent plays a **municipal planner** on a grid of developable **blocks** connected by roads and shared **infrastructure zones** (water/sewer/power capacity pools). Each episode runs through five planning phases:

```
base_structure → services_and_access → infrastructure_balance → replanning → final_validation
```

The environment fires **policy curveballs** mid-episode (flood risk appears, infrastructure capacity is cut, district targets shift) forcing the agent to replan only the affected subgraph while preserving the rest of the plan. The verifier scores both the quality of the fix *and* the discipline of the revision (over-touching clean blocks is penalised).

### Why this is hard

- **Long horizon**: hard tasks run up to 300 steps with 6 sequential curveballs
- **Partial observability**: the agent only sees an observation summary, not the full state graph
- **Compositional constraints**: greenery ratio + district mix + amenity coverage + infra capacity must all hold simultaneously
- **Scoped replanning**: curveball response must be minimal and precise (precision+recall scored)

---

## Training pipeline

### SFT warm-start

Heuristic-generated rollouts produce 321 expert trajectories across all 6 tasks (tiny/medium/hard), merged with 25 synthetic examples → **346 total SFT examples**.

![SFT training curve](assets/sft_training_curve.png)

### GRPO reinforcement learning

Starting from the SFT checkpoint, GRPO optimises directly against the verifier reward. The reward function is the same decomposed signal the environment emits at each step — no separate reward model needed.

![GRPO reward curve](assets/grpo_reward_curve.png)

---

## Results

| Model | Valid plan % | Illegal action % | Mean progress | Curveball F1 |
|---|---|---|---|---|
| Zero-shot Qwen2.5-3B | ? | ? | ? | ? |
| SFT (heuristic warm-start) | ? | ? | ? | ? |
| SFT + GRPO | ? | ? | ? | ? |
| Heuristic (ceiling) | 100% (tiny) | 0% | 1.00 (tiny) | ~0.85 |

*Fill in after running `training/eval/eval_model.py --mode all`.*

---

## Action vocabulary

| Action | Effect |
|---|---|
| `set_zoning(block_id, zone)` | Designate land-use class (residential, commercial, industrial, mixed, civic, open_space) |
| `develop(block_id, use)` | Commission construction — consumes infra capacity, requires zoning + road access |
| `reserve_open_space(block_id)` | Keep block undeveloped (greenery / floodplain / setback) |
| `upgrade_infrastructure(infra_zone, infra_type, capacity)` | Add water/sewer/power/road capacity |
| `assign_amenity(block_id, amenity_type)` | Site school / clinic / grocery / park / fire / transit |
| `redevelop(block_id, use)` | Change a developed block's use — the replanning lever under curveballs |
| `defer(block_id, phase_id)` | Push a decision to a later planning phase |

---

## Reward decomposition

`observation.last_reward_components` exposes 10 named components:

```
legality   constraints   accessibility   land_use_balance   district_quality
phase_progress   progress   replanning   revision_discipline   terminal
```

The `replanning` and `revision_discipline` components only activate after the first curveball fires, creating a natural curriculum: the agent must learn to plan correctly before it is tested on adaptive replanning.

---

## Task difficulty tiers

| Tier | Tasks | Blocks | Curveballs | Max steps |
|---|---|---|---|---|
| Tiny | tiny_a, tiny_b, tiny_c | 10 | 0 | 50 |
| Medium | medium_a | 25 | 1 | 100 |
| Hard | hard_a | 40 | 3 | 200 |
| Hard | hard1 | 75 | 6 | 300 |

---

## Verifier metrics (frozen contract)

Every step emits these 19 keys in `observation.last_metrics`:

```
illegal_action            constraint_violations    infra_overflow_count
amenity_shortfall_count   greenery_shortfall       progress_score
use_target_shortfall_count district_service_gap_count district_green_gap_count
district_mix_gap_count    accessibility_score      land_use_balance_score
district_coverage_score   phase_completion_score
affected_set_precision    affected_set_recall      unnecessary_change_count
final_valid_plan          timeout
```

---

## Quick start

```bash
# Local server
pip install -e .
server --port 8000
curl http://localhost:8000/health

# Or connect to the HF Space
curl https://aaryan369-civicflow-env.hf.space/health
```

```python
from civicflow_env.client import CivicflowEnv
from civicflow_env.models import CivicflowAction

with CivicflowEnv(base_url="https://aaryan369-civicflow-env.hf.space").sync() as env:
    obs = env.reset().observation
    print(obs.briefing)
    obs = env.step(CivicflowAction(
        action_type="set_zoning", block_id="B1", zone="residential"
    )).observation
    print(obs.last_metrics["progress_score"])
```

---

## Repository layout

```
civicflow_env/           # OpenEnv environment package
├── models.py            # frozen Action/Observation schemas
├── client.py            # WebSocket client
├── tasks.py             # JSON task loader
├── data/tasks/          # tiny_a/b/c, medium_a, hard_a, hard1
├── assets/              # sft_training_curve.png, grpo_reward_curve.png
└── server/
    ├── app.py
    ├── civicflow_env_environment.py
    ├── verifier.py
    └── state.py

training/
├── baselines/heuristic.py          # greedy expert baseline + SFT data generator
├── sft/
│   ├── generate_from_heuristic.py  # produces heuristic_rollouts.jsonl
│   ├── sft_final.jsonl             # 346 SFT examples (tiny+medium+hard)
│   └── civicflow_sft_colab.ipynb  # SFT training notebook
├── rl/
│   └── civicflow_grpo_colab.ipynb # GRPO training notebook
└── eval/
    └── eval_model.py               # zero-shot vs SFT vs GRPO comparison
```

---

## Deployment

```bash
cd civicflow_env
openenv push --repo-id Aaryan369/civicflow-env
```

Switch trainer between local and deployed Space via `ENV_BASE_URL`:
```bash
ENV_BASE_URL=https://aaryan369-civicflow-env.hf.space python training/eval/eval_model.py
```
