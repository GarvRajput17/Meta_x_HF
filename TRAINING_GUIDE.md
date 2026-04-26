# CivicFlow — Training & Evaluation Guide

## Overview

The training pipeline has two stages:

1. **SFT (Supervised Fine-Tuning)** — warm-start the model on expert heuristic trajectories so it learns basic action grammar and planning order.
2. **GRPO (RL fine-tuning)** — let the model interact with the live environment and optimise reward via the GRPO objective.

After training, evaluate on **two conditions** to measure both guided and zero-shot capability:

| Condition | Runner | Tasks | What it tests |
|---|---|---|---|
| **With hints** | `run_inference.py` | `tiny_a`, `medium_a`, `medium_b`, `hard_a` | Phased revelation + `recommended_action` pre-filled |
| **Without hints** | `run_inference_nohint.py` | `tiny_a_nohint`, `medium_a_nohint`, `medium_b_nohint`, `hard_a_nohint` | All blocks at once, raw obs, model must reason from scratch |

---

## Repository Structure

```
civicflow_env/
├── models.py                        # CivicflowAction / CivicflowObservation (frozen contracts)
├── tasks.py                         # Task loader — reads data/tasks/*.json
├── data/tasks/
│   ├── tiny_a.json                  # 6 blocks, 2 phases  [easy]
│   ├── medium_a.json                # 8 blocks, 2 phases, infra constraints  [medium]
│   ├── medium_b.json                # 25 blocks, 3 phases, 3 districts  [medium]
│   ├── hard_a.json                  # 50 blocks, 3 phases, 5 districts  [hard]
│   ├── tiny_a_nohint.json           # same tasks without phases (no-hint variants)
│   ├── medium_a_nohint.json
│   ├── medium_b_nohint.json
│   └── hard_a_nohint.json
└── server/
    ├── civicflow_env_environment.py # reset / step / state
    ├── verifier.py                  # legality checking + scoring (ground truth)
    └── state.py                     # WorldState, Block, InfraZone dataclasses

training/
├── sft/
│   ├── heuristic_rollouts.jsonl     # 307 expert steps (primary SFT data)
│   ├── sft_final.jsonl              # merged training set (307 heuristic + 25 syndata)
│   ├── generate_from_heuristic.py   # re-generates heuristic_rollouts.jsonl
│   ├── prepare_sft_data.py          # merges multiple sources → sft_final.jsonl
│   └── civicflow_sft_colab.ipynb    # Colab SFT notebook (Unsloth / TRL SFTTrainer)
├── rl/
│   └── civicflow_grpo_colab.ipynb   # Colab GRPO notebook (TRL GRPOTrainer)
└── eval/
    ├── run_inference.py             # Guided eval (with hints + recommended_action)
    ├── run_inference_nohint.py      # No-hint eval (raw obs, no recommended_action)
    └── llm_judge.py                 # Optional LLM-as-judge quality scoring

models/
└── qwen2.5-3b/                      # Base model weights (local)
```

---

## Step 1 — Regenerate SFT Data (if tasks changed)

```bash
# Re-run heuristic on all phased tasks and rebuild sft_final.jsonl
python training/sft/generate_from_heuristic.py

# Optionally merge with additional syndata sources
python training/sft/prepare_sft_data.py \
    --in training/sft/heuristic_rollouts.jsonl \
    --out-jsonl training/sft/sft_final.jsonl \
    --out-json  training/sft/sft_final.json \
    --report    training/sft/sft_merge_report.json
```

Current dataset: **332 examples** across 4 tasks (tiny_a×12, medium_a×15, medium_b×90, hard_a×190, syndata×25).

---

## Step 2 — SFT Fine-Tuning

Use `training/sft/civicflow_sft_colab.ipynb` (Unsloth on Colab A100/T4).

Key settings:
```python
model_id   = "Qwen/Qwen2.5-3B-Instruct"   # or 7B if VRAM allows
data_path  = "training/sft/sft_final.jsonl"
output_dir = "./models/civicflow-sft"

# Recommended hyperparameters
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs            = 3
learning_rate               = 2e-4
max_seq_length              = 2048
lora_r, lora_alpha          = 16, 32
```

The training format is standard chat-template messages:
```json
{"messages": [
  {"role": "system",    "content": "You are a city planner. Output ONLY the JSON shown in recommended_action..."},
  {"role": "user",      "content": "{\"block_states\": [...], \"recommended_action\": {...}, ...}"},
  {"role": "assistant", "content": "{\"action_type\": \"develop\", \"block_id\": \"B1\", \"use\": \"housing\"}"}
]}
```

Save the merged adapter weights after SFT:
```bash
# In notebook, after training:
model.save_pretrained("./models/civicflow-sft")
tokenizer.save_pretrained("./models/civicflow-sft")
```

---

## Step 3 — GRPO RL Fine-Tuning

Use `training/rl/civicflow_grpo_colab.ipynb`. Start from the SFT checkpoint.

**Reward signal** — `CivicflowObservation.reward` (per-step scalar from verifier):
- `+` for progress toward use targets, amenity coverage, greenery ratio
- `+` bonus on valid plan completion
- `−` for illegal actions (wrong zone, infra overflow, designation violation)
- `−` budget penalty as resources deplete

**Environment setup in GRPO loop:**
```python
import os
from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
from civicflow_env.models import CivicflowAction

def rollout(prompt, task_id):
    os.environ["CIVICFLOW_TASK_ID"] = task_id
    env = CivicflowEnvironment()
    obs = env.reset()
    rewards = []
    # ... step loop, collect (action, reward) pairs for GRPO
```

**Tasks for GRPO** — use phased variants only (harder signal, more structured):
```python
GRPO_TASKS = ["tiny_a", "medium_a", "medium_b", "hard_a"]
```

Key settings:
```python
model_id        = "./models/civicflow-sft"   # start from SFT checkpoint
num_generations = 4                           # rollouts per prompt
max_new_tokens  = 64
learning_rate   = 5e-6
kl_coef         = 0.05
```

Save after GRPO:
```bash
model.save_pretrained("./models/civicflow-grpo")
tokenizer.save_pretrained("./models/civicflow-grpo")
```

---

## Step 4 — Evaluation

Run both conditions on each checkpoint to measure improvement:

### Condition 1: With Hints (guided)
```bash
# Heuristic baseline (ceiling)
python training/eval/run_inference.py --heuristic --task all

# Base model (before training)
python training/eval/run_inference.py --model ./models/qwen2.5-3b --task all --out results/base_hint.json

# After SFT
python training/eval/run_inference.py --model ./models/civicflow-sft --task all --out results/sft_hint.json

# After GRPO
python training/eval/run_inference.py --model ./models/civicflow-grpo --task all --out results/grpo_hint.json
```

### Condition 2: Without Hints (zero-shot reasoning)
```bash
# Heuristic baseline
python training/eval/run_inference_nohint.py --heuristic --task all

# Base model
python training/eval/run_inference_nohint.py --model ./models/qwen2.5-3b --task all --out results/base_nohint.json

# After SFT
python training/eval/run_inference_nohint.py --model ./models/civicflow-sft --task all --out results/sft_nohint.json

# After GRPO
python training/eval/run_inference_nohint.py --model ./models/civicflow-grpo --task all --out results/grpo_nohint.json
```

### Expected results table

| Model | tiny_a hint | medium_a hint | tiny_a nohint | medium_a nohint |
|---|---|---|---|---|
| Heuristic ceiling | ~+10.2 | ~+11.5 | ~+9.0 | ~+6.0 |
| Base Qwen2.5-3B | low / loops | low / loops | very low | very low |
| After SFT | moderate | moderate | slight gain | slight gain |
| After GRPO | near ceiling | near ceiling | higher than SFT | higher than SFT |

The no-hint score improvement post-GRPO demonstrates the model has *learned* the planning rules, not just memorised the hint pattern.

---

## Key Rules the Model Must Learn

These are encoded in the SFT data and rewarded by the environment:

1. **Zone before develop** — `set_zoning` must precede `develop` on any block.
2. **Designation constraints** — blocks with `future_designation` restrict allowed uses/amenities:
   - `civic_anchor` → only `institutional` use; school/clinic/fire/park amenities
   - `green_corridor` → only `park` use
   - `transit_corridor` → only transit/park amenities, no develop
   - `entertainment_zone` → only `institutional` use
3. **Infra upgrade before overflow** — call `upgrade_infrastructure(zone, type, capacity)` before the next development would push `water_alloc + demand > water_capacity`.
4. **Reserve no-road / protected blocks first** — `reserve_open_space` on blocks with `has_road_access=False` or `is_protected=True`.
5. **Phased revelation** — new blocks are only revealed when current phase targets are met (progress ≥ 1.0, violations = 0).

---

## Reward Components (from verifier.py)

| Key | Description |
|---|---|
| `progress_delta` | Incremental progress toward use/amenity targets |
| `amenity_coverage` | Spatial service coverage score |
| `greenery_ratio` | Green blocks / total blocks |
| `budget_efficiency` | Remaining budget ratio |
| `constraint_satisfaction` | No designation/infra violations |
| `plan_completion_bonus` | Large bonus on `final_valid_plan=1` |
| `illegal_action_penalty` | −1.0 base + escalating penalty per repeat |
