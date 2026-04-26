<div align="center">

# 🏙️ CivicFlow

### *Teaching a Small Language Model to Think Like a City Planner*

**Meta × Hugging Face OpenEnv Hackathon — India 2026**  
Theme #2 · Long-Horizon Planning & Instruction Following  
Secondary: Theme #3.1 · Professional Tasks

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct-orange)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Framework](https://img.shields.io/badge/Training-TRL%20%2B%20Unsloth-green)](https://github.com/unslothai/unsloth)
[![Environment](https://img.shields.io/badge/Environment-OpenEnv-purple)](https://github.com/openenv)

</div>

---

## The Problem With "Smart" Planning

Every major city-planning AI paper of the last three years tells roughly the same story.

The model is given a planning brief. It produces a beautiful, confident, well-structured response — mixed-use zoning, greenery targets, service coverage, transit access — everything a junior planner would tick off a checklist. The evaluators nod.

Then someone asks: *"What happens when the flood-risk boundary expands at step 14, after you've already approved residential on Block B7?"*

The model freezes, loops, or starts over. It has no memory of why Block B7 was chosen. It cannot surgically revise only the affected parcels. It rewrites the whole plan — or worse, quietly ignores the constraint.

This is not a retrieval problem. It is not a context-length problem in the traditional sense. It is a **structured state-tracking problem under evolving constraints over long decision horizons** — and current LLMs, especially small ones, are genuinely bad at it.

---

## What the Research Says

### LLMs in Urban Planning: Promising But Fragile

Recent work has explored using LLMs as urban planning agents — but the results expose clear failure modes that we directly target:

- **UrbanLLM / PlanBench** studies show that while frontier models (GPT-4, Claude) can produce plausible city plans in one-shot settings, they fail dramatically on constraint satisfaction when plans span more than ~8 sequential decisions. Violation rates spike 3-5× on longer episodes.

- **CityGPT and similar works** demonstrate that zero-shot LLM agents struggle to maintain legal consistency across zoning → permitting → infrastructure dependencies. The model "forgets" earlier commitments within the same episode.

- **Multi-step spatial reasoning benchmarks** (e.g., SpatialBench, PlanEval) confirm that LLMs reduce complex planning to pattern matching against training examples — they cannot generalize to novel constraint combinations they haven't seen verbatim.

### The Long-Context Zero-Shot Wall

The harder failure is documented across multiple RLVR and long-horizon benchmarks:

> **SLMs and LLMs both suffer significant performance collapse under zero-shot, long-context, multi-constraint planning.**

- Studies on Qwen2.5, Mistral, and Llama-3 series show that zero-shot performance on structured constraint-satisfaction tasks degrades sharply when the JSON observation exceeds ~1.5K tokens of interacting constraints — even though the models can technically process the context.

- The models can *read* the constraints. They cannot *reason through their interactions* step by step, especially when early decisions have non-obvious downstream effects.

- GRPO-trained models on verifiable reward signals (DeepSeek-R1 style) show the most robust long-horizon improvement — but only when the reward is dense, deterministic, and well-decomposed. Sparse terminal rewards alone are insufficient.

CivicFlow is our direct response to this gap.

---

## The Core Idea: Teach the Model to *Plan*, Not Just *Comply*

Most planning benchmarks treat the model as a **constraint checker** — give it a state, ask if action X is legal. That's too easy and doesn't test the capability we care about.

CivicFlow treats the model as a **city planner**:

- It must *maintain* a valid district across multiple sequential phases
- It must *balance* housing, economic uses, civic amenities, greenery, and infrastructure — simultaneously
- It must *adapt surgically* when a curveball fires (flood-risk reclassification, protection order, infrastructure capacity cut)
- It must *preserve* unaffected parts of the plan — not just find a new valid state

The distinction matters. A constraint-checking agent can blindly redevelop everything after a curveball and still satisfy the terminal condition. A real planner redevelops *only what changed*, protects what's already correct, and rebalances only the affected subgraph.

**CivicFlow rewards the latter and penalizes the former.**

---

## What Makes CivicFlow Different

### 🗺️ Graph-Structured World State, Not a Grid

The environment is a **multi-layer symbolic graph**:

- **Block graph** — each block carries zoning, land use, road access, infrastructure demand, district membership, flood-risk and protection flags, and future designation
- **Infrastructure zones** — water, sewer, power, and road capacity with real allocation tracking
- **Adjacency graph** — BFS/graph traversal to compute spatial service reachability (not just global counts)
- **District layer** — named districts with independent greenery, amenity, and economic-mix targets

### 🔍 BFS-Powered Spatial Reasoning

Service coverage is not a simple global count. CivicFlow uses **BFS on the block adjacency graph** to compute whether every housing block is within `k` hops of a required amenity. This means:

- A school on the far side of the district doesn't count for blocks it can't reach within the service radius
- Amenity placement actually matters spatially — the model must reason about *where* to place things, not just *whether* to place them
- The `spatial_service_score` and `spatial_service_gap_count` metrics capture this topology-aware coverage

This is the difference between a real planner and a spreadsheet.

### 📐 Grounded in Real-World Planning Frameworks

CivicFlow constraints are not invented reward-shaping. They are grounded in four internationally recognized planning frameworks, translated into deterministic benchmark rules:

| Framework | What It Contributes to CivicFlow |
|---|---|
| **UN-Habitat** *Five Principles of Neighbourhood Design* | Mixed land use, district connectivity, no over-specialized districts, compact/integrated planning |
| **WHO** *Urban Green Spaces and Health* | Minimum greenery ratio, park access requirements, flood-buffer and ecological-corridor curveballs |
| **OECD** *Cities for All Ages* / *Walkable Cities* | District-level access to school/clinic/grocery/park/transit, age-inclusive access logic |
| **ITDP TOD Standard** | Walk/connect/transit/mix/compact principles, district transit access, short-walk service radius |

These sources were used to design constraints, not as raw training data. They justify why CivicFlow is not an arbitrary toy.

### 🏗️ Multi-Phase Episode Structure

Episodes unfold in **planning phases** — not a single dump of all blocks:

1. **Base Structure** — establish layout, allocate land use, preserve protected/flood-risk land
2. **Services & Access** — place school/clinic/grocery/fire/transit/park amenities, satisfy district access obligations
3. **Infrastructure Balance** — keep water/sewer/power feasible, preserve greenery ratio, maintain district mix
4. **Replanning** — respond to curveballs, revise only affected subgraph, avoid unnecessary changes
5. **Final Validation** — satisfy all global and district-level targets

New blocks are only revealed when the current phase targets are met. The model cannot rush ahead — it must master each phase before the next is unlocked.

---

## The Verifier: Deterministic Truth

There is **no LLM judge in the reward loop**. All scoring is deterministic code.

The verifier is a stack of independent checkers, each emitting canonical metrics:

```
apply_action()          → legality check + state mutation
compute_metrics()       → 27 canonical metrics (METRIC_KEYS)
compute_reward_components() → 10 reward components (REWARD_COMPONENT_KEYS)
```

**Canonical metrics include:**
- `illegal_action`, `constraint_violations`, `infra_overflow_count`
- `amenity_shortfall_count`, `greenery_shortfall`, `use_target_shortfall_count`
- `district_service_gap_count`, `district_green_gap_count`, `district_mix_gap_count`
- `spatial_service_gap_count`, `designation_violation_count`
- `accessibility_score`, `land_use_balance_score`, `district_coverage_score`
- `affected_set_precision`, `affected_set_recall`, `unnecessary_change_count`
- `final_valid_plan`, `phase_completion_score`, `progress_score`

---

## The Reward Function

The reward is derived entirely from canonical verifier metrics — no separate scoring system for RL vs. evaluation. This keeps the benchmark honest.

### Reward Components

| Component | Signal | What It Teaches |
|---|---|---|
| `legality` | `−1.0` per illegal action + escalating penalty for repeats | Don't take actions the environment rejects |
| `constraints` | `−weight × violation_count` | Respect hard rules at every step |
| `accessibility` | Proportional to `accessibility_score` | Place amenities where people can reach them |
| `land_use_balance` | Proportional to `land_use_balance_score` | Balance housing, economic, civic, and green uses |
| `district_quality` | Derived from district service/green/mix gaps | Satisfy district-level targets, not just global totals |
| `phase_progress` | Proportional to `phase_completion_score` | Complete phases in order; don't skip ahead |
| `progress` | Delta of `progress_score` per step | Make incremental progress toward targets each step |
| `replanning` | F1 from `affected_set_precision` × `recall` | Revise the *right* blocks, preserve the *right* ones |
| `revision_discipline` | `−1 × unnecessary_change_count` | Don't touch what doesn't need changing |
| `terminal` | Large bonus on `final_valid_plan = 1` | The whole episode is about reaching a valid end state |

### Why This Decomposition Matters

Earlier designs used only `legality + constraints + progress + replanning F1 + terminal`. That was too narrow — it didn't distinguish a plan that *technically* satisfied targets from one that was genuinely well-balanced and district-aware.

The updated reward adds:
- **`district_quality`** — forces the model to care about local access, not just global counts
- **`land_use_balance`** — penalizes over-specialization (all housing, no economy; all green, no services)
- **`phase_progress`** — gives the model a curriculum signal; early phases reward early
- **`accessibility`** — the BFS-based spatial score, not just amenity count

The reward cannot be gamed by flooding the map with parks or spamming housing — every component pushes back against a different exploit.

---

## Baselines: What Happened Before Training

We ran two baseline conditions against Qwen2.5-3B-Instruct (the same base model used for fine-tuning) to measure the gap that training must close.

### Baseline 1 — Pure Zero-Shot (No Hints)

We gave the model the raw observation: all blocks at once, all constraints listed, no guidance on what to do next.

**What happened:** Complete collapse on even Tiny tasks.

The model cannot process a structured JSON observation with 6+ interacting constraints and produce a legal, dependency-respecting action sequence reliably. It falls into loops (repeating the same action), produces malformed JSON, ignores zoning prerequisites (trying to `develop` before `set_zoning`), or attempts to develop protected blocks. Constraint violations are near-maximal from step 1.

This is the **long-context zero-shot failure** documented in the literature. The model can read the constraints individually. It cannot reason through their interactions to produce a correct action.

### Baseline 2 — Greedy Heuristic (Privileged Expert)

We implemented a hand-coded greedy planner that reads `WorldState` directly (full ground truth access — not available to any model):

**Strategy:**
1. **Triage** — bucket blocks: must-be-open (protected/flood-risk), no-road-access, available
2. **Allocate** — greedily assign roles: parks → civic/amenities → housing → retail → office → workshop → greenery floor
3. **Emit** — produce actions in dependency order: `defer` → `set_zoning` → `develop`/`assign_amenity`/`reserve_open_space`
4. **Replan** — when a curveball fires, clear invalidated blocks with `redevelop`, re-triage, fill lost roles

The heuristic has **privileged access** (direct WorldState, not just the observation) and is a non-ML reference ceiling. On Tiny tasks it reliably produces valid plans. On Medium/Hard tasks, local greedy planning begins to degrade — district-level coupling requires non-local reasoning the heuristic cannot do.

### What Giving the Model "Limited Hints" Showed

When we gave the LLM a **`recommended_action` pre-filled** in the observation (the heuristic's suggestion for the current step, effectively a hint), performance improved substantially — even without any fine-tuning. This confirms:

- The model's *reasoning* is not the blocker on Tiny tasks; its *action formatting and constraint-tracking* is
- With just a hint about *what to do next*, the model can mostly execute correctly
- This means the SFT + GRPO pipeline has a clear ceiling to aim for: the model should eventually internalize what the heuristic is hinting, without needing the hint

### Baseline Comparison

| Condition | `tiny_a` | `medium_a` | `medium_b` | `hard_a` |
|---|---|---|---|---|
| **Heuristic (privileged ceiling)** | | | | |
| **Qwen2.5-3B Zero-Shot (no hint)** | | | | |
| **Qwen2.5-3B With Hint** | | | | |
| **After SFT** | | | | |
| **After SFT + GRPO** | | | | |

> *Fill in reward / final_valid_plan / violation_count from your eval runs.*

---

## Repository Structure

```
civicflow_env/                          # OpenEnv environment package
├── models.py                           # CivicflowAction / CivicflowObservation (frozen contract)
├── tasks.py                            # Task loader → reads data/tasks/*.json
├── data/tasks/
│   ├── tiny_a.json                     # 6 blocks, 2 phases            [easy]
│   ├── medium_a.json                   # 8 blocks, 2 phases, infra     [medium]
│   ├── medium_b.json                   # 25 blocks, 3 phases, 3 dist.  [medium]
│   ├── hard_a.json                     # 50 blocks, 3 phases, 5 dist.  [hard]
│   └── *_nohint.json                   # no-hint variants (no recommended_action)
└── server/
    ├── civicflow_env_environment.py    # reset() / step() / state()
    ├── verifier.py                     # Deterministic legality + scoring (ground truth)
    └── state.py                        # WorldState, Block, InfraZone dataclasses

training/
├── baselines/
│   └── heuristic.py                    # Greedy expert baseline + SFT trace generator
├── sft/
│   ├── generate_from_heuristic.py      # Re-generates heuristic_rollouts.jsonl
│   ├── prepare_sft_data.py             # Merges sources → sft_final.jsonl
│   ├── heuristic_rollouts.jsonl        # 307 heuristic expert steps
│   ├── sft_final.jsonl                 # 332 examples (307 heuristic + 25 syndata)
│   └── civicflow_sft_colab.ipynb       # ← SFT Colab notebook (Unsloth / TRL)
└── rl/
    └── civicflow_grpo_colab.ipynb      # ← GRPO Colab notebook (TRL GRPOTrainer)
    eval/
    ├── run_inference.py                # Guided eval (with hints + recommended_action)
    ├── run_inference_nohint.py         # No-hint eval (raw obs, model reasons from scratch)
    └── llm_judge.py                    # Optional LLM-as-judge quality scoring

syndata.json / syndata1.json            # Synthetic SFT data (UN/WHO/OECD-grounded scenarios)
models/
└── qwen2.5-3b/                         # Base model weights
```

---

## Training Pipeline

### How We Prepared the Data

The SFT dataset was assembled from three sources, reflecting real planning principles:

**1. Heuristic Rollouts (307 steps)**  
The greedy heuristic planner runs on all phased tasks and saves complete episode trajectories. Each step becomes one SFT example: `(system_prompt, observation) → expert_action`. These are correct by construction — the verifier accepts every action — so no manual labelling is needed.

**2. Synthetic Data (25 examples)**  
We studied UN-Habitat neighbourhood design principles, WHO urban green space guidance, OECD proximity frameworks, and ITDP TOD standards to understand what *good* planning behavior looks like beyond the heuristic's greedy logic. We authored synthetic trajectories covering:
- Standard valid planning with mixed-use balance
- Service/greenery balancing under tight constraints
- Curveball revision (flood-risk reclassification, protection order, infrastructure cut)
- Recovery from near-miss states (partial violations that need surgical repair)

These were cleaned with a SHA-256 deduplication pipeline and validated against the verifier.

**3. Merged Dataset (`sft_final.jsonl`)**  
`prepare_sft_data.py` merges all sources, deduplicates by content-stable hash, and outputs chat-template formatted JSONL ready for TRL SFTTrainer.

**Final dataset: 332 examples across Tiny → Hard tasks.**

---

### Step 1 — SFT Warm Start

> **Notebook:** `training/sft/civicflow_sft_colab.ipynb`  
> **Platform:** Google Colab (A100 or T4)  
> **Stack:** Unsloth + TRL SFTTrainer + LoRA

```python
model_id   = "Qwen/Qwen2.5-3B-Instruct"
data_path  = "training/sft/sft_final.jsonl"
output_dir = "./models/civicflow-sft"

per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_train_epochs            = 3
learning_rate               = 2e-4
max_seq_length              = 2048
lora_r, lora_alpha          = 16, 32
```

The training format follows the standard chat template:

```json
{"messages": [
  {"role": "system",    "content": "You are a city planner. Output ONLY the JSON shown in recommended_action..."},
  {"role": "user",      "content": "{\"block_states\": [...], \"recommended_action\": {...}, ...}"},
  {"role": "assistant", "content": "{\"action_type\": \"develop\", \"block_id\": \"B1\", \"use\": \"housing\"}"}
]}
```

SFT teaches the model the **action grammar and planning order** — it learns that `set_zoning` must precede `develop`, that protected blocks can only be open space, that amenity placement requires compatible zoning. This is the essential warm-start before RL.

```bash
# Regenerate SFT data (if tasks changed)
python training/sft/generate_from_heuristic.py

# Merge sources
python training/sft/prepare_sft_data.py \
    --in training/sft/heuristic_rollouts.jsonl \
    --out-jsonl training/sft/sft_final.jsonl \
    --out-json  training/sft/sft_final.json \
    --report    training/sft/sft_merge_report.json
```

---

### Step 2 — GRPO Reinforcement Learning

> **Notebook:** `training/rl/civicflow_grpo_colab.ipynb`  
> **Starting point:** SFT checkpoint  
> **Stack:** TRL GRPOTrainer + OpenEnv environment wrapper

GRPO (Group Relative Policy Optimization) lets the model interact with the **live environment** and optimize the multi-component reward signal — not just imitate the heuristic, but discover policies the heuristic cannot.

```python
model_id        = "./models/civicflow-sft"   # start from SFT checkpoint
num_generations = 4                           # rollouts per prompt
max_new_tokens  = 64
learning_rate   = 5e-6
kl_coef         = 0.05

GRPO_TASKS = ["tiny_a", "medium_a", "medium_b", "hard_a"]
```

Each GRPO step:
1. Sample `num_generations` action completions from the model
2. Execute each in its own `CivicflowEnvironment` instance (full concurrent isolation)
3. Compute the 10-component reward vector from the verifier
4. Compute group-relative advantages and update the policy

The no-hint score improvement post-GRPO is the key metric — it demonstrates the model has *learned* the planning rules and constraint interactions, not just memorized hint patterns.

---

### Step 3 — Evaluation

```bash
# Guided eval (with hints — ceiling condition)
python training/eval/run_inference.py --model ./models/civicflow-grpo --task all --out results/grpo_hint.json

# No-hint eval (zero-shot reasoning — key generalization metric)
python training/eval/run_inference_nohint.py --model ./models/civicflow-grpo --task all --out results/grpo_nohint.json
```

| Metric | What It Measures |
|---|---|
| `final_valid_plan` rate | End-to-end planning success |
| `constraint_violations_final` | Hard constraint adherence |
| `illegal_action_count` | Action grammar correctness |
| `affected_set_precision / recall` | Surgical replanning quality |
| `unnecessary_change_count` | Revision discipline |
| `progress_score` | Partial credit (how far toward targets) |

---

## Task Difficulty Ladder

| Tier | Blocks | Phases | Districts | Curveballs | What Tests |
|---|---|---|---|---|---|
| **Tiny** (`tiny_a`) | 6 | 2 | 1 | 1 | Basic planning grammar; heuristic succeeds; zero-shot struggles |
| **Medium A** (`medium_a`) | 8 | 2 | 1 | 1 | Infra constraints; capacity planning |
| **Medium B** (`medium_b`) | 25 | 3 | 3 | 2 | District-level targets; greenery/amenity balance |
| **Hard** (`hard_a`) | 50 | 3 | 5 | 3 | Multi-district coupling; sequential shocks; heuristic degrades |

**Intended behavior:**
- *Tiny* → heuristic succeeds, zero-shot fails, SFT improves
- *Medium* → heuristic degrades (local greedy becomes brittle), trained model should track district targets
- *Hard* → heuristic clearly struggles; only GRPO-trained model maintains replanning discipline across multiple interacting shocks

---

## Key Planning Rules the Model Must Learn

These are encoded in SFT data and enforced by the verifier:

1. **Zone before develop** — `set_zoning` must precede `develop` on any block
2. **Designation constraints** — blocks with `future_designation` restrict allowed uses:
   - `civic_anchor` → only `institutional` use; school/clinic/fire/park amenities
   - `green_corridor` → only `park` use
   - `transit_corridor` → only transit/park amenities, no develop
3. **Infra upgrade before overflow** — call `upgrade_infrastructure` before demand would exceed capacity
4. **Reserve first** — `reserve_open_space` on no-road-access and protected blocks before any other action
5. **Phased revelation** — new blocks revealed only when current phase targets are met (progress ≥ 1.0, violations = 0)
6. **Surgical replanning** — after curveball, `redevelop` only affected blocks, preserve the rest

---

## Final Comparison Table

> *Fill in from eval runs after training.*

| Model | tiny_a (hint) | medium_a (hint) | tiny_a (no hint) | medium_a (no hint) | hard_a (no hint) |
|---|---|---|---|---|---|
| Heuristic ceiling (privileged) | | | | | |
| Qwen2.5-3B base (zero-shot) | | | | | |
| Qwen2.5-3B + SFT | | | | | |
| Qwen2.5-3B + SFT + GRPO | | | | | |

---

## Setup & Local Run

```bash
# 1. Clone and install
pip install openenv-core
pip install -e ./civicflow_env

# 2. Start local server
cd civicflow_env
uv run server --host 0.0.0.0 --port 8000

# 3. Validate
curl http://localhost:8000/health
# → open http://localhost:8000/docs
# → open http://localhost:8000/web

# 4. Run heuristic baseline
python training/baselines/heuristic.py --task all --verbose

# 5. Run zero-shot LLM baseline
python training/eval/run_inference_nohint.py \
    --model ./models/qwen2.5-3b --task all \
    --out results/base_nohint.json
```

---

## Design Philosophy

CivicFlow is not a prompt engineering exercise or a frontier-model demo. It is a **verifiable RL environment** built on three principles:

**Determinism over vibes.** Every reward signal comes from code that you can read, audit, and re-run. The verifier is the single source of truth for both training and evaluation. We never invented a different notion of success for RL than the one used for evaluation.

**Grounded constraints, not arbitrary rules.** The planning constraints in CivicFlow reflect real international guidance — UN-Habitat, WHO, OECD, ITDP. Not because we're implementing a national planning code, but because the constraints should be justifiable to someone who knows what good urban planning looks like.

**Planning, not compliance.** The task is not to classify a single action as legal or illegal. It is to maintain a valid, balanced, liveable district plan across a long, evolving, multi-phase episode — the same thing a real city planner has to do.

---

<div align="center">

*Built for the Meta × Hugging Face OpenEnv Hackathon, India 2026.*

</div>
