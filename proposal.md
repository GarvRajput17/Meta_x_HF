# CivicFlow — Long-Horizon Municipal Planning Under Evolving Constraints

**OpenEnv Hackathon (India 2026) — Proposal**

**Primary theme:** #2 — (Super) Long-Horizon Planning & Instruction Following  
**Secondary alignment:** #3.1 — Professional Tasks

---

## 1. One-line idea

We propose **CivicFlow**, an OpenEnv environment where an LLM acts as a municipal planner operating on a graph of **parcels, roads, and utility zones**, making sequential zoning and permit decisions under explicit legal, infrastructure, service-coverage, and budget constraints while adapting to policy changes over a long horizon.

---

## 2. Why this problem

LLMs are often good at producing plans that **sound plausible**, but they are brittle at:

- tracking evolving structured state,
- respecting hard constraints over many steps,
- revising only the affected parts of a plan after a change,
- avoiding illegal or contradictory actions,
- maintaining consistency across long-horizon decisions.

This is exactly the kind of capability gap the hackathon’s **Theme #2** is targeting: long-running planning beyond shallow next-token reasoning.

Unlike open-ended “good design” tasks, municipal planning can be formulated with:

- explicit graph state,
- explicit actions,
- deterministic transitions,
- rule-based legality checks,
- terminal validity checks,
- dense intermediate rewards.

That makes it a strong fit for **RL with verifiable rewards**.

---

## 3. Problem statement

An agent receives a municipal planning scenario defined over:

- a **parcel graph**,
- a **road network**,
- **utility capacities**,
- zoning and adjacency rules,
- public-service coverage rules,
- amenity and daily-needs requirements,
- greenery / open-space targets,
- budget and phasing limits.

The agent must produce a sequence of planning actions, such as:

- rezoning parcels,
- approving or denying permits,
- adding or upgrading road/utility support,
- reserving service coverage,
- deferring parcels to later phases,
- revising the plan when policy changes occur.

The episode is successful if the final municipal plan satisfies all hard constraints and meets target planning objectives with minimal unnecessary revisions.

---

## 4. Theme alignment

### Theme #2 — Long-Horizon Planning

This is the primary fit.

- The task unfolds across many dependent actions.
- Early bad decisions create downstream failures.
- Policy changes require partial replanning rather than full reset.
- The agent must track evolving state over long trajectories.

### Theme #3.1 — Professional Tasks

This is a strong secondary fit.

- Municipal/zoning/planning is a real professional workflow.
- The agent interacts with structured state and rule systems.
- Outputs can be validated programmatically instead of by vibes.

---

## 5. Environment design

### 5.1 State representation

The environment state is a multi-layer graph:

#### Parcel graph

Each parcel node contains:

- parcel id
- area
- current zoning
- current land use
- frontage / road access flag
- buildable / protected / flood-risk flags
- permit status
- phase assignment

#### Road graph

Each road/intersection node or edge contains:

- road segment id
- connectivity
- road type / class
- closure / active status
- connected parcels

#### Utility graph

Each utility zone contains:

- water capacity
- sewer capacity
- power capacity
- currently allocated demand
- served parcels

#### Service and liveability layer

For realistic modern-city planning tasks:

- school coverage
- fire-station coverage
- clinic / healthcare coverage
- grocery / daily-needs coverage
- park access coverage
- greenery / open-space coverage
- transit-stop accessibility

### 5.2 Action space

The agent does not write free-form essays as the primary output.  
It emits **structured planning actions**.

Core actions:

- `rezone(parcel_id, zone_type)`
- `approve_permit(parcel_id, use_type)`
- `deny_permit(parcel_id)`
- `upgrade_utility(zone_id, utility_type, amount)`
- `reserve_service(parcel_id, service_type)`
- `defer(parcel_id, phase_id)`

Medium-tier actions:

- `add_road(parcel_a, parcel_b)`
- `change_road_class(segment_id, new_class)`
- `reallocate_capacity(zone_id, amount)`
- `supersede_decision(decision_id, justification)`

### 5.3 Transition dynamics

Each action deterministically updates state.

Examples:

- rezoning changes parcel eligibility for future permits,
- permit approval consumes service and utility capacity,
- utility upgrades expand feasible development,
- policy changes can invalidate previously good plans.

This makes the environment a true sequential decision process rather than a static scoring task.

---

## 6. Verifier design

The verifier is **not one judge model**. It is a stack of deterministic checkers.

### 6.1 Action legality checker

Checks whether the chosen action is legal in the current state.

Examples:

- cannot approve residential permit without road access,
- cannot violate zoning compatibility,
- cannot exceed current utility capacity,
- cannot develop protected parcel.

### 6.2 State constraint checker

After every action, checks:

- utility overflow,
- disconnected developed parcels,
- illegal adjacencies,
- service coverage violations,
- amenity shortfall,
- greenery / open-space shortfall,
- poor daily-needs accessibility,
- budget overrun,
- phase violations.

### 6.3 Replanning / affected-scope checker

When a curveball happens, the environment labels which prior decisions should be revisited.

The agent is scored on:

- revising the right parts,
- preserving unaffected parts,
- avoiding under-revision,
- avoiding over-revision.

### 6.4 Terminal validity checker

At episode end:

- are planning targets met?
- are all hard constraints satisfied?
- is the plan deployable under current policy?

### 6.5 Optional eval-only LLM judge

If time allows, we add an **eval-only** rubric judge for:

- coherence of explanation,
- quality of rationale,
- human readability.

This is not required for the core reward loop.

---

## 7. Reward design

The reward should be mostly deterministic and hard to game.

### Core reward terms

- **R1: legality reward**
  - positive for legal actions
  - negative for illegal actions

- **R2: constraint satisfaction**
  - penalty for each active rule violation

- **R3: progress reward**
  - reward for moving toward target land-use, amenity, and liveability goals

- **R4: replanning F1**
  - precision/recall on affected decision set after policy change

- **R5: unnecessary change penalty**
  - penalize modifying unaffected parcels or infrastructure

- **R6: terminal success reward**
  - large positive reward if final plan is valid and complete

### Anti-gaming measures

- no reward for vague text without valid action payload,
- no reward for stalling,
- penalty for illegal but confident action chains,
- terminal penalty if planning targets are not met,
- reward only when state actually improves.

---

## 8. Difficulty tiers

### Easy / Tiny

Purpose: first runnable training and baseline tier

- 8–15 parcels
- 6–12 road segments
- 1–2 utility zones
- 4–6 hard constraints
- 1 curveball
- 8–20 actions

Example goals:

- create a small mixed-use neighborhood,
- preserve road access,
- stay within utility capacity,
- maintain one service-coverage rule,
- satisfy a minimum park / greenery target.

### Medium / Small

Purpose: richer long-horizon behavior, maybe evaluation or stretch training

- 20–40 parcels
- 15–30 road segments
- 3–5 utility zones
- 8–12 hard constraints
- 2–3 curveballs
- 20–50 actions

Adds:

- flood-risk or protected parcels,
- multiple public-service rules,
- daily-needs amenity mix,
- greenery / open-space minimums,
- walkability / access targets,
- phased budget allocation,
- affordable housing or industrial buffer rule.

### Hard / Showcase

Purpose: demo-only or held-out evaluation

- 50–100 parcels
- 40–80 road segments
- 5–10 utility zones
- 12–20+ hard constraints
- 3–6 curveballs
- 50–120 actions

---

## 9. Example task

### Initial state

A district has:

- 12 parcels,
- one arterial road and two side-road branches,
- one water/sewer utility zone,
- one school service node,
- one clinic node,
- one grocery / daily-needs node candidate,
- one park parcel candidate.

### Constraints

- developed parcels must have road access,
- residential parcels must remain within school coverage,
- residential parcels must maintain access to basic amenities,
- sewer capacity must not exceed limit,
- industrial parcels cannot border school parcel,
- park minimum = 1,
- minimum greenery/open-space coverage must be met,
- budget cap = fixed.

### Target

Produce:

- 5 residential parcels,
- 2 commercial parcels,
- 1 park parcel,
- basic amenity coverage for the neighborhood.

### Curveball

“Parcel P8 is reclassified as flood-risk protected land.”

The agent must revise only affected planning decisions without rewriting unrelated parts of the plan.

---

## 10. Why RL is justified here

This environment is designed so that:

- the state evolves after every action,
- mistakes have delayed downstream impact,
- replanning quality matters,
- multiple local decisions interact globally,
- the reward can be computed from explicit rules.

Prompting and SFT may improve formatting and some local behavior, but RL is well-matched to:

- optimizing action policy under delayed reward,
- reducing illegal action sequences,
- learning revision discipline under policy changes,
- improving long-horizon consistency.

This is more defensible than a fuzzy open-ended “design a good city” problem.

---

## 11. Baselines

We plan to compare:

- **Zero-shot small model**
- **Few-shot / structured prompt baseline**
- **SFT model**
- **SFT + RL model**

Optional:

- stronger frontier-model prompted baseline for demo ceiling

Primary trainable small model candidate:

- `Qwen2.5-3B-Instruct`

Primary comparison metrics:

- illegal action rate
- final valid-plan rate
- active constraint violation count
- affected-set F1 after curveball
- unnecessary revision penalty

---

## 12. Why judges may like it

### Environment Innovation

This is not a toy grid world. It is a graph-based, professional-planning environment with legal, infrastructural, and service constraints.

### Storytelling

The story is intuitive:

“LLMs can propose plausible civic plans, but they struggle to maintain legal and infrastructural consistency over long, evolving planning sequences.”

### Showing improvement

The metrics are easy to visualize:

- violation count goes down,
- valid-plan rate goes up,
- replanning precision/recall improves.

### Reward and pipeline clarity

The reward comes from deterministic checks on graph state and planning legality, which is easier to trust than an all-LLM-judge setup.

---

## 13. 24-hour hackathon scope

### What we will actually build

#### Must-have MVP

- one OpenEnv-compliant environment,
- Tiny tier only,
- 3 hand-authored municipal maps,
- deterministic rule-based verifier,
- zero-shot baseline,
- structured-prompt baseline,
- SFT run if possible,
- RL run if time permits,
- Hugging Face Space + README + training notebook.

#### Stretch

- medium-tier map,
- curveball affected-set evaluation,
- stronger model comparison,
- qualitative demo.

### What we will not overcommit to

- full GIS geometry engine,
- free-form architecture judge,
- huge procedural map generator,
- large-scale training across many tiers.

---

## 14. Team split

### Person 1 — Environment + Verifier

- OpenEnv scaffolding
- graph state representation
- transition logic
- legality + constraint checkers
- metrics and logging

### Person 2 — Tasks + Ground Truth

- hand-author 3 Tiny maps
- define constraints and targets
- define curveballs and affected-set labels
- baseline prompt design
- failure case collection

### Person 3 — Models + Training

- baseline model runs
- SFT dataset construction
- SFT fine-tuning
- RL integration if time allows
- plots and comparison table

---

## 15. Final pitch

**CivicFlow** is a long-horizon municipal planning environment where an LLM must act legally and consistently over a graph of parcels, roads, and utilities while policies evolve mid-episode.

The contribution is not just another planning prompt. It is a **verifiable RL environment** for learning:

- action legality,
- structured state tracking,
- constrained replanning,
- long-horizon policy consistency.

It is original enough to stand out, deterministic enough to score cleanly, and practical enough to build as a strong OpenEnv hackathon submission.
