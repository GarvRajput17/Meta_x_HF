# CivicFlow Design Notes

This document explains the **updated benchmark direction** for CivicFlow:

- how the task framing changed,
- how Tiny / Medium / Hard are now meant to differ,
- what real-world planning sources inspired the constraints,
- what dataset strategy we plan to use for SFT,
- and how SFT and RL connect to the same canonical verifier.

This is not the README.  
It is an internal project note so future contributors and agents understand the design choices behind the environment.

---

## 1. Core shift in the environment

The original environment direction was too close to an administrative workflow:

- `parcel`-style wording,
- `approve_permit` / `deny_permit`,
- small local changes,
- mostly static planning targets.

That was not the benchmark we actually wanted.

### Updated framing

CivicFlow now treats the model as a **city planner**, not a permit clerk.

The benchmark is about:

- planning the structure of an urban district,
- balancing housing, economic uses, civic uses, greenery, and services,
- maintaining district-level accessibility and liveability,
- adapting to policy and infrastructure changes over a long horizon,
- revising only the affected subgraph when conditions change.

The environment still uses deterministic graph state, but the **planning story** is now closer to modern neighbourhood / district planning than permit approval.

---

## 2. Current benchmark philosophy

We want three things at once:

1. **Theme 2 fit**  
   The environment should test long-horizon planning and replanning, not just static classification.

2. **Deterministic verification**  
   The environment should remain RLVR-friendly. Most scoring should come from code-based checks, not subjective judges.

3. **Research/storytelling value**  
   The task should feel like a real planning problem, not a toy grid world.

This means the benchmark should not ask:

- “Can the model describe a city?”

It should ask:

- “Can the model maintain a valid district plan across multiple stages, under changing constraints, while preserving unaffected structure?”

---

## 3. Updated task structure

Episodes are now intended to feel **phase-based**.

### Planning phases

1. **Base structure**
   - establish the district layout,
   - allocate land use,
   - preserve protected / flood-risk land,
   - create the initial green/open-space backbone.

2. **Services and access**
   - place school / clinic / grocery / fire / transit / park access,
   - satisfy district-level local-access obligations,
   - avoid over-specialized districts.

3. **Infrastructure balance**
   - keep water / sewer / power feasible,
   - preserve greenery ratio,
   - maintain district mix and viability.

4. **Replanning**
   - respond to policy changes, flood-risk expansion, protection orders, or capacity cuts,
   - revise only the affected districts or subgraph,
   - avoid unnecessary changes elsewhere.

5. **Final validation**
   - reach a valid plan,
   - satisfy global and district-level targets,
   - finish without illegal actions or unresolved violations.

This is a much stronger Theme 2 story than a one-shot static layout task.

---

## 4. Updated difficulty ladder

### Tiny

Tiny is the trainable MVP tier.

Desired properties:

- solvable by the heuristic,
- nontrivial for zero-shot models,
- simple enough to get non-zero reward early,
- usually only one coupled pressure at a time.

Current role:

- prove the environment is feasible,
- generate baseline plots,
- generate SFT expert traces,
- support the first RL runs.

### Medium

Medium should no longer be “just bigger Tiny.”

Desired properties:

- district-level targets matter,
- curveballs affect multiple blocks or one district/infra zone,
- local greedy planning starts to degrade,
- replanning requires preserving unaffected structure.

Current updated direction:

- district-level amenity / greenery / economic-use targets,
- broader affected sets,
- curveballs like flood-buffer expansion and infra-capacity cuts.

### Hard

Hard should behave like a true long-horizon planning stress test.

Desired properties:

- sequential policy shocks,
- interacting district targets,
- coupled tradeoffs between greenery, accessibility, density, and capacity,
- heuristic should degrade meaningfully.

Current updated direction:

- multiple district targets,
- multi-stage curveballs,
- broader affected sets,
- rebalancing across several zones rather than one-block replacements.

---

## 5. Real-world inspiration used for constraints

We do **not** claim CivicFlow implements one specific national planning code.

Instead, we take inspiration from well-known planning frameworks and convert them into **deterministic benchmark constraints**.

### A. UN-Habitat

Used for:

- connected neighbourhood structure,
- mixed land use,
- compact / integrated planning,
- limited land-use specialization,
- public-space and street-network thinking.

Relevant sources:

- UN-Habitat, *Five Principles of Neighbourhood Design*  
  https://unhabitat.org/five-principles-of-neighbourhood-design
- UN-Habitat discussion note PDF  
  https://unhabitat.org/sites/default/files/documents/2019-05/five_principles_of_sustainable_neighborhood_planning.pdf
- UN-Habitat, *MY Neighbourhood*  
  https://unhabitat.org/my-neighbourhood

How this maps into CivicFlow:

- district mix constraints,
- no over-specialized district,
- global and district-level land-use balance,
- “connected / integrated / compact” planning logic.

### B. WHO urban green space guidance

Used for:

- greenery / open-space importance,
- park access and green-space access,
- public-health framing of green urban form.

Relevant sources:

- WHO, *Urban green spaces and health*  
  https://www.who.int/europe/publications/i/item/WHO-EURO-2016-3352-43111-60341
- WHO GreenUr tool  
  https://www.who.int/europe/tools-and-toolkits/greenur--the-green-urban-spaces-and-health-tool
- WHO, *Urban green spaces: a brief for action*  
  https://www.who.int/europe/publications/i/item/9789289052498

How this maps into CivicFlow:

- minimum greenery ratio,
- district green-space targets,
- park/open-space requirements,
- floodplain and ecological-corridor style curveballs.

### C. OECD proximity / age-inclusive city framing

Used for:

- local access to daily needs,
- district-level access to school / clinic / grocery / park / transit,
- age-friendly and child-friendly access logic.

Relevant sources:

- OECD, *Cities for All Ages*  
  https://www.oecd.org/en/publications/cities-for-all-ages_f0c8fefa-en/full-report/exploring-policies-for-age-inclusive-cities_1c4d306e.html
- OECD, *Walkable cities with access to services and amenities for all*  
  https://www.oecd.org/en/publications/oecd-regions-and-cities-at-a-glance-2024_f42db3bf-en/full-report/walkable-cities-with-access-to-services-and-amenities-for-all_68282faa.html

How this maps into CivicFlow:

- accessibility score,
- district amenity requirements,
- “not just global totals, but local access” constraint logic,
- district target overrides during curveballs.

### D. ITDP TOD Standard

Used for:

- walk / connect / transit / mix / compact principles,
- district access to transit,
- mixed-use and mobility-aware planning.

Relevant sources:

- ITDP TOD Standard  
  https://tod.itdp.org/tod-standard.html
- ITDP TOD framework  
  https://tod.itdp.org/tod-standard/tod-standard-framework.html
- TOD scorecard reference  
  https://itdp.org/library/standards-and-guides/tod3-0/the-tod-standard-scorecard/

How this maps into CivicFlow:

- district transit access requirements,
- mixed-use core logic,
- “short-walk access to services” planning framing,
- compact, connected district design.

---

## 6. How these sources are used

### They are NOT the direct SFT dataset

We are **not** treating OECD / WHO / UN-Habitat / ITDP documents as raw supervised training data.

Reasons:

- they are policy and guidance documents,
- they do not match CivicFlow’s action schema,
- they do not contain environment trajectories,
- they are not formatted as step-by-step planner interactions.

### They ARE used in three important ways

#### 1. Benchmark design

They inspire:

- what constraints matter,
- what makes Medium/Hard realistically difficult,
- what kinds of curveballs are plausible,
- which planning dimensions should appear in reward design.

#### 2. SFT trace authoring

They help define what “good planning behavior” should look like in expert traces.

For example:

- district-level service access matters,
- greenery cannot be treated as a cosmetic afterthought,
- mixed-use balance matters,
- transit / walkability matters,
- replanning should preserve unaffected areas.

#### 3. README / storytelling

They help justify why CivicFlow is not arbitrary:

- it reflects real planning principles,
- the constraints are grounded in modern city-planning thinking,
- the benchmark is not just invented reward-shaping.

---

## 7. Planned SFT dataset

For the SFT warm-start, we plan to use a **small custom CivicFlow trajectory dataset**.

### We are NOT planning to use

- a giant external public planning dataset,
- raw guideline documents as direct training samples,
- generic planning essays,
- free-form instruction data unrelated to the CivicFlow action interface.

### We ARE planning to use

- **hand-authored expert trajectories**
- **strong-model-generated traces filtered by the deterministic verifier**
- **a few repaired / revised trajectories showing curveball handling**

### Intended contents

We want roughly:

- `8–12` good trajectories total for the warm start,
- mostly on Tiny tasks,
- at least:
  - standard valid planning trajectory,
  - one service / greenery balancing trajectory,
  - one curveball revision trajectory,
  - one recovery-from-near-miss trajectory.

### What each SFT example should contain

Each example should follow the same interface the model will later use during inference:

- system prompt,
- task briefing,
- current observation,
- structured planning action output,
- next observation if doing multi-turn trajectories.

Examples should teach:

- legal action formatting,
- phase-aware planning,
- district-aware reasoning,
- selective replanning after shocks,
- preservation of unaffected structure.

### Source of the traces

Best practical recipe:

1. Use heuristic planner outputs as the starting point.
2. Manually inspect and keep only good traces.
3. Add a few stronger, more human-like revised traces where needed.
4. Optionally use a stronger LLM to draft traces, but keep only verifier-clean outputs.

This gives us a small, high-quality, environment-specific SFT dataset.

---

## 8. Canonical metrics vs RL reward

This is an important design rule for the project.

### Canonical metrics

These are the official environment outputs and should be stable across:

- heuristic baseline,
- zero-shot baseline,
- SFT model,
- RL model.

Current metric families include:

- legality
- hard constraint violations
- infra overflow
- amenity shortfall
- greenery shortfall
- use target shortfall
- district service / green / mix gaps
- accessibility score
- land-use balance score
- district coverage score
- phase completion score
- progress score
- affected-set precision / recall
- unnecessary change count
- terminal validity

### RL reward

RL should use a reward derived from the canonical metrics, not a totally separate scoring system.

That means:

- same benchmark logic,
- same verifier truth,
- different aggregation and shaping for training.

Current reward components include:

- `legality`
- `constraints`
- `accessibility`
- `land_use_balance`
- `district_quality`
- `phase_progress`
- `progress`
- `replanning`
- `revision_discipline`
- `terminal`

### Design principle

We can tune the **weights or shaping** used for RL,
but we should not invent a different notion of success for RL than the one used for evaluation.

So:

- **canonical metrics stay fixed**
- **RL reward is derived from them**

That keeps the benchmark honest.

---

## 9. Why the reward changed

The earlier reward design was too narrow:

- legality
- constraint satisfaction
- one progress score
- replanning F1
- unnecessary revision
- terminal validity

That was okay for Tiny, but it did not reflect modern planning concerns strongly enough.

The updated reward adds:

- district-level access,
- district-level green / mix quality,
- phase completion,
- land-use balance,
- better long-horizon planning signals.

This is closer to what a real planner would actually care about while still staying deterministic.

---

## 10. Why Medium and Hard changed

Earlier Medium/Hard were too close to:

- “one block became invalid, move it somewhere else.”

That was not strong enough for Theme 2.

The updated versions now try to create:

- broader affected sets,
- district-level target changes,
- infrastructure capacity cuts,
- greenery target changes,
- service-access target changes,
- stronger need to preserve unaffected structure.

This is much closer to:

- long-horizon planning,
- dependency-aware revision,
- meaningful heuristic failure,
- realistic benchmark progression.

---

## 11. Current benchmark expectation

The intended behavior is:

- **Tiny**  
  heuristic succeeds, zero-shot struggles, SFT improves.

- **Medium**  
  heuristic degrades because local greedy planning becomes brittle.

- **Hard**  
  heuristic should clearly struggle; a trained model may still not fully solve it, but should perform better on validity, replanning, and structured consistency.

This is the benchmark story we want.

---

## 12. Open design decisions still remaining

These are not fully closed yet:

- whether to rename action verbs to even more planner-like language,
- whether to add explicit density / mobility / road-connection actions,
- whether district targets should become more formalized in all task files,
- how far Medium/Hard should go before they become too hard for useful RL signal,
- how much of the accessibility logic should stay topological vs distance-weighted.

These are benchmark-evolution questions, not blockers for the Tiny path.

---

## 13. Practical conclusion

The current intended workflow is:

1. Keep Tiny stable and solvable.
2. Use heuristic + filtered expert traces for SFT warm start.
3. Use the deterministic canonical verifier for all baselines and RL.
4. Keep OECD / WHO / UN-Habitat / ITDP as **design inspiration and justification**, not raw direct training data.
5. Make Medium/Hard increasingly district-level and replanning-heavy so they genuinely reflect Theme 2.

That is the updated CivicFlow direction.
