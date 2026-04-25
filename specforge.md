SpecForge — An RL Environment for Long-Horizon Engineering Judgment

**OpenEnv Hackathon (India 2026) — Project Plan v2**
**Primary theme: #2 (Long-Horizon Planning & Instruction Following)**
**Secondary alignment: #3.1 (Professional Tasks). Stretch: #4 (Self-Improvement).**

---

## Changelog from v1

Three execution-level fixes applied. Strategy and thesis unchanged.

1. **Training target recalibrated.** Tiny is the primary training tier; Small is a stretch goal. Medium and Large are **evaluation + demo only** — we do not promise training curves on them. This makes the reward curves we show real, given onsite compute reality.
2. **Reward system pruned from 9 weighted terms to 5 live training terms + 2 hard floors + 2 terminal-only signals.** Removes a weekend of weight-tuning hell. Anti-gaming structure is preserved (and clearer).
3. **Content load cut to a realistic floor.** 8 PRDs minimum (3 Tiny / 3 Small / 2 Medium), 12 curveballs, 30 hand-labeled affected-sets. SFT warmup trace construction moved from "Section 9 footnote" to P3's explicit deliverable.

Plus four micro-fixes: tightened justified-supersession check, confirmed no reserved-tool-name collisions, Wandb-run-URL placement in README, no MP4s committed to HF Space.

---

## 0. TL;DR

Frontier LLMs fail on multi-day engineering tasks not because they can't *store* information, but because their **policy of tool use, commitment, and renegotiation** breaks down under long horizons.

**SpecForge** is an OpenEnv-compliant environment where a model plays a staff engineer running an extended system-design engagement. A structured **Constraint Ledger — exposed as a tool the agent must call** — provides perfect external memory. The hard part isn't remembering; it's knowing **when to query, when to commit, when to renegotiate, and how to defend decisions under pressure.** That's what we train.

We train five policy capabilities, validate against machine-checkable engineering artifacts (DDL, OpenAPI, capacity math, Mermaid diagrams), and produce reward curves on the Tiny tier with quantitative + qualitative evidence on harder tiers. The demo is a split-screen: untrained vs. trained agent, same PRD, same curveballs, the failures lighting up red on the left.

---

## 1. Thesis

**The headline (for the README, blog, and pitch):**

> Long-horizon agents fail not because they can't store information but because their policy of tool use, commitment, and renegotiation breaks down under pressure. SpecForge provides an RL environment where an external Constraint Ledger makes perfect memory available, and the agent is trained on the policies memory alone can't give you: when to query, when to commit, when to revise, and how to defend those choices.

**The boundary statement (sits openly in the README — it's the limitations move that separates a submission that survives scrutiny from one that doesn't):**

> We are not training the model to remember. Memory is solved by the Constraint Ledger tool. We are training the policy of *when* to query that tool, when to commit decisively, when to renegotiate vs. hold firm, and how to defend decisions under scrutiny. These are long-horizon judgment behaviors that persist across context resets and that no amount of prompting reliably produces.

This explicit framing absorbs the strongest critique a paper-reading judge can make ("forgetting is a RAG problem") and converts it into our differentiator.

---

## 2. Problem statement

Anyone who has tried to use Claude / GPT-4 / Gemini to design a real system across multiple sessions has hit the same wall. The model:

- Forgets foundational decisions made earlier and silently violates them
- Hedges constantly to avoid committing to anything specific
- Over-revises (panic-rewrites everything) or under-revises (misses cascading impacts) when reality shifts
- Caves to mild pushback ("you're right, let me reconsider…") even when its position was correct
- Treats unjustified drift and justified renegotiation as the same kind of plan change

These are the failure modes that prevent LLMs from being trusted as autonomous engineering partners on tasks that span weeks rather than minutes. They are also failure modes that **no amount of prompting reliably fixes**.

The same failure pattern shows up well beyond software — in legal drafting, medical treatment plans, policy authoring, and long research plans — anywhere a binding charter must be maintained over a long horizon. We lead with software because it's the most demo-able instance, but the framework is domain-general.

---

## 3. Theme alignment

| Theme | Fit |
|---|---|
| **#2 — (Super) Long-Horizon Planning** | **Primary.** Multi-phase trajectories, 10–35 constraints (Tiny/Small training tiers), sparse + dense rewards, recovery from early commitments, durable representations across context boundaries. |
| **#3.1 — Professional Tasks** | Secondary. The agent produces real engineering artifacts (DDL, OpenAPI, capacity math) that we parse and validate programmatically. |
| **#4 — Self-Improvement** | Stretch / future direction. Once trained, the agent could generate adversarial PRDs targeting its own weaknesses. README-only mention. |

We position primarily under Theme 2 because the failure mode being trained is the long horizon. Professional artifacts are the mechanism for clean reward signal.

---

## 4. The five policy claims (what we actually train)

These are the trainable policies — each is something tool-augmentation alone cannot fix.

### 4.1 Query discipline
**When** to consult the ledger. Prompted models either over-query (paranoid, wastes turns) or under-query (confident, drifts). RL finds the sweet spot because query-cost vs. violation-cost creates a trainable tradeoff. **This is our specific counter to the "just give it RAG" objection — the tool exists, and using it well is itself a learned skill.**

### 4.2 Decisiveness under pressure
Prompted models hedge ("we could consider…", "one option would be…") to dodge punishment for specific commitments. The environment's specificity reward + decisiveness floor make hedging a losing strategy. RL learns to commit when evidence warrants.

### 4.3 Curveball triage
When reality shifts (compliance rule arrives mid-engagement), precision/recall on the set of *affected* prior decisions. Pure judgment — pattern-matching new facts against live state. Untrained models massively over- or under-revise. Cleanly trainable with a precision/recall reward.

### 4.4 Renegotiation discipline
When to mark a constraint superseded (with justification), when to hold firm, when to find a creative workaround that preserves both. Dense signal: unjustified supersession → penalty; justified supersession → neutral; creative preservation → reward.

### 4.5 Argumentative robustness
Surviving the Adversarial Reviewer. Distinguishing genuine reviewer points (update your position) from social pressure (defend it). Prompted models collapse under skeptical follow-ups. Training agents on this is exactly the sycophancy-resistance work frontier labs are starting to publish about.

All five are **policy-learning** claims. All five are things tool-augmentation alone cannot fix. That is the thesis.

---

## 5. Environment design

### 5.1 Core loop (phased)

The agent inhabits a phased engineering engagement. Each phase demands specific artifacts; the env validates them.

Phase turn ranges given for the **Tiny tier** (our actual training target). Other tiers scale proportionally.

| Phase | Tiny turns | What happens |
|---|---|---|
| **0 — Intake** | 1–8 | Receive PRD + dossier (company context, team skills, budget, compliance regime). Agent must **interrogate** — ask the Stakeholder Bot clarifying questions, flag PRD contradictions, identify missing info. **Reward for questions that uncover genuine ambiguities; penalty for redundant ones.** |
| **1 — RFC** | 8–20 | Produce a structured architecture RFC: components, data flow, tech stack with justifications, risk register. Env parses RFC into machine-readable form. Every commitment ("PostgreSQL 16") enters the Ledger with provenance. |
| **2 — Depth** | 20–35 | Flesh out hard parts: DDL schema, API contracts, capacity math (QPS / storage / bandwidth — we do the arithmetic check), failure modes, rollout plan. |
| **3 — Curveball** | ~38 | Env fires 1 scripted event (compliance change, budget cut, deprecated API). Designed so naive responses violate prior commitments. Agent must identify affected decisions, produce a diff, justify revisions. |
| **4 — Adversarial review** | 40–50 | A Skeptical Reviewer Bot grills the agent: *"Your schema has no index on `email` but you said p95 lookup < 10ms. Defend this."* Bad plans crumble; good plans survive. Terminal reward decided here. |

Small tier: ~100 turns, 2 curveballs. Medium tier: ~180 turns, 4 curveballs (eval-only). Large tier: ~280 turns, 6 curveballs (showcase episode for demo only).

### 5.2 The Constraint Ledger (the heart of the env)

The ledger is an **explicit tool the agent calls** — not a magical background system. This is the structural change that absorbs the critique and makes our reward signal honest.

```python
ledger.query(topic: str, scope: Scope = None) -> list[ConstraintEntry]
ledger.commit(decision: str, rationale: str, scope: Scope) -> entry_id
ledger.supersede(old_id: str, new_decision: str, justification: str,
                 cite_event_id: str) -> entry_id
ledger.list_active(scope: Scope = None) -> list[ConstraintEntry]
```

Tool names verified against OpenEnv reserved set (`reset` / `step` / `state` / `close`) — no collisions.

Every entry carries:
- **Source** — `PRD §2.3` / `Stakeholder turn 34` / `Agent commitment turn 78` / `Curveball turn 145`
- **Type** — hard invariant / soft preference / derived consequence
- **Scope** — global / component-specific / phase-specific
- **Status** — active / superseded / renegotiated-with-justification

Every agent output is parsed for decisions. Each runs against the active ledger. A violation fires:
1. A penalty to the reward
2. A trace: `turn 147 violates constraint introduced at turn 0 by PRD §2.3`
3. A log entry that becomes the demo's killer visualization (timeline with arrows from late-turn violations back to their origins)

**Renegotiation is first-class — and its check is tightened in v2.** `supersede()` is only counted as justified if:
- `cite_event_id` references a real curveball or stakeholder event in the live env state, AND
- the supersession happens within 10 turns of that event firing, AND
- the justification text actually mentions the cited event's content (string check + light semantic match)

Otherwise: marked as unjustified drift, penalty applies. This closes the "fluent justification for any drift" exploit. We are not training rigidity; we are training disciplined flexibility.

### 5.3 Multi-stakeholder simulation

Different personas fire questions and pressure at different phases. Each has its own agenda and vocabulary:

- **CEO** — timeline, cost, competitive features, rarely technical
- **PM** — UX, feature prioritization, edge cases
- **Security/Compliance** — data handling, auth flows, audit trails
- **Ops/SRE** — observability, failure modes, on-call burden
- **Senior engineer** — uncomfortable technical questions
- **Junior engineer** — naive questions that test whether the plan is actually understandable

The agent must project a single coherent world model through different lenses. This tests a second capability beyond memory: maintaining coherence across audiences. **For Tiny tier: 3 personas (CEO, PM, Senior engineer). Add the rest as content allows.**

### 5.4 Artifacts (the secret weapon for clean reward)

The agent doesn't just talk — it produces structured, parseable artifacts.

| Artifact | Format | What we validate |
|---|---|---|
| RFC document | Labeled markdown | Section completeness; ledger-linked decisions |
| Architecture diagram | Mermaid | Parseable; components match RFC |
| Database schema | SQL DDL | `sqlparse` → check indexes against latency claims |
| API contract | Simplified OpenAPI | Endpoint coverage matches RFC |
| Capacity worksheet | Structured numbers | Arithmetic actually evaluated |
| Risk register | Risk + mitigation pairs | Coverage of identified failure modes |
| Decision log | Agent's own commitment list | Must reconcile with the Ledger |

**Real artifacts unlock real validation.** When the agent writes "p95 < 50ms" in the RFC and emits a schema without an index on the lookup column, that's a programmatic contradiction we catch. **The reward signal isn't vibes — it's truth conditions.** This is the magic that makes judges trust the curves.

**Day-7 fallback subset** if parsers slip: keep RFC + DDL + capacity worksheet (the three highest-signal validators). Drop OpenAPI / Mermaid as parseable bonuses.

### 5.5 Curveballs

A curveball library with **12 events** across categories:
- **Compliance shifts** (3) — SOC2, GDPR data residency, HIPAA scope
- **Team changes** (2) — lead leaves, headcount cut
- **Budget changes** (2) — 30% cut, infra cost overrun
- **External dependencies** (3) — deprecated API, vendor outage, library CVE
- **Scope changes** (2) — executive demands feature X by Y

Each curveball, paired with each PRD it can fire on, has a **hand-labeled correct affected-set**: the constraints in the active ledger that *should* be revisited. Used to score recall and precision of the agent's response. Floor target: **30 (curveball, PRD) → affected-set labels**, prioritizing Tiny + Small tiers.

---

## 6. Reward system (rebalanced in v2)

### 6.1 Five live training rewards (what tunes during GRPO)

| # | Reward | Frequency | Captures | Starting weight |
|---|---|---|---|---|
| **R1** | Constraint violation | Per turn | Penalty for any committed decision that contradicts active ledger | **−1.0** per violation |
| **R2** | Query discipline | Per decision | Penalty if a decision is made in a domain where relevant ledger entries existed and the agent did not query first | **0.5** |
| **R3** | Curveball recovery F1 | Per curveball | Precision & recall on affected-constraint set (over-revision and under-revision both punished) | **1.2** ← heaviest |
| **R4** | Adversarial review | Episode terminal | Score from Reviewer Bot's grilling — challenges survived = +, exposed incoherence = − | **1.0** |
| **R5** | Decision specificity | Per turn a decision is made | Vague non-commitments don't enter the ledger; specific ones earn a small + | **0.4** |

**R3 is the curve we sell.** It captures the capability that matters most and is the hardest to game.

### 6.2 Two hard floors (episode-level constraints, not tunable rewards)

These are **not** weighted reward terms — they're floors. If violated, the episode reward is clipped to a fixed penalty regardless of other signals. This stops the agent from gaming "be silent and don't violate anything" or "just say 'we'll figure it out later' for everything."

- **Decisiveness floor.** Agent must make ≥N concrete commitments by end of Phase 2 (Depth). Below floor → episode penalty.
- **Coverage floor.** Final plan must address ≥X% of PRD requirements. Below floor → episode penalty.

(N and X tuned per tier. Tiny: N=6, X=80%.)

### 6.3 Two terminal-only signals (eval, not training)

- **Phase completion check.** Required artifacts exist and are internally consistent at each phase boundary. Reported in eval, folded into Adversarial Review at terminal — not separately weighted during training.
- **Rubric judge.** Holistic LLM-as-judge on the full artifact pack vs. PRD. **Run on eval episodes only**, not every training step. Too expensive and noisy to weight during the GRPO loop.

### 6.4 Anti-gaming measures (preserved from v1)

- **Decisiveness floor** (above) — refusing to decide doesn't dodge violations
- **Specificity check** (R5) — *"We'll use a suitable database"* doesn't count
- **Coverage floor** (above) — silent omissions detected and penalized
- **Query-without-decision exploit** — spammed queries that don't precede decisions in that domain → penalty inside R2
- **Pre-emptive supersession exploit** — closed by tightened justification check (§5.2): supersession must cite a real event within 10 turns
- **Artifact-as-cover exploit** — parseable but trivial artifact ("schema: TODO") fails coverage and specificity

### 6.5 The reward curves the demo shows

1. **Constraint violation rate** per episode — falls
2. **Query precision** — fraction of agent queries that were timely and relevant — climbs. *This curve is our answer to the RAG critique.*
3. **Curveball recovery F1** — precision/recall on affected-constraint sets — climbs
4. **Adversarial review score** — climbs
5. **Decision specificity score** — fraction of agent statements specific enough to enter the ledger — climbs

Five curves on the **Tiny tier**, with the strongest two extended onto **Small** if convergence permits. We do not promise curves on Medium / Large.

---

## 7. Curriculum

| Tier | Constraints | Turns | Curveballs | Domain | Role |
|---|---|---|---|---|---|
| **Tiny** | 10 | ~50 | 1 | URL shortener (single service) | **Primary training target** |
| **Small** | 20 | ~100 | 2 | Basic social feed (multi-service) | **Stretch training; eval if no time** |
| **Medium** | 35 | ~180 | 4 | Ride-sharing backend | **Eval + demo only** |
| **Large** | 60 | ~280 | 6 | Healthcare records platform | **Single showcase episode for demo** |

We start training on Tiny for clean signal fast. Honest re-statement: **the reward curves we publish are on Tiny.** If Tiny converges with onsite compute headroom, we extend to Small. Medium / Large are inference-only — we run the trained model on them once each to capture the qualitative behavior for the demo, and we are explicit in the README that they are not training tiers.

This is dramatically more honest than promising curves across all tiers, and judges respect honest scoping.

---

## 8. Baselines & evaluation (this is where we directly answer the critique)

| Baseline | Why include |
|---|---|
| **Random / blank-slate** | Floor. Sanity check. |
| **Untrained base model — no ledger** | Original story. "Look, it forgets things." |
| **Untrained base model — ledger as tool, no prompt help** | Shows tool-augmentation alone isn't enough — model doesn't use the tool well. |
| **Untrained base model — ledger + strong prompt: "always query before deciding"** | The critic's proposed solution. **If we beat this, we have a finding.** |
| **Trained model — ledger** | Our system. |
| **Frontier model (Claude Sonnet) — ledger + best prompt** | The expensive ceiling. If our small trained model approaches it, that's a separate cost-efficiency story. |

**Either way we have a story:**
- If trained-small > prompted-frontier on violation rate → "training learns disciplined tool use that prompting can't reproduce"
- If trained-small ≈ prompted-frontier → "small trained model matches frontier at a fraction of inference cost"

We do not lose this comparison. That's good experimental design.

---

## 9. Training pipeline

**Stack:**
- **OpenEnv (latest release)** — `Environment` / `MCPEnvironment` base classes, `openenv.yaml` manifest, Gym-style `reset` / `step` / `state` API
- **Unsloth** for SFT warmup; **HF TRL (GRPO)** for the RL loop
- **Colab notebook** for the training script (judges must be able to re-run it)
- **Hugging Face Space** for env hosting (discoverability + judge-runnable)
- **HF Hub + Wandb** for the trained model and run logs (Wandb URL for the **specific final training run** linked in README, not just "we used Wandb")

**Steps:**
1. **SFT warmup** on a small set of expert trajectories on Tiny tier (~25 episodes — owned by P3, see §11). Optional but accelerates convergence within hackathon timeline.
2. **RL training (GRPO)** on Tiny curriculum until violation rate plateaus.
3. **Curriculum step-up:** Tiny → Small **only if Tiny converges with time to spare**. Don't sacrifice Tiny curve quality for Small attempts.
4. **Eval** on held-out PRDs in each trained tier + qualitative runs on Medium / Large for demo.
5. **Export** plots, before/after artifact comparisons, and the key demo episode trajectory.

**Compute target:** 3B base model for the primary trained agent (e.g., Qwen-2.5-3B). 7–8B as stretch only if Tiny converges in <1 day onsite and credits remain.

---

## 10. Demo script (~90 seconds)

**Open** with one of the team on camera (15 seconds):

> "I've been using Claude for 4 months to help design this system. I lost count of the times it forgot we chose Postgres at the start and suggested MongoDB at turn 200. We decided to train that problem away."

**Cut** to split-screen. Same Medium-tier PRD (inference-only — we are honest in the README that training was on Tiny + Small). Same curveballs. Both agents running.

**Curveball: "Compliance requires EU data residency."**

- **Left (untrained):** *"We'll move the primary database to AWS eu-west-1."* 🔴 Env flashes: violates §4.1 (turn 0): no AWS, GCP-only org policy.
- **Right (trained):** *"EU residency triggers changes to (1) GCP region — move to europe-west1, (2) ClickHouse deployment which is currently US-only, (3) logging pipeline shipping to Datadog US. Revised plan attached."* ✅

**Cut** to the five reward curves climbing (Tiny tier).
**Cut** to the constraint-violation timeline shrinking episode by episode.
**Cut** to the **query-precision curve** — *the chart that says we're not just RAG.*

**End:**
> "We didn't prompt-engineer this behavior. We trained it in. And we trained the part prompting can't reach."

---

## 11. Team split (3 people)

| Person | Owns |
|---|---|
| **P1 — Systems** | OpenEnv scaffolding; Constraint Ledger engine (incl. tightened supersede check); artifact parsers (RFC + DDL + capacity first; Mermaid + OpenAPI as time allows); reward aggregator; HF Space deployment |
| **P2 — Content** | **8 PRDs floor** (3 Tiny / 3 Small / 2 Medium); 12 curveballs; 30 (curveball, PRD) affected-set labels; 3 stakeholder persona scripts (CEO / PM / Senior eng) for Tiny + Small, additional personas for Medium; reviewer-bot challenge bank; rubric-judge prompt; blog/video writeup |
| **P3 — Training** | **SFT warmup data construction** (~25 expert traces on Tiny — moved here from Section 9 footnote); Unsloth / TRL pipeline; training runs on Tiny (and Small if time); reward curves; baseline-vs-trained evals; README results section; demo recording |

**Days 1–3:** all three on OpenEnv basics together (everyone needs to know the manifest and the loop).
**Day 4 onwards:** split. **30-min daily sync.**

**P2 stretch (only if floor cleared by D8):** +4 more PRDs, +3 more personas, expand affected-set labels to 50.

---

## 12. Timeline / milestones

Working backward from submission. Adjust as needed.

| Phase | Days | Deliverables |
|---|---|---|
| **Foundation** | D1–D3 | OpenEnv skeleton, Ledger v0 (in-memory dict), one Tiny PRD end-to-end runnable |
| **Core env** | D4–D7 | Artifact parsers (RFC + DDL + capacity), full Ledger with provenance + tightened supersede, 3 stakeholder personas, 5 curveballs wired, R1+R2+R5 reward live |
| **Content build (parallel)** | D5–D9 | 8 PRDs floor (3 Tiny / 3 Small / 2 Medium), 12 curveballs, 30 affected-set labels, reviewer prompts |
| **Training prep** | D8–D10 | Unsloth/TRL pipeline runs end-to-end against env; 25 SFT warmup traces prepared; baseline runs logged |
| **Training (onsite)** | D11–D12 | RL on Tiny first; Small only if time. Five reward curves logged; eval episodes captured |
| **Demo & write-up** | D11–D13 | Split-screen demo recorded; README polished (with Wandb run URL, **not** an MP4); blog/video published; HF Space live |
| **Buffer** | D13 | Submit. **No commits after the deadline** — the brief explicitly says post-deadline changes won't be considered. |

**Note on HF Space:** link videos by URL (YouTube / HF Posts). Do not commit MP4 files into the Space repo — the brief warns about env size.

---

## 13. Mapping to judging criteria

| Criterion | Weight | How we hit it |
|---|---|---|
| **Environment Innovation** | 40% | "Engineering judgment under long horizons" is unexplored. Constraint Ledger as explicit tool + machine-checkable artifacts + adversarial review is a new combination. Five trainable policies, none of which are addressed by current envs. The framework is domain-general (legal/medical/policy) — that breadth strengthens novelty. |
| **Storytelling** | 30% | Real-engineer testimonial → split-screen demo → reward curves climbing → query-precision chart as answer to skeptics. The "AI forgets we chose Postgres" pain is universally relatable to technical *and* non-technical audiences. |
| **Showing Improvement** | 20% | Five reward curves on Tiny; baseline-vs-trained side-by-side; before/after artifact comparison; constraint-violation timeline narrowing visually across episodes. Honest scoping (curves on trained tiers only) signals research maturity. |
| **Reward & Pipeline** | 10% | Five live reward terms with anti-gaming structure (floors + tightened supersede check + 5 named exploits closed); Unsloth/TRL Colab notebook re-runnable by judges; HF Space hosted; Wandb run linked. |

**Hidden upside:** the explicit "what we don't train" section in the README, plus the explicit "trained on Tiny only" honesty, signals research maturity to a paper-reading judge. That signal lifts the storytelling and innovation scores together.

---

## 14. Open decisions to lock before sprint

1. **Domain commitment.** Software only for the hackathon, README explicitly claims framework generalizes (legal/medical/policy). → **Locked: software-only.**
2. **Artifact ambition.** RFC + DDL + capacity (must-have) + Mermaid + OpenAPI (stretch). → **Locked: must-have first, stretch as time allows.**
3. **Name.** SpecForge / Keystone / Charter / Bedrock / Invariant / Blueprint / Cornerstone? → **No recommendation — pick what feels right.** Default: SpecForge.
4. **Model size.** → **Locked: 3B base for primary run** (Qwen-2.5-3B). 7–8B as stretch.
5. **SFT warmup or pure RL?** → **Locked: SFT warmup on ~25 expert traces** (P3-owned).
6. **Frontier-model baseline.** → **Locked: Claude Sonnet** for the prompted-with-tool baseline.

---

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Artifact parsers buggy → reward signal noisy | Aggressive unit tests on parsers in D4–D7; locked must-have subset (RFC + DDL + capacity) means stretch parsers can drop without breaking reward signal |
| Tiny doesn't converge in onsite time | SFT warmup carries the lift; if pure RL stalls, present SFT+RL combined curve and be honest about the split |
| "Just use RAG" critique still lands | Query-precision chart + prompted-frontier baseline + the explicit limitations section |
| Demo lacks emotional hook | Real-engineer testimonial recorded D1; backup: voiceover the same idea over screen capture |
| Over-scoping content (P2) | Floor is 8 PRDs / 12 curveballs / 30 affected-sets. Stretch to 12/16/50 only if floor cleared by D8 |
| Reviewer bot too easy or too hard | Tune challenge bank during training; if reward saturates, add adversarial challenges; if it bottoms out, soften early challenges |
| HF Space deploy fails on submission day | Get Space live by D7 with placeholder env; iterate in place rather than waiting until end |
| Reward gaming we didn't anticipate | Save 1 day of training budget for a "patch round" after watching real trajectories |
| Small tier tempting but Tiny still noisy | **Hard rule: do not start Small until Tiny curve is publishable.** Better one clean curve than two noisy ones. |

---

## 16. Future directions (mention in README, do not commit)

- **Theme 4 stretch.** Once trained, let the agent generate adversarial PRDs targeting its own weaknesses. Self-improvement loop on top of the same environment.
- **Domain transfer.** Same constraint-ledger + artifact-validation machinery → legal contract drafting, medical treatment planning, policy drafting. The framework is domain-general.
- **Composable library.** Expose `ConstraintLedger` as a standalone library other OpenEnv contributors can reuse — the constraint-tracking pattern is broadly useful.
- **Benchmark.** SpecForge PRD set could become a long-horizon planning benchmark independent of training — useful to frontier labs evaluating staff-engineer capability.

---

## Appendix A — Anatomy of a Tiny PRD (illustrative)

```
PRD-001: URL Shortener

Hard constraints (entered in ledger at t=0):
  C1.  Stack must use Python (team has zero Go/Rust experience)
  C2.  Cloud provider: GCP only (org policy)
  C3.  No MongoDB (existing org policy, see infra/policies.md)
  C4.  Monthly infra budget: $500
  C5.  GDPR compliant; user data residency = EU only
  C6.  p95 redirect latency < 50ms
  C7.  Custom-slug feature must be available at launch
  C8.  Launch in 4 weeks
  C9.  Must integrate with existing auth at /auth/v1
  C10. Team size: 2 engineers, 0.5 designer

Soft preferences:
  S1. Prefer managed services over self-hosted
  S2. Existing observability stack: Datadog (US) — note conflict with C5

Stakeholder dossier:
  CEO:  cares about launch date and cost
  PM:   custom slugs and analytics
  Senior eng: GDPR, audit logs, schema reviews

Curveball (fires turn ~38):
  "Marketing wants link-click analytics with 30-day history."
  Correct affected set: capacity math (storage), DB schema (events table),
  potentially budget (storage cost).
```

## Appendix B — Constraint Ledger entry shape

```python
@dataclass
class ConstraintEntry:
    id: str
    text: str
    source: Source            # PRD §x.y / Stakeholder turn N / Agent turn N / Curveball turn N
    type: Literal["hard", "soft", "derived"]
    scope: Scope              # global / component / phase
    status: Literal["active", "superseded", "renegotiated"]
    introduced_turn: int
    superseded_by: str | None
    superseded_cite: str | None    # event_id justifying supersession
    superseded_turn: int | None    # for the within-10-turns check
    justification: str | None      # required when status != "active"
```

## Appendix C — Reward formula (v2: 5 live terms + floors)

```
# === Live training reward ===
total_reward = (
      w_violation     * (-1) * num_violations           # R1: per-turn
    + w_query_disc    * query_discipline_score          # R2: query precision
    + w_curveball     * curveball_f1_score              # R3: per curveball, heaviest
    + w_adversarial   * adversarial_review_score        # R4: terminal
    + w_specificity   * decision_specificity_score      # R5: anti-vagueness
)

# Floors (clip episode reward if violated):
if commitments_made_by_phase_2 < N:
    total_reward = floor_penalty
if prd_coverage_fraction < X:
    total_reward = floor_penalty

# Starting weights (rebalance after first run):
w_violation     = 1.0
w_query_disc    = 0.5
w_curveball     = 1.2     # heaviest single signal — capability we sell
w_adversarial   = 1.0
w_specificity   = 0.4

# Tiny tier floors:
N = 6 commitments, X = 0.80

# === Eval-only signals (not in training loss) ===
phase_completion_score      # checked at boundaries, reported
rubric_judge_score          # LLM judge on full artifact pack
```

---

## Final framing for the README opener

> Frontier LLMs have memory tools. They still ship plans where clause 47 contradicts clause 3.
>
> The bottleneck isn't recall — it's **engineering judgment under long horizons**: when to consult what you've committed to, when to commit decisively, when to revise vs. hold, and how to defend a decision under pressure.
>
> SpecForge is an OpenEnv environment that trains exactly that. We do not train memory. We train the policies that perfect memory still can't give you.
