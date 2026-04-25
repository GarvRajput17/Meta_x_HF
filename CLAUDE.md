# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This repo is **reference material for a Meta OpenEnv hackathon submission** (India, April 2026). It currently contains only markdown documentation — no source code, build system, or tests. Prior commits (`source codes added`, `All2All implementation`, `First working code`) introduced code that was subsequently removed in `21a1c01 Delete Downloads directory`; if source is reintroduced, update this file.

The four markdown files are the canonical context for any work done here:

- [Themes_and_judging_criteria.md](Themes_and_judging_criteria.md) — the five hackathon themes and the weighted rubric (Environment Innovation 40%, Storytelling 30%, Reward Improvement 20%, Pipeline 10%). Also lists **non-negotiable submission requirements**.
- [pariticipant_help_guide.md](pariticipant_help_guide.md) — end-to-end build guide: picking the task, the minimum RL loop, reward design, anti-reward-hacking, stack choice, team split, 1-day plan.
- [HackathonFAQs.md](HackathonFAQs.md) — deep Q&A on RL-for-LLMs concepts (RLVR, GRPO, reward hacking, process supervision).
- [hackathon_resources.md](hackathon_resources.md) — canonical links to OpenEnv GitHub/docs, HF hub, tutorials, and reward-engineering papers.

## Hard constraints from the rules

When producing a submission artifact from this repo, these are stated as **non-negotiable** in `Themes_and_judging_criteria.md`:

- Build on **OpenEnv (latest release)** — use its `Environment` / `MCPEnvironment` base classes, standard Gym-style API (`reset`, `step`, `state`), and a valid `openenv.yaml` manifest. Do not reinvent the interface.
- Respect **client / server separation** — clients must never import server internals.
- Do **not** use reserved tool names (`reset`, `step`, `state`, `close`) for MCP tools.
- Training script must use **Unsloth or HuggingFace TRL**, ideally a re-runnable Colab.
- Environment must be deployed to a **Hugging Face Space**; the Space URL goes in the README (judges pull from URL; post-deadline commits are ignored).
- README must embed reward/loss plots (committed as `.png`/`.jpg`, not left only in Colab/deleted W&B runs) and link any video/blog assets. Do not commit large video files to the HF Space.

## Architectural guidance (from the participant guide)

If/when building the environment in this repo, the guide prescribes a specific shape — follow it rather than improvising:

- **Environment owns world dynamics and scoring; trainer owns optimization; model only learns to act through the interface.** Keep these separated.
- Scaffold with the OpenEnv CLI. Environment is a Python package exposed via a **FastAPI app** with action/observation dataclasses, a state representation, and `reset`/`step` methods.
- **Design reward before trainer.** Use *multiple independent* reward functions (execution success, correctness, format compliance, timeouts, resource use, safety, anti-cheat) — a single scalar is easy to hack.
- **RLVR over learned reward models** when the task is verifiable. Prefer GRPO-style training (TRL).
- **Inference, not the optimizer step, usually dominates runtime** in LLM RL — this is the stated reason Unsloth is in the stack. Factor this into perf decisions.
- **Curriculum first, scale last**: confirm `reset`/`step`/rewards/timeouts work locally and remotely before increasing batch size or task diversity. Deploy an early version of the env to a Space before serious training to surface packaging issues.
- **QLoRA save gotcha**: do not upcast a 4-bit model to 16-bit and then merge LoRA weights naively — use the proper merged-save path or keep adapters separate.

## Working in this repo today

There is nothing to build, lint, or test. Edits are to the markdown files. Use standard `git` for history. If you add source code, update this file with the actual build/test/run commands rather than leaving this section in place.
