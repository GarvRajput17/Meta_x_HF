# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "trl>=0.12.0",
#   # peft 0.18.0 imports EmbeddingParallel from transformers>=5.4 which
#   # unsloth doesn't pin yet — stay on 0.17.x.
#   "peft>=0.13,<0.18",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
#   "trackio>=0.0.7",
#   "openenv-core>=0.2.2",
#   "openenv-civicflow-env @ git+https://huggingface.co/spaces/Aaryan369/civicflow-env",
#   "vllm>=0.19,<0.20",
# ]
# ///
"""GRPO training for CivicFlow — built on the SFT adapter.

CRITICAL FIXES baked in (mirrors @hades-022/scamshield):
  (A) PatchFastRL("GRPO", FastLanguageModel) BEFORE any model load.
  (B) Monkey-patch unsloth.kernels.utils.matmul_lora to be dtype-safe
      (Unsloth PR #4918, still open). Without this the GRPO inner
      re-forward crashes with:
        RuntimeError: self and mat2 must have the same dtype,
                      but got Half and Float
      because fast_dequantize returns fp16 while PEFT upcasts X to fp32
      outside the trainer's bf16 autocast context.
  (C) Hot-patch vllm.lora.worker_manager.create_lora_manager signature
      (Unsloth issue #3962) — vLLM 0.15+ added a positional arg the
      base class doesn't accept.
  (D) Force vllm.LLM(enforce_eager=True) at construction time — vLLM
      0.19.1's V1 inductor `_decompose_size_nodes` crashes on bnb-4bit
      Qwen2.5. Eager mode keeps paged attention + continuous batching.
  (E) Don't call get_peft_model() after loading the SFT adapter — GRPO
      continues training the SAME LoRA that SFT produced.
"""
from __future__ import annotations

import json
import os
import sys
import time

# ─── H200 CUDA warm-up: driver may not be ready when the container starts ───
def _wait_for_cuda(max_wait_s: int = 60) -> None:
    """Retry torch.cuda.is_available() until the driver answers.

    On H200 hf-jobs containers we've seen `cudaGetDeviceCount` return error
    802 "system not yet initialized" on first call. A short retry loop fixes
    it without affecting other flavors (where it succeeds immediately).
    """
    import torch
    deadline = time.time() + max_wait_s
    last_err = None
    while time.time() < deadline:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                _ = torch.zeros(1).cuda()
                print(f"      ✓ CUDA ready: {torch.cuda.get_device_name(0)}")
                return
        except Exception as exc:
            last_err = exc
        time.sleep(2)
    raise RuntimeError(f"CUDA failed to initialize within {max_wait_s}s: {last_err}")

print("[0/6] Warming up CUDA driver ...")
_wait_for_cuda()

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
HF_TOKEN  = os.environ["HF_TOKEN"]
SFT_REPO  = os.environ.get("SFT_REPO", "Aaryan369/civicflow-sft-qwen2.5-3b")
OUT_REPO  = os.environ.get("PUSH_TO",  "Aaryan369/civicflow-grpo-qwen2.5-3b")
MAX_SEQ   = int(os.environ.get("MAX_SEQ_LEN", 2048))
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN", 1024))
MAX_COMPLETION_LEN = int(os.environ.get("MAX_COMPLETION_LEN", 256))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 150))
ROLLOUTS  = int(os.environ.get("ROLLOUTS_PER_PROMPT", 4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 4))
LR        = float(os.environ.get("LR", 5e-6))
KL_BETA   = float(os.environ.get("KL_BETA", 0.05))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 1.0))
TOP_P     = float(os.environ.get("TOP_P", 1.0))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", 0.55))
DISABLE_VLLM = os.environ.get("DISABLE_VLLM", "0").lower() in {"1", "true", "yes"}
USE_VLLM  = not DISABLE_VLLM
VLLM_ENFORCE_EAGER = True
OUT_DIR   = "/tmp/grpo_adapter"

if USE_VLLM:
    os.environ.setdefault("UNSLOTH_VLLM_NO_FLASHINFER", "1")

GRPO_TASKS = ["tiny_a", "medium_a", "medium_b", "hard_a"]
SYSTEM = (
    "You are a CivicFlow city planner. Read the state and reply with one JSON "
    "object only: the next action. No markdown, no explanation."
)

print("=" * 70)
print(" CIVICFLOW GRPO TRAINING")
print("=" * 70)
print(f"  sft_repo:     {SFT_REPO}")
print(f"  push_to:      {OUT_REPO}")
print(f"  max_steps:    {MAX_STEPS}")
print(f"  rollouts:     {ROLLOUTS}")
print(f"  use_vllm:     {USE_VLLM}")
print()

# ─────────────────────────────────────────────────────────────────────
# 1. PatchFastRL FIRST
# ─────────────────────────────────────────────────────────────────────
print("[1/6] Applying PatchFastRL('GRPO', FastLanguageModel) ...")
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
print("      ✓ PatchFastRL applied")


def _patch_unsloth_matmul_lora_dtype_safe() -> None:
    """Tracks Unsloth PR #4918 — fix dtype mismatch in fused LoRA matmul."""
    import torch as _t
    try:
        import unsloth.kernels.utils as _u
        import unsloth.kernels.fast_lora as _fl
    except Exception as exc:
        print(f"      ! could not import unsloth kernels: {exc}")
        return

    _fast_dequantize = _u.fast_dequantize
    _torch_matmul = _u.torch_matmul
    _Float8Tensor = getattr(_u, "Float8Tensor", None)
    _fp8_linear = getattr(_u, "fp8_linear", None)

    def matmul_lora(X, W, W_quant, A, B, s, out=None):
        reshape = False
        if X.dim() == 3:
            batch, seq_len, _d = X.shape
            X = X.view(-1, X.shape[-1])
            reshape = True

        if _Float8Tensor is not None and isinstance(W, _Float8Tensor):
            if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
                W_full = W.dequantize()
            else:
                W_full = W.contiguous()
            if X.dtype != W_full.dtype:
                X = X.to(W_full.dtype)
            out = _torch_matmul(X, W_full.t(), out=out)
        elif _fp8_linear is not None and W.dtype == _t.float8_e4m3fn:
            out = _fp8_linear(X, W, W_quant)
        else:
            W_full = _fast_dequantize(W, W_quant, use_global_buffer=True)
            if X.dtype != W_full.dtype:
                X = X.to(W_full.dtype)
            out = _torch_matmul(X, W_full.t(), out=out)
            if W_quant is not None:
                del W_full

        if A is not None:
            dtype = X.dtype
            At, Bt = A.t(), B.t()
            XA = _torch_matmul(X, At.to(dtype))
            out.addmm_(XA, Bt.to(dtype), alpha=s)

        return out.view(batch, seq_len, -1) if reshape else out

    matmul_lora.__name__ = "matmul_lora_dtype_safe"
    _u.matmul_lora = matmul_lora
    _fl.matmul_lora = matmul_lora


_patch_unsloth_matmul_lora_dtype_safe()
print("      ✓ matmul_lora hot-patched (PR #4918)")


def _patch_vllm_lora_manager_signature() -> None:
    """Patch BOTH the class methods and the v1 call site.

    vllm 0.19.1's `vllm/v1/worker/lora_model_runner_mixin.py` calls
    `self.lora_manager.create_lora_manager(model, vllm_config)` but the
    base class only accepts `(self, model)`. Earliest reliable fix:
    rewrite the call site to drop `vllm_config` before it ever calls
    the manager.
    """
    if not USE_VLLM:
        return
    import sys, importlib

    # 1) Force-import all the relevant vllm submodules so classes exist.
    for modname in (
        "vllm",
        "vllm.lora",
        "vllm.lora.worker_manager",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.lora_model_runner_mixin",
    ):
        try:
            importlib.import_module(modname)
        except Exception as exc:
            print(f"      ! couldn't import {modname}: {exc}")

    # 2) Patch every loaded class named *WorkerLoRAManager.
    patched_classes = 0
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for name in dir(mod):
            try:
                cls = getattr(mod, name)
            except Exception:
                continue
            if not isinstance(cls, type):
                continue
            if cls.__name__ not in ("LRUCacheWorkerLoRAManager", "WorkerLoRAManager"):
                continue
            if not hasattr(cls, "create_lora_manager"):
                continue
            original = cls.create_lora_manager
            if getattr(original, "_argdrop", False):
                continue
            def _make(_orig):
                def wrapped(self, model, *args, **kwargs):
                    return _orig(self, model)
                wrapped._argdrop = True
                return wrapped
            try:
                cls.create_lora_manager = _make(original)
                patched_classes += 1
                print(f"      ✓ patched {cls.__module__}.{cls.__name__}.create_lora_manager")
            except Exception as exc:
                print(f"      ! could not patch {cls.__name__}: {exc}")

    # 3) Patch the v1 call site directly (this is where the crash happens).
    try:
        import vllm.v1.worker.lora_model_runner_mixin as _mixin
        mixin_cls = getattr(_mixin, "LoRAModelRunnerMixin", None)
        if mixin_cls is not None and hasattr(mixin_cls, "load_lora_model"):
            original_load = mixin_cls.load_lora_model
            if not getattr(original_load, "_argdrop_call_site", False):
                def patched_load_lora_model(self, model, *args, **kwargs):
                    # Old API: create_lora_manager(model) only.
                    return self.lora_manager.create_lora_manager(model)
                patched_load_lora_model._argdrop_call_site = True
                mixin_cls.load_lora_model = patched_load_lora_model
                print(f"      ✓ patched LoRAModelRunnerMixin.load_lora_model call site")
    except Exception as exc:
        print(f"      ! could not patch v1 mixin: {exc}")

    if patched_classes == 0:
        print("      ! WARNING: no WorkerLoRAManager classes were patched — relying on call-site patch")


_patch_vllm_lora_manager_signature()
if USE_VLLM:
    print("      ✓ vllm LoRA manager hot-patched (issue #3962)")


def _force_vllm_enforce_eager() -> None:
    if not USE_VLLM:
        return
    try:
        import vllm as _vllm
    except Exception as exc:
        print(f"      ! could not import vllm: {exc}")
        return
    LLM_cls = getattr(_vllm, "LLM", None)
    if LLM_cls is None:
        return
    original_init = LLM_cls.__init__
    if getattr(original_init, "_eager", False):
        return
    def _wrapped(self, *args, **kwargs):
        kwargs.setdefault("enforce_eager", True)
        return original_init(self, *args, **kwargs)
    _wrapped._eager = True
    LLM_cls.__init__ = _wrapped


_force_vllm_enforce_eager()
if USE_VLLM:
    print("      ✓ vllm.LLM enforce_eager=True default")

# ─────────────────────────────────────────────────────────────────────
# 2. Login + Trackio
# ─────────────────────────────────────────────────────────────────────
print("[2/6] Login + Trackio init ...")
from huggingface_hub import login, HfApi
import trackio
login(token=HF_TOKEN, add_to_git_credential=False)
trackio.init(
    project="civicflow",
    name="grpo-qwen2.5-3b",
    space_id="Aaryan369/civicflow-trackio",
    config={"start_from": SFT_REPO, "stage": "grpo"},
)

# ─────────────────────────────────────────────────────────────────────
# 3. Load SFT-adapted model (continue training same LoRA)
# ─────────────────────────────────────────────────────────────────────
print(f"[3/6] Loading SFT-adapted model from {SFT_REPO} ...")
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_REPO,
    max_seq_length=MAX_SEQ,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    fast_inference=USE_VLLM,
    max_lora_rank=16,
    gpu_memory_utilization=GPU_MEM_UTIL,
    token=HF_TOKEN,
)
print("      ✓ Model + SFT LoRA loaded; GRPO continues training same LoRA")

# ─────────────────────────────────────────────────────────────────────
# 4. Prompts dataset
# ─────────────────────────────────────────────────────────────────────
print("[4/6] Building prompt dataset ...")
from datasets import Dataset

def _render_state_nl(world, obs) -> str:
    """Render env state as ### STATE-style NL — must match SFT format closely."""
    ps = obs.planning_summary or {}
    support = ps.get("planner_support", {})
    legal = obs.legal_actions_summary or {}
    lines = [
        f"Task: {world.task_id}",
        f"Step: {obs.step_index} | Phase: {obs.current_phase} — {obs.phase_objective}",
        "",
        "Briefing:",
        (obs.briefing or "").strip(),
        "",
        "Targets still to satisfy:",
    ]
    rem_u = support.get("remaining_use_targets") or {}
    if rem_u:
        lines.append("  Uses: " + ", ".join(f"{k}={v}" for k, v in sorted(rem_u.items()) if v > 0))
    else:
        lines.append("  Uses: (none)")
    am = support.get("remaining_amenities") or []
    lines.append("  Amenities: " + (", ".join(am) if am else "(none)"))
    gn = support.get("green_blocks_needed", 0)
    lines.append(f"  Extra green / open-space blocks needed (approx): {gn}")
    lines.append(f"  City greenery ratio now: {ps.get('greenery_ratio', 0)} (floor {world.targets.min_greenery_ratio})")
    lines.append("")
    lines.append("Blocks (order is arbitrary; use block_id):")
    blocks_iter = list(world.blocks.values()) if isinstance(world.blocks, dict) else list(world.blocks or [])
    for b in blocks_iter[:30]:
        seg = f"  - {b.block_id}: {'zoned' if b.zone else 'unzoned'}, {'built' if b.use else 'not built'}"
        if getattr(b, "future_designation", None):
            seg += f" — designation={b.future_designation}"
        if not getattr(b, "has_road_access", True):
            seg += " — no road access"
        if getattr(b, "is_protected", False):
            seg += " — protected land"
        seg += f" infra {b.infra_zone}; demands water={b.water_demand}, sewer={b.sewer_demand}, power={b.power_demand}"
        lines.append(seg)
    lines.append("")
    lines.append("Infrastructure (alloc / capacity per zone):")
    for zid, caps in (ps.get("infra_zones") or {}).items():
        lines.append(f"  {zid}: {caps}")
    lines.append("")
    lines.append("Hint: candidate block ids by action family (not exhaustive):")
    for fam, ids in (legal or {}).items():
        if isinstance(ids, list) and ids:
            preview = ", ".join(ids[:6]) + (f" (+{len(ids)-6} more)" if len(ids) > 6 else "")
            lines.append(f"  {fam}: {preview}")
    lines.append("")
    lines.append("Last action: (episode start)")
    lines.append("")
    lines.append("Return the next action as a JSON object, nothing else.")
    return "\n".join(lines)


def build_prompts(n_per_task: int = 16):
    """Build prompts in EXACTLY the SFT format: system + ### STATE/### TASK user."""
    from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
    rows = []
    for task in GRPO_TASKS:
        for _ in range(n_per_task):
            os.environ["CIVICFLOW_TASK_ID"] = task
            env = CivicflowEnvironment()
            obs = env.reset()
            world = env._world
            state_nl = _render_state_nl(world, obs)
            user_msg = f"### STATE\n{state_nl}\n\n### TASK\nGenerate the next optimal action as JSON."
            prompt_text = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            rows.append({"prompt": prompt_text, "task_id": task})
    return Dataset.from_list(rows)


prompts_ds = build_prompts(n_per_task=16)
print(f"      ✓ {len(prompts_ds)} prompts")

# ─────────────────────────────────────────────────────────────────────
# 5. Reward fn — in-process verifier
# ─────────────────────────────────────────────────────────────────────
import re

_VALID_ACTION_TYPES = {
    "set_zoning", "develop", "reserve_open_space", "upgrade_infrastructure",
    "assign_amenity", "redevelop", "defer",
}
# JSON object — handles nested-free actions, optional code fences.
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_DEBUG_PRINTED = {"n": 0}
_STATS = {"parse_ok": 0, "parse_fail": 0, "legal": 0, "illegal": 0, "total": 0}


def _extract_text(comp):
    if isinstance(comp, list) and comp:
        return comp[0].get("content", "") if isinstance(comp[0], dict) else str(comp[0])
    return str(comp)


def _parse_action(text):
    """Return (payload_dict, raw_match) or (None, None)."""
    # Strip fences
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # Find first {...}
    m = _JSON_RE.search(t)
    if not m:
        return None, None
    try:
        return json.loads(m.group(0)), m.group(0)
    except Exception:
        return None, None


def _run_episode(task_id, completion_text, max_steps=8):
    """Roll a multi-step episode where step 1 is the model's action.

    For steps 2..max_steps we use a `defer` no-op so the env keeps progressing
    (useful to let `final_valid_plan` and other terminal signals fire).
    Returns dict with keys: parse_ok, legal, env_reward, components.
    """
    from civicflow_env.server.civicflow_env_environment import CivicflowEnvironment
    from civicflow_env.models import CivicflowAction

    payload, raw = _parse_action(completion_text)
    result = {
        "parse_ok": 0.0, "legal": 0.0, "env_reward": 0.0,
        "valid_action_type": 0.0, "raw_emitted": completion_text[:200],
    }
    if payload is None or not isinstance(payload, dict):
        return result
    result["parse_ok"] = 1.0
    if payload.get("action_type") in _VALID_ACTION_TYPES:
        result["valid_action_type"] = 1.0

    try:
        os.environ["CIVICFLOW_TASK_ID"] = task_id
        env = CivicflowEnvironment()
        env.reset()
        # Step 1: model's action
        obs = env.step(CivicflowAction(**{k: v for k, v in payload.items()
                                          if k in CivicflowAction.model_fields}))
        cum = float(obs.reward or 0.0)
        # Heuristic: if illegal_action metric is 0, we count this as legal
        is_legal = float(obs.last_metrics.get("illegal_action", 1)) == 0.0
        result["legal"] = 1.0 if is_legal else 0.0
        # Roll forward a few defer steps to surface terminal signal cheaply
        for _ in range(max_steps - 1):
            if obs.done:
                break
            try:
                obs = env.step(CivicflowAction(action_type="defer", phase_id=0))
                cum += float(obs.reward or 0.0)
            except Exception:
                break
        result["env_reward"] = cum
    except Exception as e:
        result["env_reward"] = -2.0  # distinct from legitimate -1.0 penalty
        result["raw_emitted"] = f"[exc:{type(e).__name__}] " + result["raw_emitted"]
    return result


_episode_cache = {}

def _get_episode(task_id, text):
    key = (task_id, text)
    if key not in _episode_cache:
        _episode_cache[key] = _run_episode(task_id, text)
        _STATS["total"] += 1
        if _episode_cache[key]["parse_ok"]:
            _STATS["parse_ok"] += 1
        else:
            _STATS["parse_fail"] += 1
        if _episode_cache[key]["legal"]:
            _STATS["legal"] += 1
        elif _episode_cache[key]["parse_ok"]:
            _STATS["illegal"] += 1

        # Log first few completions to stdout so we can SEE what the model emits
        if _DEBUG_PRINTED["n"] < 8:
            r = _episode_cache[key]
            print(f"[reward.debug] task={task_id} parse={r['parse_ok']} "
                  f"legal={r['legal']} env_r={r['env_reward']:+.3f} "
                  f"emit={r['raw_emitted']!r}", flush=True)
            _DEBUG_PRINTED["n"] += 1
        # Periodic aggregate
        if _STATS["total"] % 50 == 0:
            tot = _STATS["total"]
            print(f"[reward.stats] n={tot} parse_ok={_STATS['parse_ok']/tot:.2%} "
                  f"legal={_STATS['legal']/tot:.2%} "
                  f"illegal={_STATS['illegal']/tot:.2%}", flush=True)
    return _episode_cache[key]


def reward_format(prompts, completions, task_id, **_):
    return [_get_episode(t, _extract_text(c))["parse_ok"] for c, t in zip(completions, task_id)]

def reward_action_type(prompts, completions, task_id, **_):
    return [_get_episode(t, _extract_text(c))["valid_action_type"] for c, t in zip(completions, task_id)]

def reward_legal(prompts, completions, task_id, **_):
    return [_get_episode(t, _extract_text(c))["legal"] for c, t in zip(completions, task_id)]

def reward_env(prompts, completions, task_id, **_):
    return [_get_episode(t, _extract_text(c))["env_reward"] for c, t in zip(completions, task_id)]

def reward_composite(prompts, completions, task_id, **_):
    """Bounded [0, 1] training signal — env_reward is monitor-only.

    Including the unbounded -10 env_reward as a training signal causes
    catastrophic advantage spikes (kl→300+, grad_norm→9000). Restricting
    the gradient to bounded format/type/legal keeps GRPO stable; env
    reward is still tracked as a separate metric.
    """
    out = []
    for c, t in zip(completions, task_id):
        r = _get_episode(t, _extract_text(c))
        composite = (
            0.25 * r["parse_ok"]
            + 0.25 * r["valid_action_type"]
            + 0.50 * r["legal"]
        )
        out.append(composite)
    return out

# ─────────────────────────────────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────────────────────────────────
print("[5/6] Configuring GRPOTrainer ...")
from trl import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(
    output_dir=OUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_generations=ROLLOUTS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    max_steps=MAX_STEPS,
    save_steps=100,
    save_total_limit=1,
    logging_steps=1,
    bf16=True,
    beta=KL_BETA,
    max_grad_norm=0.3,
    report_to=["trackio"],
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    temperature=TEMPERATURE,
    top_p=TOP_P,
    remove_unused_columns=False,
    use_vllm=USE_VLLM,
    vllm_mode="colocate",
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_composite,
        reward_format,
        reward_action_type,
        reward_legal,
        reward_env,
    ],
    args=cfg,
    train_dataset=prompts_ds,
)

print(f"[6/6] Training GRPO for {MAX_STEPS} steps ...")
trainer.train()
print("      ✓ GRPO training complete")

# Save adapter only (no merge — keeps quality, smaller upload)
print(f"\nSaving LoRA adapter to {OUT_DIR}")
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"Pushing to https://huggingface.co/{OUT_REPO}")
model.push_to_hub(OUT_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(OUT_REPO, token=HF_TOKEN)
trackio.finish()
print("=" * 70)
print(" DONE")
print("=" * 70)
