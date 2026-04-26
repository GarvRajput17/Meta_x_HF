# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.51.3,<4.57",
#   "peft==0.17.1",
#   "accelerate",
#   "huggingface_hub",
#   "safetensors",
# ]
# ///
"""Merge GRPO LoRA into SFT base on GPU and push a clean (un-quantized) repo.

Strategy: snapshot SFT repo locally, strip quantization_config from
config.json on disk, then load → merge GRPO adapter → save state_dict
manually → push. Avoids every fragile AutoConfig re-load path.
"""
import json
import os
import shutil

import torch
from huggingface_hub import HfApi, login, snapshot_download
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ["HF_TOKEN"]
SFT_REPO = "Aaryan369/civicflow-sft-qwen2.5-3b"
GRPO_ADAPTER = "Aaryan369/civicflow-grpo-qwen2.5-3b"
OUT_REPO = "Aaryan369/civicflow-grpo-merged-qwen2.5-3b"

login(token=HF_TOKEN)

# ─── 1. Snapshot SFT repo locally and clean its config.json on disk ───
print(f"[1/5] Snapshotting {SFT_REPO} ...")
local_sft = snapshot_download(
    SFT_REPO,
    token=HF_TOKEN,
    local_dir="/tmp/sft_local",
    local_dir_use_symlinks=False,
)
cfg_path = os.path.join(local_sft, "config.json")
# If the local file is a symlink into the HF cache, replace it with a real file
if os.path.islink(cfg_path):
    real = os.path.realpath(cfg_path)
    os.unlink(cfg_path)
    shutil.copyfile(real, cfg_path)
with open(cfg_path) as f:
    cfg_json = json.load(f)
cfg_json.pop("quantization_config", None)
cfg_json.pop("_load_in_4bit", None)
cfg_json.pop("_load_in_8bit", None)
cfg_json.pop("_quant_config", None)
cfg_json["torch_dtype"] = "bfloat16"
with open(cfg_path, "w") as f:
    json.dump(cfg_json, f, indent=2)
print(f"      ✓ stripped quantization_config, set torch_dtype=bfloat16")

# ─── 2. Load the cleaned base in bf16 (no quant path triggered) ───
print(f"[2/5] Loading cleaned base in bf16 ...")
config = AutoConfig.from_pretrained(local_sft)
for attr in ("quantization_config", "_pre_quantization_dtype"):
    if hasattr(config, attr):
        try:
            delattr(config, attr)
        except AttributeError:
            setattr(config, attr, None)
base = AutoModelForCausalLM.from_pretrained(
    local_sft,
    config=config,
    dtype=torch.bfloat16,
    device_map="auto",
)
print(f"      ✓ base dtype={next(base.parameters()).dtype}")

# ─── 3. Attach + merge GRPO adapter ───
print(f"[3/5] Loading GRPO adapter {GRPO_ADAPTER} and merging ...")
m = PeftModel.from_pretrained(base, GRPO_ADAPTER, token=HF_TOKEN)
merged = m.merge_and_unload()
print("      ✓ merge_and_unload complete")

# ─── 4. Manual save: state_dict + config (no peft/quant residue) ───
print("[4/5] Saving merged model manually ...")
out_dir = "/tmp/grpo_merged"
os.makedirs(out_dir, exist_ok=True)

# Reuse the cleaned config dict directly — don't go through merged.config.save_pretrained
# because that may re-emit quant fields via attribute_map.
out_cfg = dict(cfg_json)
out_cfg.pop("quantization_config", None)
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(out_cfg, f, indent=2)

# State dict as a single safetensors file (Qwen 3B ~6GB in bf16, fits in one shard)
state = {
    k: v.detach().to("cpu", torch.bfloat16).contiguous()
    for k, v in merged.state_dict().items()
}
save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
print(f"      ✓ wrote {len(state)} tensors → model.safetensors")

# generation_config (if present)
gen_path = os.path.join(local_sft, "generation_config.json")
if os.path.exists(gen_path):
    shutil.copy(gen_path, os.path.join(out_dir, "generation_config.json"))

# Tokenizer (use the GRPO repo's tokenizer for consistency with how it was trained)
tok = AutoTokenizer.from_pretrained(GRPO_ADAPTER, token=HF_TOKEN)
tok.save_pretrained(out_dir)
print(f"      ✓ tokenizer saved")

# ─── 5. Push ───
print(f"[5/5] Pushing to {OUT_REPO} ...")
api = HfApi(token=HF_TOKEN)
api.create_repo(OUT_REPO, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=out_dir, repo_id=OUT_REPO, repo_type="model")
print(f"DONE -> https://huggingface.co/{OUT_REPO}")
