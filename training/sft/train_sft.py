# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth==2025.10.10",
#   "trl>=0.23.0",
#   "transformers>=4.46.0",
#   "datasets>=3.0.0",
#   "accelerate>=1.0.0",
#   "peft>=0.13.0",
#   "trackio>=0.0.7",
#   "huggingface_hub>=0.27.0",
# ]
# ///
"""SFT warm-start for CivicFlow on HF Jobs (A100-large)."""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, login
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
import trackio

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT_REPO = "Aaryan369/civicflow-sft-qwen2.5-3b"
DATA_REPO = "Aaryan369/civicflow-sft-data"
SFT_FILE = "sft_final.txt"
MAX_SEQ = 3072

SYSTEM = (
    "You are a CivicFlow city planner. Read the state and reply with one JSON "
    "object only: the next action. No markdown, no explanation."
)

def load_sft_txt(path: str):
    rows = []
    truncated_hint = 0
    for chunk in Path(path).read_text(encoding="utf-8").split("\n---\n"):
        chunk = chunk.strip()
        if "### ACTION" not in chunk:
            continue
        head, _, action = chunk.partition("### ACTION")
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": head.strip()},
                {"role": "assistant", "content": action.strip()},
            ]
        })
    print(f"[data] loaded {len(rows)} examples")
    return rows


def main():
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    trackio.init(
        project="civicflow",
        name="sft-qwen2.5-3b",
        space_id="Aaryan369/civicflow-trackio",
        config={"model": MODEL_ID, "stage": "sft", "max_seq": MAX_SEQ},
    )

    # Pull SFT corpus from the env repo we just pushed
    local_txt = hf_hub_download(
        repo_id=DATA_REPO, repo_type="dataset",
        filename=SFT_FILE, token=hf_token,
    )
    rows = load_sft_txt(local_txt)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        token=hf_token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def _format(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    dataset = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)}
        for r in rows
    ])

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    cfg = SFTConfig(
        output_dir="./out_sft",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        max_length=MAX_SEQ,
        packing=False,
        report_to=["trackio"],
        push_to_hub=True,
        hub_model_id=OUT_REPO,
        hub_strategy="end",
        hub_private_repo=False,
        hub_token=hf_token,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=cfg,
        dataset_text_field="text",
    )
    trainer.train()

    # Save merged 16-bit weights so GRPO can load via standard HF loaders
    model.save_pretrained_merged("./out_sft_merged", tokenizer, save_method="merged_16bit")
    from huggingface_hub import HfApi
    HfApi(token=hf_token).upload_folder(
        folder_path="./out_sft_merged",
        repo_id=OUT_REPO,
        repo_type="model",
    )
    trackio.finish()
    print(f"[done] pushed {OUT_REPO}")


if __name__ == "__main__":
    main()
