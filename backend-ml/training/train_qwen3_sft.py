"""
train_qwen3_sft.py - Supervised fine-tuning for ShadowOps action policy.

This script replaces unstable GRPO with SFT on teacher-labeled actions.
Dataset format (JSONL preferred):
  {"prompt": "...", "completion": "ALLOW|BLOCK|FORK|QUARANTINE", "metadata": {...}}

Usage:
  python training/train_qwen3_sft.py
  python training/train_qwen3_sft.py --max-steps 400 --output-dir ./shadowops_qwen3_1p7b_sft_model
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VALID_ACTIONS = {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[dict] = []

    # Support JSON array and JSONL.
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON dataset must be a list of records.")
        records = parsed
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def normalize_records(records: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    dropped = 0

    for rec in records:
        prompt = str(rec.get("prompt", "")).strip()
        completion = rec.get("completion", rec.get("response", ""))
        completion = str(completion).strip().upper()

        if not prompt or completion not in VALID_ACTIONS:
            dropped += 1
            continue

        # Keep training target concise and deterministic.
        normalized.append(
            {
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt}{completion}",
            }
        )

    if not normalized:
        raise ValueError("No valid records found after normalization.")

    print(f"Loaded valid records: {len(normalized)} (dropped: {dropped})")
    return normalized


def train_sft(args: argparse.Namespace) -> None:
    # Import Unsloth before Transformers to avoid patch-order warnings.
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dataset_path = Path(args.dataset)
    raw_records = load_dataset(dataset_path)
    records = normalize_records(raw_records)

    random.Random(args.seed).shuffle(records)

    split_idx = int(len(records) * (1.0 - args.eval_ratio))
    split_idx = max(1, min(split_idx, len(records) - 1))
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    print(f"Train/Eval split: {len(train_records)} / {len(eval_records)}")

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        fast_inference=False,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
        bf16=args.bf16,
        fp16=not args.bf16,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    print("Starting SFT training...")
    result = trainer.train()
    metrics = result.metrics if hasattr(result, "metrics") else {}
    print(f"Training complete. Metrics: {metrics}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShadowOps SFT trainer (Qwen3-1.7B + LoRA)")
    parser.add_argument("--dataset", default="training/qwen3_train_dataset.json")
    parser.add_argument("--model-name", default="unsloth/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="./shadowops_qwen3_1p7b_sft_model")

    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-ratio", type=float, default=0.05)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use BF16 instead of FP16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)
