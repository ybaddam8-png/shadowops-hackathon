"""
Supervised fine-tuning warm-start for ShadowOps Qwen3 adapters.

This script prepares the action-only ShadowOps task before GRPO.
It is intentionally safe to smoke-test on a laptop with --smoke-test
or --skip-model-load and is intended to run full training on cloud GPU.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.shadowops_training_common import (  # noqa: E402
    DEFAULT_SFT_METRICS_PATH,
    DEFAULT_SFT_OUTPUT_DIR,
    MODEL_OPTIONS,
    add_shared_smoke_args,
    build_sft_training_text,
    compact_metrics,
    compute_parameter_delta,
    configure_runtime_noise_filters,
    configure_torch_runtime,
    count_trainable_parameters,
    ensure_dirs,
    ensure_trainable_lora,
    evaluate_model_on_dataset,
    generate_datasets,
    load_fast_model,
    load_training_stack,
    preflight_dataset_check,
    run_logic_smoke_test,
    run_parse_action_tests,
    select_supported_kwargs,
    validate_training_runtime,
    capture_trainable_snapshot,
    write_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ShadowOps SFT warm-start adapter.")
    parser.add_argument("--run-sft", action="store_true", help="Explicit compatibility flag; SFT is the default action for this script.")
    parser.add_argument("--skip-grpo", action="store_true", help="Compatibility flag for combined pipelines; ignored in this script.")
    parser.add_argument("--model-name", default=MODEL_OPTIONS["1.7b"], help="Base Qwen3 model name.")
    parser.add_argument("--sft-epochs", type=float, default=2.0, help="Number of SFT epochs.")
    parser.add_argument("--learning-rate", "--sft-learning-rate", dest="learning_rate", type=float, default=2e-4, help="SFT learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--sft-output-dir", type=Path, default=DEFAULT_SFT_OUTPUT_DIR, help="Adapter output directory.")
    parser.add_argument("--sft-train-size", type=int, default=500, help="Training sample count.")
    parser.add_argument("--sft-val-size", type=int, default=100, help="Validation sample count.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Trainer logging interval.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint save interval.")
    add_shared_smoke_args(parser)
    return parser


def train_sft(args: argparse.Namespace) -> int:
    ensure_dirs()

    if args.smoke_test:
        smoke = run_logic_smoke_test()
        print("SFT smoke test:", "PASS" if smoke["smoke_test_passed"] else "FAIL")
        print(f"Final report: training/final_training_report.json")
        return 0 if smoke["smoke_test_passed"] else 1

    train_samples, val_samples, audit = generate_datasets(
        train_size=args.sft_train_size,
        val_size=args.sft_val_size,
        save=True,
    )
    preflight_dataset_check(audit)

    if args.dry_run or args.skip_model_load:
        parse_tests = run_parse_action_tests()
        dry_metrics = {
            "mode": "dry-run" if args.dry_run else "skip-model-load",
            "parse_tests": parse_tests,
            "dataset_audit_path": "training/dataset_audit.json",
            "output_dir": str(args.sft_output_dir.relative_to(ROOT_DIR)) if args.sft_output_dir.is_relative_to(ROOT_DIR) else str(args.sft_output_dir),
        }
        write_json(DEFAULT_SFT_METRICS_PATH, dry_metrics)
        print("SFT dry-run complete.")
        return 0

    if not validate_training_runtime(args.model_name):
        return 1

    configure_runtime_noise_filters()
    training_stack = load_training_stack("sft")
    if training_stack is None:
        return 1

    torch_module = training_stack["torch"]
    configure_torch_runtime(torch_module)
    Dataset = training_stack["Dataset"]
    TrainerConfig = training_stack["TrainerConfig"]
    TrainerClass = training_stack["TrainerClass"]
    FastModel = training_stack["FastModel"]

    print(f"Loading SFT base model: {args.model_name}")
    model, tokenizer = load_fast_model(FastModel, args.model_name, max_seq_len=args.max_seq_len)
    model = ensure_trainable_lora(
        FastModel,
        model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    base_metrics = evaluate_model_on_dataset(
        model,
        tokenizer,
        val_samples,
        batch_size=4,
        max_seq_len=args.max_seq_len,
        max_new_tokens=8,
    )
    base_metrics["label"] = "raw_model_before_sft"

    trainable_params, total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    train_records = [
        {"text": build_sft_training_text(tokenizer, sample["prompt"], sample["completion"])}
        for sample in train_samples
    ]
    val_records = [
        {"text": build_sft_training_text(tokenizer, sample["prompt"], sample["completion"])}
        for sample in val_samples
    ]
    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(val_records)

    trainer_config_kwargs = {
        "output_dir": str(args.sft_output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.sft_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": 42,
        "max_seq_length": args.max_seq_len,
        "fp16": not torch_module.cuda.is_bf16_supported(),
        "bf16": torch_module.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
        "packing": False,
    }
    trainer_config_kwargs, dropped_config = select_supported_kwargs(TrainerConfig.__init__, trainer_config_kwargs)
    if dropped_config:
        print("Skipping unsupported SFTConfig args:", ", ".join(dropped_config))
    trainer_config = TrainerConfig(**trainer_config_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "dataset_text_field": "text",
    }
    trainer_signature = inspect.signature(TrainerClass.__init__)
    trainer_param_names = set(trainer_signature.parameters)
    if "args" in trainer_param_names:
        trainer_kwargs["args"] = trainer_config
    elif "config" in trainer_param_names:
        trainer_kwargs["config"] = trainer_config
    if "processing_class" in trainer_param_names:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_param_names:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer_kwargs, dropped_trainer = select_supported_kwargs(TrainerClass.__init__, trainer_kwargs)
    if dropped_trainer:
        print("Skipping unsupported SFTTrainer args:", ", ".join(dropped_trainer))

    before_snapshot = capture_trainable_snapshot(model)
    trainer = TrainerClass(**trainer_kwargs)
    print("Starting SFT training...")
    trainer.train()

    FastModel.for_inference(model)
    sft_metrics = evaluate_model_on_dataset(
        model,
        tokenizer,
        val_samples,
        batch_size=4,
        max_seq_len=args.max_seq_len,
        max_new_tokens=8,
    )
    sft_metrics["label"] = "sft_model"

    args.sft_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.sft_output_dir))
    tokenizer.save_pretrained(str(args.sft_output_dir))

    metrics_payload = {
        "model_name": args.model_name,
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "base_model_metrics": compact_metrics(base_metrics),
        "sft_metrics": compact_metrics(sft_metrics),
        "lora_parameter_delta": compute_parameter_delta(before_snapshot, model),
        "output_dir": str(args.sft_output_dir.relative_to(ROOT_DIR)) if args.sft_output_dir.is_relative_to(ROOT_DIR) else str(args.sft_output_dir),
    }
    write_json(DEFAULT_SFT_METRICS_PATH, metrics_payload)
    print(f"SFT adapter saved to {args.sft_output_dir}")
    print("SFT metrics written to training/sft_metrics.json")
    return 0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(train_sft(args))


if __name__ == "__main__":
    main()
