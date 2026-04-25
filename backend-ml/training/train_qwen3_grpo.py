"""
ShadowOps GRPO training entrypoint for Qwen3.

Design goals:
- SFT warm-start teaches the action task first.
- GRPO resumes from the SFT adapter, not the raw base model.
- Evaluation scores raw base, SFT, and SFT+GRPO on the same validation split.
- Smoke tests stay laptop-safe and do not load large models unless requested.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import sys
from pathlib import Path
from xml.parsers.expat import model

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.shadowops_training_common import (  # noqa: E402
    CLOUD_FALLBACK_COMMAND,
    CLOUD_GRPO_COMMAND,
    CLOUD_SFT_COMMAND,
    DEFAULT_DEMO_BENCHMARK_JSON,
    DEFAULT_DEMO_BENCHMARK_MD,
    DEFAULT_GRPO_OUTPUT_DIR,
    DEFAULT_HEALTH_REPORT_PATH,
    DEFAULT_MODEL_EVAL_JSON,
    DEFAULT_MODEL_EVAL_MD,
    DEFAULT_MODEL_POLICY_COMPARISON_JSON,
    DEFAULT_MODEL_POLICY_COMPARISON_MD,
    DEFAULT_SFT_METRICS_PATH,
    DEFAULT_SFT_OUTPUT_DIR,
    MODEL_OPTIONS,
    RewardHealthTracker,
    TrainingPreflightError,
    add_shared_smoke_args,
    build_evaluation_bundle,
    build_training_health_report,
    capture_trainable_snapshot,
    compact_metrics,
    compute_parameter_delta,
    configure_runtime_noise_filters,
    configure_torch_runtime,
    ensure_dirs,
    ensure_trainable_lora,
    evaluate_model_on_dataset,
    evaluate_saved_model,
    evaluate_policy_on_dataset,
    evaluate_training_gate,
    format_prompt_for_model,
    generate_datasets,
    load_hard_negative_samples,
    load_validation_samples_for_benchmark,
    get_total_gpu_memory_gb,
    load_fast_model,
    load_training_stack,
    make_health_callback,
    make_reward_function,
    metric_delta,
    pick_best_checkpoint,
    preflight_dataset_check,
    read_json,
    run_logic_smoke_test,
    run_demo_benchmark,
    run_model_policy_comparison,
    run_reward_diagnostics,
    print_reward_diagnostics,
    run_parse_action_tests,
    run_subprocess,
    select_supported_kwargs,
    validate_training_runtime,
    check_reward_variance,
    write_json,
    generate_final_reports,
)


def resolve_repo_path(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    return (ROOT_DIR / path_value).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ShadowOps GRPO adapter from an SFT warm-start.")
    parser.add_argument("--run-sft", action="store_true", help="Run SFT before GRPO using the current CLI settings.")
    parser.add_argument("--skip-grpo", action="store_true", help="Run preflight and optional SFT, but stop before GRPO.")
    parser.add_argument("--evaluate-baselines-only", action="store_true", help="Evaluate Random, Heuristic, Q-aware, and Oracle baselines without loading a model.")
    parser.add_argument("--evaluate-model", action="store_true", help="Evaluate a saved checkpoint against validation data and Q-aware policy.")
    parser.add_argument("--model-path", type=Path, default=None, help="Saved model/checkpoint path for --evaluate-model.")
    parser.add_argument("--compare-against-policy", action="store_true", help="Print explicit deltas versus Q-aware policy in model evaluation mode.")
    parser.add_argument("--eval-report-path", type=Path, default=None, help="Optional JSON path for the model evaluation report.")
    parser.add_argument("--eval-split", choices=("validation", "hard_negative", "combined"), default="validation", help="Dataset split for --evaluate-model.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap for model evaluation samples.")
    parser.add_argument("--reward-diagnostics", action="store_true", help="Print reward variation diagnostics without loading a model.")
    parser.add_argument("--model-name", default=MODEL_OPTIONS["1.7b"], help="Base Qwen3 model name.")
    parser.add_argument("--resume-from-sft", type=Path, default=DEFAULT_SFT_OUTPUT_DIR, help="Path to the SFT adapter to resume from.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRPO_OUTPUT_DIR, help="GRPO adapter output directory.")
    parser.add_argument("--train-size", "--sft-train-size", dest="train_size", type=int, default=500, help="Training sample count.")
    parser.add_argument("--val-size", "--sft-val-size", dest="val_size", type=int, default=100, help="Validation sample count.")
    parser.add_argument("--sft-epochs", type=float, default=2.0, help="SFT epochs when --run-sft is used.")
    parser.add_argument("--sft-learning-rate", type=float, default=2e-4, help="SFT learning rate when --run-sft is used.")
    parser.add_argument("--sft-output-dir", type=Path, default=DEFAULT_SFT_OUTPUT_DIR, help="SFT output directory when --run-sft is used.")
    parser.add_argument("--max-steps", type=int, default=800, help="GRPO max steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="GRPO learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device GRPO batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="GRPO gradient accumulation steps.")
    parser.add_argument("--num-generations", type=int, default=8, help="Number of sampled completions per prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="GRPO sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="GRPO top-p.")
    parser.add_argument("--top-k", type=int, default=50, help="GRPO top-k.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Maximum generated action tokens.")
    parser.add_argument("--explanation-max-new-tokens", type=int, default=96, help="Token budget reserved for future structured explanation generation.")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum prompt sequence length.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Checkpoint / evaluation interval.")
    parser.add_argument("--val-eval-eps", type=int, default=100, help="Validation examples to score for reports.")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Trainer logging interval.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trainer configuration.")
    add_shared_smoke_args(parser)
    return parser


def maybe_run_sft(args: argparse.Namespace) -> None:
    if not args.run_sft or args.dry_run or args.skip_model_load:
        return

    sft_script = ROOT_DIR / "training" / "train_qwen3_sft.py"
    try:
        sft_script_rel = str(sft_script.relative_to(ROOT_DIR))
    except ValueError:
        sft_script_rel = str(sft_script)
    command = [
        sys.executable,
        sft_script_rel,
        "--model-name",
        args.model_name,
        "--sft-epochs",
        str(args.sft_epochs),
        "--batch-size",
        "1",
        "--grad-accum",
        "8",
        "--max-seq-len",
        str(args.max_seq_len),
        "--learning-rate",
        str(args.sft_learning_rate),
        "--sft-output-dir",
        str(args.sft_output_dir),
        "--sft-train-size",
        str(args.train_size),
        "--sft-val-size",
        str(max(args.val_size, args.val_eval_eps)),
    ]
    print("Running SFT warm-start before GRPO...")
    run_subprocess(command, ROOT_DIR)


def evaluate_all_variants(
    *,
    model_name: str,
    resume_from_sft: Path,
    grpo_output_dir: Path,
    eval_samples: list[dict],
    max_seq_len: int,
    eval_batch_size: int,
    max_new_tokens: int,
) -> tuple[dict | None, dict | None, dict | None]:
    raw_metrics = evaluate_saved_model(
        model_path_or_name=model_name,
        val_samples=eval_samples,
        max_seq_len=max_seq_len,
        batch_size=eval_batch_size,
        max_new_tokens=max_new_tokens,
    )
    if raw_metrics is not None:
        raw_metrics["label"] = "raw_model"

    sft_metrics = None
    if resume_from_sft.exists():
        sft_metrics = evaluate_saved_model(
            model_path_or_name=str(resume_from_sft),
            val_samples=eval_samples,
            max_seq_len=max_seq_len,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
        )
        if sft_metrics is not None:
            sft_metrics["label"] = "sft_model"

    grpo_metrics = None
    if grpo_output_dir.exists():
        grpo_metrics = evaluate_saved_model(
            model_path_or_name=str(grpo_output_dir),
            val_samples=eval_samples,
            max_seq_len=max_seq_len,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
        )
        if grpo_metrics is not None:
            grpo_metrics["label"] = "sft_grpo_model"

    return raw_metrics, sft_metrics, grpo_metrics


def _resolve_eval_report_paths(path_value: Path | None) -> tuple[Path, Path]:
    if path_value is None:
        return DEFAULT_MODEL_EVAL_JSON, DEFAULT_MODEL_EVAL_MD
    json_path = resolve_repo_path(path_value)
    md_path = json_path.with_suffix(".md")
    return json_path, md_path


def _display_repo_path(path_value: Path) -> str:
    try:
        return str(path_value.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path_value)


def _load_eval_samples(eval_split: str, max_eval_samples: int | None) -> tuple[list[dict], dict]:
    val_samples, audit = load_validation_samples_for_benchmark()
    if eval_split == "validation":
        samples = val_samples
    elif eval_split == "hard_negative":
        samples = load_hard_negative_samples()
    elif eval_split == "combined":
        samples = val_samples + load_hard_negative_samples()
    else:
        raise ValueError(f"Unsupported eval split: {eval_split}")
    if max_eval_samples is not None:
        samples = samples[: max(0, int(max_eval_samples))]
    return samples, audit


def _write_model_eval_markdown(report: dict, output_md: Path) -> None:
    model = report.get("model_metrics") or {}
    q_aware = report.get("q_aware_baseline") or {}
    gate = report.get("training_gate") or {}
    lines = [
        "# ShadowOps Model Evaluation Report",
        "",
        f"Model: `{report.get('model_name_or_path')}`",
        f"Checkpoint: `{report.get('checkpoint_path')}`",
        f"Eval split: `{report.get('eval_split')}`",
        f"Samples: {report.get('sample_count')}",
        "",
        "## Metrics",
        "",
        "| Metric | Model | Q-aware | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in (
        "exact_match",
        "safety_accuracy",
        "unsafe_decision_rate",
        "false_positive_rate",
        "reward_mean",
        "invalid_output_rate",
    ):
        model_value = model.get(metric)
        q_value = q_aware.get(metric)
        delta = (report.get("delta_vs_q_aware") or {}).get(metric)
        lines.append(
            "| "
            + metric
            + " | "
            + ("n/a" if model_value is None else f"{float(model_value):.3f}")
            + " | "
            + ("n/a" if q_value is None else f"{float(q_value):.3f}")
            + " | "
            + ("n/a" if delta is None else f"{float(delta):+.3f}")
            + " |"
        )
    lines.extend(
        [
            "",
            "## Training Gate",
            "",
            f"Status: **{gate.get('training_gate_status', 'UNKNOWN')}**",
            "",
            f"Reason: {gate.get('reason', 'n/a')}",
            "",
            f"Recommended next action: {gate.get('recommended_next_action', 'n/a')}",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def _write_model_eval_report(report: dict, output_json: Path, output_md: Path) -> None:
    write_json(output_json, report)
    _write_model_eval_markdown(report, output_md)
    try:
        json_label = output_json.relative_to(ROOT_DIR)
    except ValueError:
        json_label = output_json
    try:
        md_label = output_md.relative_to(ROOT_DIR)
    except ValueError:
        md_label = output_md
    print(f"Saved: {json_label}")
    print(f"Saved: {md_label}")


def run_model_evaluation(args: argparse.Namespace) -> int:
    samples, audit = _load_eval_samples(args.eval_split, args.max_eval_samples)
    output_json, output_md = _resolve_eval_report_paths(args.eval_report_path)
    q_aware_metrics = evaluate_policy_on_dataset(samples, "q_aware", seed=args.seed)

    if args.model_path is None:
        candidate_path = args.output_dir
        model_load_target = str(candidate_path)
        model_path_text = _display_repo_path(candidate_path)
        model_path_exists = candidate_path.exists()
    else:
        candidate_path = resolve_repo_path(args.model_path)
        model_load_target = str(candidate_path)
        model_path_text = str(args.model_path) if not args.model_path.is_absolute() else _display_repo_path(args.model_path)
        model_path_exists = candidate_path.exists()
    model_metrics = None
    error = None

    if args.skip_model_load:
        error = "Model loading skipped by --skip-model-load."
    else:
        if args.model_path is not None and not model_path_exists:
            error = f"Model path does not exist: {model_path_text}"
        else:
            try:
                model_metrics = evaluate_saved_model(
                    model_path_or_name=model_load_target,
                    val_samples=samples,
                    max_seq_len=args.max_seq_len,
                    batch_size=args.eval_batch_size,
                    max_new_tokens=args.max_new_tokens,
                )
                if model_metrics is None:
                    error = "Model stack unavailable. Check torch, datasets, transformers, trl, unsloth, CUDA, and checkpoint path."
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

    if model_metrics is not None:
        model_metrics["model_name_or_path"] = model_path_text
        model_metrics["checkpoint_path"] = model_path_text

    gate = evaluate_training_gate(model_metrics, q_aware_metrics)
    delta = metric_delta(model_metrics, q_aware_metrics) if model_metrics is not None else None
    report = {
        "model_name_or_path": model_path_text,
        "checkpoint_path": model_path_text,
        "eval_split": args.eval_split,
        "sample_count": len(samples),
        "dataset_audit": {
            "train_sample_count": audit.get("train_sample_count"),
            "val_sample_count": audit.get("val_sample_count"),
            "train_val_overlap_count": audit.get("train_val_overlap_count"),
        },
        "model_metrics": compact_metrics(model_metrics),
        "q_aware_baseline": compact_metrics(q_aware_metrics),
        "delta_vs_q_aware": delta,
        "training_gate": gate,
        "training_gate_status": gate["training_gate_status"],
        "training_gate_passed": gate["training_gate_passed"],
        "reason": gate["reason"],
        "model_load_error": error,
    }
    _write_model_eval_report(report, output_json, output_md)
    if args.compare_against_policy:
        print("Delta vs Q-aware:")
        for metric, value in (delta or {}).items():
            print(f"  {metric}: {value:+.3f}")
    if error and not args.skip_model_load:
        print(f"Model evaluation failed: {error}")
        return 1
    print(f"Training gate: {gate['training_gate_status']} - {gate['reason']}")
    return 0


def run_grpo_training(
    *,
    args: argparse.Namespace,
    train_samples: list[dict],
    eval_samples: list[dict],
) -> tuple[dict, dict | None]:
    training_stack = load_training_stack("grpo")
    if training_stack is None:
        raise RuntimeError("Could not load the GRPO training stack.")

    torch_module = training_stack["torch"]
    configure_torch_runtime(torch_module)
    Dataset = training_stack["Dataset"]
    TrainerConfig = training_stack["TrainerConfig"]
    TrainerClass = training_stack["TrainerClass"]
    FastModel = training_stack["FastModel"]
    transformers_module = training_stack["transformers"]

    model_source = str(args.resume_from_sft)
    print(f"Loading GRPO start checkpoint from {model_source}")
    model, tokenizer = load_fast_model(FastModel, model_source, max_seq_len=args.max_seq_len)
    model = ensure_trainable_lora(
        FastModel,
        model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    with contextlib.suppress(Exception):
        FastModel.for_training(model)

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_new_tokens = args.max_new_tokens
        model.generation_config.do_sample = True
        model.generation_config.temperature = args.temperature
        model.generation_config.top_p = args.top_p
        model.generation_config.top_k = args.top_k
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    tracker = RewardHealthTracker(num_generations=args.num_generations)
    reward_fn = make_reward_function(tracker)
    prepared_samples = []
    for sample in train_samples:
        prepared = dict(sample)
        prepared["prompt"] = format_prompt_for_model(tokenizer, sample["prompt"])
        prepared["query"] = prepared["prompt"]
        prepared_samples.append(prepared)
    grpo_dataset = Dataset.from_list(prepared_samples)

    trainer_config_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.eval_steps,
        "save_total_limit": 2,
        "save_safetensors": True,
        "report_to": "none",
        "seed": args.seed,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_new_tokens,
        "max_prompt_length": args.max_seq_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "optim": "paged_adamw_8bit",
        "warmup_ratio": 0.05,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "disable_tqdm": False,
        "use_vllm": False,
        "fp16": not torch_module.cuda.is_bf16_supported(),
        "bf16": torch_module.cuda.is_bf16_supported(),
    }
    trainer_config_kwargs, dropped_config = select_supported_kwargs(TrainerConfig.__init__, trainer_config_kwargs)
    if dropped_config:
        print("Skipping unsupported GRPOConfig args:", ", ".join(dropped_config))
    trainer_config = TrainerConfig(**trainer_config_kwargs)

    trainer_init_signature = inspect.signature(TrainerClass.__init__)
    trainer_param_names = set(trainer_init_signature.parameters)
    trainer_kwargs = {
        "model": model,
        "train_dataset": grpo_dataset,
        "callbacks": [make_health_callback(tracker, transformers_module)],
    }
    if "reward_funcs" in trainer_param_names:
        trainer_kwargs["reward_funcs"] = reward_fn
    elif "reward_fn" in trainer_param_names:
        trainer_kwargs["reward_fn"] = reward_fn
    else:
        raise RuntimeError("Installed GRPOTrainer does not expose a reward function parameter.")
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
        print("Skipping unsupported GRPOTrainer args:", ", ".join(dropped_trainer))

    before_snapshot = capture_trainable_snapshot(model)
    # Fix for Unsloth + PEFT: GRPOTrainer expects warnings_issued dict
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.warnings_issued["estimate_tokens"] = True
    trainer = TrainerClass(**trainer_kwargs)

    pre_metrics = evaluate_model_on_dataset(
        model,
        tokenizer,
        eval_samples,
        batch_size=args.eval_batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
    )
    pre_metrics["label"] = "pre_grpo_sft_model"
    print(
        "Pre-GRPO metrics:",
        f"exact={pre_metrics['exact_match']:.2%}",
        f"safety={pre_metrics['safety_accuracy']:.2%}",
        f"valid={pre_metrics['valid_action_rate']:.2%}",
    )

    print("Starting GRPO training...")
    trainer.train()

    with contextlib.suppress(Exception):
        FastModel.for_inference(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    post_metrics = evaluate_model_on_dataset(
        model,
        tokenizer,
        eval_samples,
        batch_size=args.eval_batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
    )
    post_metrics["label"] = "sft_grpo_model"
    print(
        "Post-GRPO metrics:",
        f"exact={post_metrics['exact_match']:.2%}",
        f"safety={post_metrics['safety_accuracy']:.2%}",
        f"valid={post_metrics['valid_action_rate']:.2%}",
    )

    lora_delta = compute_parameter_delta(before_snapshot, model)
    best_checkpoint_path = pick_best_checkpoint(
        model_name=args.model_name,
        output_dir=args.output_dir,
        val_samples=eval_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    training_results = {
        "pre_grpo_metrics": compact_metrics(pre_metrics),
        "grpo_metrics": compact_metrics(post_metrics),
        "lora_parameter_delta": lora_delta,
        "best_checkpoint_path": best_checkpoint_path,
        "reward_std_zero_fraction": tracker.reward_std_zero_fraction,
        "grad_norm_zero_fraction": tracker.grad_norm_zero_fraction,
        "invalid_output_rate": tracker.invalid_output_rate,
        "action_distribution": tracker.action_distribution,
        "entropy": tracker.entropy,
    }
    return training_results, tracker


def run_pipeline(args: argparse.Namespace) -> int:
    ensure_dirs()
    args.resume_from_sft = resolve_repo_path(args.resume_from_sft)
    args.sft_output_dir = resolve_repo_path(args.sft_output_dir)
    args.output_dir = resolve_repo_path(args.output_dir)

    if args.reward_diagnostics:
        val_samples, _ = load_validation_samples_for_benchmark()
        diagnostics = run_reward_diagnostics(val_samples)
        print_reward_diagnostics(diagnostics)
        return 0

    if args.evaluate_model:
        return run_model_evaluation(args)

    if args.evaluate_baselines_only:
        run_demo_benchmark(
            output_json=DEFAULT_DEMO_BENCHMARK_JSON,
            output_md=DEFAULT_DEMO_BENCHMARK_MD,
        )
        run_model_policy_comparison(
            output_json=DEFAULT_MODEL_POLICY_COMPARISON_JSON,
            output_md=DEFAULT_MODEL_POLICY_COMPARISON_MD,
        )
        return 0

    args.val_size = max(args.val_size, args.val_eval_eps)
    args.val_eval_eps = max(50, args.val_eval_eps)

    if args.smoke_test:
        smoke = run_logic_smoke_test()
        print("GRPO smoke test:", "PASS" if smoke["smoke_test_passed"] else "FAIL")
        print("Final report: training/final_training_report.json")
        return 0 if smoke["smoke_test_passed"] else 1

    train_samples, val_samples, dataset_audit = generate_datasets(
        train_size=args.train_size,
        val_size=args.val_size,
        save=True,
    )

    try:
        preflight_dataset_check(dataset_audit)
    except TrainingPreflightError as exc:
        print(f"Dataset preflight failed: {exc}")
        return 1

    reward_variance = check_reward_variance(train_samples[: min(len(train_samples), 160)])
    if not reward_variance["passed"]:
        print("Reward variance preflight failed:", reward_variance)
        return 1

    eval_samples = val_samples[: min(len(val_samples), args.val_eval_eps)]
    parse_tests = run_parse_action_tests()
    prior_report = read_json(ROOT_DIR / "training" / "final_training_report.json", default={}) or {}
    prior_smoke_pass = prior_report.get("training_ready_criteria", {}).get("smoke_test_passes", False)

    maybe_run_sft(args)

    if args.skip_grpo:
        raw_metrics = None
        sft_metrics = None
        if not args.skip_model_load and not args.dry_run and args.resume_from_sft.exists():
            if validate_training_runtime(args.model_name):
                raw_metrics, sft_metrics, _ = evaluate_all_variants(
                    model_name=args.model_name,
                    resume_from_sft=args.resume_from_sft,
                    grpo_output_dir=args.output_dir,
                    eval_samples=eval_samples,
                    max_seq_len=args.max_seq_len,
                    eval_batch_size=args.eval_batch_size,
                    max_new_tokens=args.max_new_tokens,
                )
        metrics_by_label, oracle_check = build_evaluation_bundle(
            eval_samples,
            raw_model_metrics=raw_metrics,
            sft_metrics=sft_metrics,
            grpo_metrics=None,
        )
        health_report = build_training_health_report(
            pre_train_metrics=raw_metrics,
            sft_metrics=sft_metrics,
            grpo_metrics=None,
            tracker=None,
            baseline_metrics={label: metrics_by_label[label] for label in ("random", "heuristic", "q_aware")},
            oracle_metrics=metrics_by_label["oracle"],
            lora_parameter_delta=None,
            oracle_check=oracle_check,
        )
        write_json(DEFAULT_HEALTH_REPORT_PATH, health_report)
        final_report = generate_final_reports(
            dataset_audit=dataset_audit,
            parse_tests=parse_tests,
            reward_variance=reward_variance,
            oracle_check=oracle_check,
            metrics_by_label=metrics_by_label,
            training_health_report=health_report,
            smoke_test_result=bool(prior_smoke_pass),
            best_checkpoint_path=None,
        )
        print(final_report["status"]["final_status"])
        return 0

    if args.dry_run or args.skip_model_load:
        metrics_by_label, oracle_check = build_evaluation_bundle(eval_samples)
        health_report = build_training_health_report(
            pre_train_metrics=None,
            sft_metrics=None,
            grpo_metrics=None,
            tracker=None,
            baseline_metrics={label: metrics_by_label[label] for label in ("random", "heuristic", "q_aware")},
            oracle_metrics=metrics_by_label["oracle"],
            lora_parameter_delta=None,
            oracle_check=oracle_check,
        )
        write_json(DEFAULT_HEALTH_REPORT_PATH, health_report)
        final_report = generate_final_reports(
            dataset_audit=dataset_audit,
            parse_tests=parse_tests,
            reward_variance=reward_variance,
            oracle_check=oracle_check,
            metrics_by_label=metrics_by_label,
            training_health_report=health_report,
            smoke_test_result=bool(prior_smoke_pass),
            best_checkpoint_path=None,
        )
        print("Dry-run complete.")
        print(final_report["status"]["final_status"])
        return 0

    if not args.resume_from_sft.exists():
        print(f"SFT adapter not found: {args.resume_from_sft}")
        print("Run SFT first or pass --run-sft.")
        return 1

    if not validate_training_runtime(args.model_name):
        return 1

    vram_gb = get_total_gpu_memory_gb()
    if vram_gb is not None and vram_gb < 8:
        print(f"Detected {vram_gb:.1f} GB VRAM. Fallback GRPO command if needed:")
        print(CLOUD_FALLBACK_COMMAND)

    configure_runtime_noise_filters()

    raw_metrics, sft_metrics, _ = evaluate_all_variants(
        model_name=args.model_name,
        resume_from_sft=args.resume_from_sft,
        grpo_output_dir=args.output_dir,
        eval_samples=eval_samples,
        max_seq_len=args.max_seq_len,
        eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    training_results, tracker = run_grpo_training(
        args=args,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    _, _, grpo_metrics = evaluate_all_variants(
        model_name=args.model_name,
        resume_from_sft=args.resume_from_sft,
        grpo_output_dir=args.output_dir,
        eval_samples=eval_samples,
        max_seq_len=args.max_seq_len,
        eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    metrics_by_label, oracle_check = build_evaluation_bundle(
        eval_samples,
        raw_model_metrics=raw_metrics,
        sft_metrics=sft_metrics,
        grpo_metrics=grpo_metrics,
    )

    health_report = build_training_health_report(
        pre_train_metrics=raw_metrics,
        sft_metrics=sft_metrics,
        grpo_metrics=grpo_metrics,
        tracker=tracker,
        baseline_metrics={label: metrics_by_label[label] for label in ("random", "heuristic", "q_aware")},
        oracle_metrics=metrics_by_label["oracle"],
        lora_parameter_delta=training_results["lora_parameter_delta"],
        oracle_check=oracle_check,
    )
    write_json(DEFAULT_HEALTH_REPORT_PATH, health_report)

    final_report = generate_final_reports(
        dataset_audit=dataset_audit,
        parse_tests=parse_tests,
        reward_variance=reward_variance,
        oracle_check=oracle_check,
        metrics_by_label=metrics_by_label,
        training_health_report=health_report,
        smoke_test_result=bool(prior_smoke_pass),
        best_checkpoint_path=training_results["best_checkpoint_path"],
    )

    print()
    print("Cloud SFT command:")
    print(CLOUD_SFT_COMMAND)
    print("Cloud GRPO command:")
    print(CLOUD_GRPO_COMMAND)
    print(final_report["status"]["final_status"])
    return 0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(run_pipeline(args))


if __name__ == "__main__":
    main()
