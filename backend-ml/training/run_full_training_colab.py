"""
Cloud orchestration helper for full ShadowOps training.

Run this from backend-ml on Colab, Kaggle, or another GPU machine:

python training/run_full_training_colab.py

Use --print-only to inspect the planned commands without executing them.
Use --oom-fallback to switch the GRPO stage to the lower-memory fallback.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def build_steps(use_oom_fallback: bool) -> list[tuple[str, list[str]]]:
    grpo_command = [
        sys.executable,
        "training/train_qwen3_grpo.py",
        "--model-name",
        "unsloth/Qwen3-1.7B",
        "--resume-from-sft",
        "training/checkpoints/qwen3_sft_adapter",
        "--max-steps",
        "800",
        "--num-generations",
        "6" if use_oom_fallback else "8",
        "--temperature",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "50",
        "--max-new-tokens",
        "8",
        "--batch-size",
        "1",
        "--grad-accum",
        "8" if use_oom_fallback else "4",
        "--val-eval-eps",
        "50" if use_oom_fallback else "100",
        "--eval-batch-size",
        "4",
        "--learning-rate",
        "1e-5",
        "--output-dir",
        "training/checkpoints/qwen3_sft_grpo_adapter",
    ]

    return [
        ("Compile checks", [sys.executable, "-m", "py_compile", "training/train_qwen3_grpo.py"]),
        ("Compile checks", [sys.executable, "-m", "py_compile", "training/train_qwen3_sft.py"]),
        ("Compile checks", [sys.executable, "-m", "py_compile", "shadowops_env.py"]),
        ("Smoke tests", [sys.executable, "training/train_qwen3_grpo.py", "--smoke-test", "--skip-model-load"]),
        ("Smoke tests", [sys.executable, "training/train_qwen3_sft.py", "--smoke-test", "--skip-model-load"]),
        ("Dataset audit", [sys.executable, "training/train_qwen3_grpo.py", "--dry-run", "--skip-model-load"]),
        (
            "SFT training + validation",
            [
                sys.executable,
                "training/train_qwen3_sft.py",
                "--model-name",
                "unsloth/Qwen3-1.7B",
                "--sft-epochs",
                "2",
                "--batch-size",
                "1",
                "--grad-accum",
                "8",
                "--max-seq-len",
                "256",
                "--learning-rate",
                "2e-4",
                "--sft-output-dir",
                "training/checkpoints/qwen3_sft_adapter",
            ],
        ),
        ("GRPO training + final validation + final report", grpo_command),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ShadowOps cloud training sequence.")
    parser.add_argument("--print-only", action="store_true", help="Print the commands without executing them.")
    parser.add_argument("--oom-fallback", action="store_true", help="Use the lower-memory GRPO fallback.")
    args = parser.parse_args()

    steps = build_steps(use_oom_fallback=args.oom_fallback)
    for index, (stage, command) in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {stage}")
        print("  " + " ".join(command))
        if args.print_only:
            continue
        completed = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
