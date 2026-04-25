"""
One-command ShadowOps hackathon judge demo.

Run from backend-ml:
    python demo_judge.py
"""

from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

COMPILE_TARGETS = (
    "main.py",
    "shadowops_env.py",
    "shadowops_cli.py",
    "agent_memory.py",
    "risk_accumulator.py",
    "evidence_planner.py",
    "domain_policies.py",
    "safe_outcome.py",
    "training/shadowops_training_common.py",
    "training/train_qwen3_grpo.py",
    "demo_replays.py",
    "demo_judge.py",
)


def compile_checks() -> None:
    print("[1/5] Compile checks")
    for relative_path in COMPILE_TARGETS:
        target = ROOT_DIR / relative_path
        py_compile.compile(str(target), doraise=True)
        print(f"  OK {relative_path}")


def dataset_audit_summary() -> None:
    print("\n[2/5] Dataset audit summary")
    audit_path = ROOT_DIR / "training" / "dataset_audit.json"
    train_path = ROOT_DIR / "training" / "qwen3_train_dataset.json"
    val_path = ROOT_DIR / "training" / "qwen3_val_dataset.json"

    if not audit_path.exists():
        raise RuntimeError("training/dataset_audit.json is missing.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train_samples = json.loads(train_path.read_text(encoding="utf-8"))
    val_samples = json.loads(val_path.read_text(encoding="utf-8"))

    expected_train = audit["train_sample_count"]
    expected_val = audit["val_sample_count"]
    if len(train_samples) != expected_train or len(val_samples) != expected_val:
        raise RuntimeError(
            "Dataset count mismatch: "
            f"train file={len(train_samples)} audit={expected_train}; "
            f"val file={len(val_samples)} audit={expected_val}"
        )

    print(f"  train samples: {expected_train}")
    print(f"  validation samples: {expected_val}")
    print(f"  train/val overlap: {audit['train_val_overlap_count']}")
    print(f"  duplicate prompts: {audit['duplicate_prompt_count']}")
    print(f"  preflight: {'PASS' if audit.get('passed_preflight') else 'FAIL'}")
    print(f"  validation action distribution: {audit['action_distribution']['val']}")


def benchmark_only_evaluation() -> None:
    print("\n[3/5] Benchmark-only evaluation")
    from training.shadowops_training_common import run_demo_benchmark

    run_demo_benchmark()


def replay_scenarios() -> None:
    print("\n[4/5] Replay attack scenarios")
    from demo_replays import build_replay_results, print_replays

    results = build_replay_results()
    print_replays(results)
    failed = [row for row in results if not row["outcome"].startswith("PASS")]
    if failed:
        names = ", ".join(row["name"] for row in failed)
        raise RuntimeError(f"Replay scenario(s) need review: {names}")


def dashboard_hint() -> None:
    print("\n[5/5] Optional CLI dashboard")
    print("  python shadowops_cli.py --episodes 10 --speed 0.08")


def main() -> int:
    try:
        compile_checks()
        dataset_audit_summary()
        benchmark_only_evaluation()
        replay_scenarios()
        dashboard_hint()
    except Exception as exc:
        print(f"\nDEMO JUDGE FAILED: {exc}")
        return 1

    print("\nDEMO JUDGE PASS: GPU training pending; laptop-safe Q-aware policy is verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
