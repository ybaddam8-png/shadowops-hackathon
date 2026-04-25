"""Check that ShadowOps submission artifacts are present and honest."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend-ml"
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
OUTPUT_JSON = REPORTS_DIR / "submission_artifact_check.json"
OUTPUT_MD = REPORTS_DIR / "submission_artifact_check.md"


REQUIRED_FILES = {
    "root_openenv": REPO_ROOT / "openenv.yaml",
    "backend_openenv_wrapper": BACKEND_DIR / "openenv_shadowops_env.py",
    "backend_openenv_config": BACKEND_DIR / "openenv.yaml",
    "schema_contract": BACKEND_DIR / "schema_contract.json",
    "benchmark_json": TRAINING_DIR / "demo_benchmark_report.json",
    "benchmark_md": TRAINING_DIR / "demo_benchmark_report.md",
    "hidden_eval_json": REPORTS_DIR / "hidden_eval_report.json",
    "hidden_eval_md": REPORTS_DIR / "hidden_eval_report.md",
    "multistep_json": REPORTS_DIR / "multistep_episode_report.json",
    "multistep_md": REPORTS_DIR / "multistep_episode_report.md",
    "reward_diagnostics_json": REPORTS_DIR / "reward_diagnostics_report.json",
    "reward_diagnostics_md": REPORTS_DIR / "reward_diagnostics_report.md",
    "reward_curve_data": REPORTS_DIR / "reward_curve_data.json",
    "reward_curve_report": REPORTS_DIR / "reward_curve_report.md",
    "action_only_sft_audit_json": REPORTS_DIR / "action_only_sft_dataset_audit.json",
    "action_only_sft_audit_md": REPORTS_DIR / "action_only_sft_dataset_audit.md",
    "model_policy_gate_json": REPORTS_DIR / "model_policy_gate_report.json",
    "model_policy_gate_md": REPORTS_DIR / "model_policy_gate_report.md",
    "submission_readme": REPO_ROOT / "docs" / "SUBMISSION_README_DRAFT.md",
    "judge_readme_block": REPO_ROOT / "docs" / "JUDGE_README_BLOCK.md",
    "pitch_script": REPO_ROOT / "docs" / "DEMO_PITCH_SCRIPT.md",
    "judge_checklist": REPO_ROOT / "docs" / "JUDGE_DEMO_CHECKLIST.md",
    "champion_checkpoint_guide": REPO_ROOT / "docs" / "CHAMPION_CHECKPOINT_INTEGRATION.md",
    "kaggle_cells": REPO_ROOT / "docs" / "KAGGLE_TRAINING_CELLS.md",
}

REQUIRED_PLOTS = (
    TRAINING_DIR / "plots" / "baseline_policy_comparison.png",
    TRAINING_DIR / "plots" / "reward_breakdown.png",
    TRAINING_DIR / "plots" / "safety_accuracy_comparison.png",
    TRAINING_DIR / "plots" / "unsafe_decision_rate.png",
    TRAINING_DIR / "plots" / "model_policy_comparison.png",
    TRAINING_DIR / "plots" / "safety_reward_comparison.png",
)

SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)

FAKE_CLAIM_PATTERNS = (
    re.compile(r"\btrained model (improved|beats|wins)\b", re.IGNORECASE),
    re.compile(r"\bSFT.*GRPO.*improved\b", re.IGNORECASE),
    re.compile(r"\bcheckpoint wins\b", re.IGNORECASE),
)


def _git_ls_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def _scan_text_files(paths: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    secrets = []
    fake_claims = []
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".md", ".ps1", ".sh", ".py", ".json", ".yaml", ".yml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(REPO_ROOT))
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                secrets.append({"file": rel, "pattern": pattern.pattern})
        for pattern in FAKE_CLAIM_PATTERNS:
            if pattern.search(text):
                fake_claims.append({"file": rel, "pattern": pattern.pattern})
    return secrets, fake_claims


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_submission_artifact_check() -> dict[str, Any]:
    tracked = _git_ls_files()
    missing = {name: str(path.relative_to(REPO_ROOT)) for name, path in REQUIRED_FILES.items() if not path.exists()}
    missing_plots = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_PLOTS if not path.exists()]
    reward_curve_data = _load_json(REPORTS_DIR / "reward_curve_data.json") or {}
    has_training_logs = bool(reward_curve_data.get("has_real_training_logs"))
    reward_curve_plot = TRAINING_DIR / "plots" / "reward_curve.png"
    training_curve_missing = has_training_logs and not reward_curve_plot.exists()
    bad_tracked = [
        path
        for path in tracked
        if any(
            marker in path
            for marker in (
                "node_modules/",
                "__pycache__/",
                "unsloth_compiled_cache/",
                "shadowops_qwen3_",
                ".safetensors",
                ".pt",
                ".pth",
                "training/checkpoints/",
            )
        )
    ]
    nested_repo_tracked = [path for path in tracked if path.startswith("shadowops-hackathon/")]

    scan_roots = [REPO_ROOT / "docs", REPO_ROOT / "scripts", BACKEND_DIR / "training"]
    scan_files = [path for root in scan_roots if root.exists() for path in root.rglob("*") if path.is_file()]
    secrets, fake_claims = _scan_text_files(scan_files)

    gate = _load_json(REPORTS_DIR / "model_policy_gate_report.json") or {}
    checkpoint = _load_json(REPORTS_DIR / "checkpoint_comparison_report.json") or {}
    checkpoint_metrics_pending = not any(row.get("metrics") for row in checkpoint.get("checkpoints", []))
    failures = []
    warnings = []
    if missing:
        failures.append("required artifacts missing")
    if missing_plots:
        failures.append("submission plots missing")
    if training_curve_missing:
        failures.append("real training logs exist but reward_curve.png is missing")
    if secrets:
        failures.append("possible secrets or tokens found")
    if fake_claims:
        failures.append("possible unsupported trained-model claim found")
    if bad_tracked:
        failures.append("large/cache/checkpoint artifacts are tracked")
    if nested_repo_tracked:
        failures.append("nested duplicate repo folder is tracked")
    if checkpoint_metrics_pending or gate.get("gate_status") == "PENDING_REAL_CHECKPOINT_EVAL":
        warnings.append("real trained checkpoint metrics are pending")
    if not has_training_logs:
        warnings.append("real trainer_state.json logs are pending; training curves will appear after SFT/GRPO runs")

    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "missing_required_files": missing,
        "missing_plots": missing_plots,
        "training_curve_missing": training_curve_missing,
        "has_real_training_logs": has_training_logs,
        "possible_secrets": secrets,
        "possible_fake_claims": fake_claims,
        "bad_tracked_artifacts": bad_tracked[:50],
        "nested_duplicate_repo_tracked": nested_repo_tracked[:50],
        "model_policy_gate_status": gate.get("gate_status", "missing"),
        "checkpoint_metrics_pending": checkpoint_metrics_pending,
        "checked_tracked_file_count": len(tracked),
    }


def write_submission_artifact_check(report: dict[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# ShadowOps Submission Artifact Check",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {item}" for item in report["failures"] or ["none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"] or ["none"])
    lines.extend(["", "## Model Policy Gate", "", f"- status: {report['model_policy_gate_status']}"])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = build_submission_artifact_check()
    write_submission_artifact_check(report)
    print(f"Submission artifact check: {report['status']}")
    print(f"Saved: {OUTPUT_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {OUTPUT_MD.relative_to(BACKEND_DIR)}")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
