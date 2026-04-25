"""Validate submission artifacts and unsafe tracked files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend-ml"
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
PLOTS_DIR = TRAINING_DIR / "plots"

REQUIRED_FILES = {
    "openenv_root_yaml": REPO_ROOT / "openenv.yaml",
    "openenv_wrapper": BACKEND_DIR / "openenv_shadowops_env.py",
    "schema_contract": BACKEND_DIR / "schema_contract.json",
    "benchmark_report": TRAINING_DIR / "demo_benchmark_report.json",
    "reward_curve_data": REPORTS_DIR / "reward_curve_data.json",
    "reward_curve_report": REPORTS_DIR / "reward_curve_report.md",
    "repo_cleanup_report_json": REPORTS_DIR / "repo_cleanup_report.json",
    "repo_cleanup_report_md": REPORTS_DIR / "repo_cleanup_report.md",
}

OPTIONAL_REPORTS = {
    "hidden_eval_report": REPORTS_DIR / "hidden_eval_report.json",
    "model_policy_comparison_report": REPORTS_DIR / "model_policy_comparison.json",
}

REQUIRED_PNGS = [
    "reward_curve.png",
    "reward_std_curve.png",
    "completion_length_curve.png",
    "invalid_output_rate_curve.png",
    "model_policy_comparison.png",
    "safety_reward_comparison.png",
]

TRACKED_BLOCK_PATTERNS = (
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".venv/",
    "venv/",
    "__pycache__/",
)

DANGEROUS_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _has_pending_real_logs() -> bool:
    reward_curve_data = REPORTS_DIR / "reward_curve_data.json"
    if not reward_curve_data.exists():
        return True
    try:
        payload = _read_json(reward_curve_data)
    except json.JSONDecodeError:
        return False
    status = str(payload.get("status", ""))
    notes = payload.get("notes") or []
    if status == "PENDING_REAL_TRAINING_LOGS":
        return True
    return any(str(note) == "PENDING_REAL_TRAINING_LOGS" for note in notes)


def check_submission_artifacts() -> dict[str, Any]:
    required_missing = [name for name, path in REQUIRED_FILES.items() if not path.exists()]
    optional = {name: path.exists() for name, path in OPTIONAL_REPORTS.items()}

    broken_pngs: list[str] = []
    for name in REQUIRED_PNGS:
        path = PLOTS_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            broken_pngs.append(str(path.relative_to(REPO_ROOT)))

    tracked = _git_ls_files()
    tracked_unsafe = []
    for path in tracked:
        abs_path = REPO_ROOT / path
        # If the file is already deleted from the working tree, do not fail the in-flight cleanup.
        if not abs_path.exists():
            continue
        if (
            any(fragment in path for fragment in TRACKED_BLOCK_PATTERNS)
            or path.endswith(DANGEROUS_SUFFIXES)
            or path.endswith(".env")
            or ".env." in path
        ):
            tracked_unsafe.append(path)

    fake_claims: list[str] = []
    model_policy_path = OPTIONAL_REPORTS["model_policy_comparison_report"]
    if model_policy_path.exists():
        payload = _read_json(model_policy_path)
        delta = payload.get("delta_vs_q_aware", {})
        reward_delta = delta.get("reward_mean") if isinstance(delta, dict) else None
        if reward_delta is not None and not isinstance(reward_delta, (int, float)):
            fake_claims.append("model_policy_comparison.delta_vs_q_aware.reward_mean must be numeric or absent.")

    if required_missing or broken_pngs or tracked_unsafe or fake_claims:
        status = "FAIL"
    elif _has_pending_real_logs():
        status = "WARN"
    else:
        status = "PASS"

    report = {
        "status": status,
        "required_missing": required_missing,
        "optional_reports": optional,
        "broken_reward_curve_pngs": broken_pngs,
        "tracked_unsafe_paths": tracked_unsafe,
        "fake_model_improvement_claims": fake_claims,
        "pending_real_training_logs": _has_pending_real_logs(),
    }
    json_path = REPORTS_DIR / "submission_artifact_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_lines = [
        "# Submission Artifact Check",
        "",
        f"- status: `{status}`",
        f"- pending_real_training_logs: `{report['pending_real_training_logs']}`",
        "",
        "## Missing Required Files",
        *(["- none"] if not required_missing else [f"- `{name}`" for name in required_missing]),
        "",
        "## Broken Reward Curve PNGs",
        *(["- none"] if not broken_pngs else [f"- `{name}`" for name in broken_pngs]),
        "",
        "## Tracked Unsafe Paths",
        *(["- none"] if not tracked_unsafe else [f"- `{path}`" for path in tracked_unsafe]),
        "",
        "## Fake Model Improvement Claims",
        *(["- none"] if not fake_claims else [f"- {item}" for item in fake_claims]),
    ]
    (REPORTS_DIR / "submission_artifact_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report


def main() -> int:
    report = check_submission_artifacts()
    print(f"submission_artifact_status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
