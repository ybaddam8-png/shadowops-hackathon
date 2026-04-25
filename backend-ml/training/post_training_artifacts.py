"""Generate post-training artifacts safely with pending-aware behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
PLOTS_DIR = TRAINING_DIR / "plots"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.compare_training_runs import generate_run_comparison
from training.generate_reward_curves import collect_real_curve_series, generate_reward_curves


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exists_non_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _collect_real_eval_evidence() -> dict[str, Any]:
    candidates = [
        REPORTS_DIR / "model_eval_report.json",
        TRAINING_DIR / "model_eval_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = _read_json(candidate)
            return {
                "status": "FOUND",
                "source": str(candidate.relative_to(BACKEND_DIR)),
                "training_gate_status": payload.get("training_gate_status"),
                "delta_vs_q_aware": payload.get("delta_vs_q_aware"),
            }
    return {"status": "PENDING_REAL_TRAINING_LOGS"}


def _run_submission_checker() -> dict[str, Any]:
    # Import lazily to keep this script lightweight when called standalone.
    from training.check_submission_artifacts import check_submission_artifacts

    return check_submission_artifacts()


def generate_post_training_artifacts() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    curve_series, curve_notes = collect_real_curve_series()
    curve_report = generate_reward_curves()
    run_comparison = generate_run_comparison(PLOTS_DIR)
    model_eval = _collect_real_eval_evidence()
    submission_check = _run_submission_checker()

    if curve_series is None:
        status = "PENDING_REAL_TRAINING_LOGS"
    elif submission_check.get("status") == "FAIL":
        status = "FAIL"
    elif submission_check.get("status") == "WARN":
        status = "WARN"
    else:
        status = "PASS"

    report = {
        "status": status,
        "curve_notes": curve_notes,
        "reward_curve_status": curve_report.get("status"),
        "run_comparison_status": run_comparison.get("status"),
        "model_eval_status": model_eval.get("status"),
        "model_eval": model_eval,
        "submission_checker_status": submission_check.get("status"),
        "artifacts": {
            "reward_curve_data": "training/reports/reward_curve_data.json",
            "reward_curve_report": "training/reports/reward_curve_report.md",
            "run_comparison_report": "training/plots/run_comparison_report.json",
            "submission_checker_report": "training/reports/submission_artifact_report.json",
            "plots_non_empty": {
                "reward_curve.png": _exists_non_empty(PLOTS_DIR / "reward_curve.png"),
                "reward_std_curve.png": _exists_non_empty(PLOTS_DIR / "reward_std_curve.png"),
                "completion_length_curve.png": _exists_non_empty(PLOTS_DIR / "completion_length_curve.png"),
                "invalid_output_rate_curve.png": _exists_non_empty(PLOTS_DIR / "invalid_output_rate_curve.png"),
                "model_policy_comparison.png": _exists_non_empty(PLOTS_DIR / "model_policy_comparison.png"),
                "safety_reward_comparison.png": _exists_non_empty(PLOTS_DIR / "safety_reward_comparison.png"),
            },
        },
        "never_faked_metrics": True,
    }

    report_json = REPORTS_DIR / "post_training_artifacts_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md = REPORTS_DIR / "post_training_artifacts_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Post-Training Artifacts Report",
                "",
                f"- status: `{report['status']}`",
                f"- reward curves: `{report['reward_curve_status']}`",
                f"- run comparison: `{report['run_comparison_status']}`",
                f"- model evaluation evidence: `{report['model_eval_status']}`",
                f"- submission checker: `{report['submission_checker_status']}`",
                "",
                "If real trainer logs/checkpoints are missing, status remains `PENDING_REAL_TRAINING_LOGS`.",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    report = generate_post_training_artifacts()
    print(f"post_training_artifacts_status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
