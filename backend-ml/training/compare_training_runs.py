"""Compare available training/evaluation artifacts without inventing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"

ARTIFACT_FILES = {
    "trainer_state": TRAINING_DIR / "trainer_state.json",
    "metrics_jsonl": TRAINING_DIR / "metrics.jsonl",
    "reward_curve_data": REPORTS_DIR / "reward_curve_data.json",
    "demo_benchmark_report": TRAINING_DIR / "demo_benchmark_report.json",
    "hidden_eval_report": REPORTS_DIR / "hidden_eval_report.json",
    "reward_diagnostics_report": TRAINING_DIR / "reward_diagnostics_report.json",
    "model_eval_report_training": TRAINING_DIR / "model_eval_report.json",
    "model_eval_report_reports": REPORTS_DIR / "model_eval_report.json",
    "model_policy_comparison_training": TRAINING_DIR / "model_policy_comparison.json",
    "model_policy_comparison_reports": REPORTS_DIR / "model_policy_comparison.json",
    "checkpoint_comparison_report": REPORTS_DIR / "checkpoint_comparison_report.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_status() -> tuple[str, list[str], dict[str, str]]:
    notes: list[str] = []
    available: dict[str, str] = {}
    for name, path in ARTIFACT_FILES.items():
        if path.exists():
            available[name] = str(path.relative_to(BACKEND_DIR))
    has_real_logs = "trainer_state" in available or "metrics_jsonl" in available
    status = "PASS" if has_real_logs else "PENDING_REAL_TRAINING_LOGS"
    if not has_real_logs:
        notes.append("PENDING_REAL_TRAINING_LOGS")
    return status, notes, available


def _extract_benchmark_fields() -> dict[str, Any]:
    bench_path = ARTIFACT_FILES["demo_benchmark_report"]
    if not bench_path.exists():
        return {"available": False}
    bench = _read_json(bench_path)
    rows = bench.get("metrics")
    if not isinstance(rows, list):
        return {"available": True, "rows": 0}
    qaware = next((row for row in rows if isinstance(row, dict) and row.get("policy") == "q_aware"), None)
    return {
        "available": True,
        "rows": len(rows),
        "q_aware_reward_mean": _safe_float((qaware or {}).get("reward_mean")),
        "q_aware_safety_accuracy": _safe_float((qaware or {}).get("safety_accuracy")),
    }


def _extract_model_policy() -> dict[str, Any]:
    path = _resolve_first([ARTIFACT_FILES["model_policy_comparison_reports"], ARTIFACT_FILES["model_policy_comparison_training"]])
    if path is None:
        return {"available": False, "status": "PENDING_REAL_TRAINING_LOGS"}
    payload = _read_json(path)
    return {
        "available": True,
        "source": str(path.relative_to(BACKEND_DIR)),
        "model_reward_mean": _safe_float(payload.get("model", {}).get("reward_mean")),
        "q_aware_reward_mean": _safe_float(payload.get("q_aware", {}).get("reward_mean")),
        "delta_vs_q_aware": _safe_float(payload.get("delta_vs_q_aware", {}).get("reward_mean")),
    }


def _extract_hidden_eval() -> dict[str, Any]:
    path = ARTIFACT_FILES["hidden_eval_report"]
    if not path.exists():
        return {"available": False}
    payload = _read_json(path)
    return {
        "available": True,
        "source": str(path.relative_to(BACKEND_DIR)),
        "status": payload.get("status", "UNKNOWN"),
        "unsafe_decision_rate": _safe_float(payload.get("unsafe_decision_rate")),
    }


def generate_run_comparison(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status, notes, available = _extract_status()
    report = {
        "status": status,
        "notes": notes,
        "available_artifacts": available,
        "benchmark": _extract_benchmark_fields(),
        "hidden_eval": _extract_hidden_eval(),
        "model_policy": _extract_model_policy(),
    }
    output_path = output_dir / "run_comparison_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare available run artifacts")
    parser.add_argument("--output-dir", default="training/plots", help="Output directory relative to backend-ml")
    args = parser.parse_args()
    output_dir = (BACKEND_DIR / args.output_dir).resolve()
    report = generate_run_comparison(output_dir)
    print(f"run_comparison_status={report['status']}")
    print(f"run_comparison_report={output_dir / 'run_comparison_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
