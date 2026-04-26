"""Compatibility wrapper for real-artifact ShadowOps plotting.

This module intentionally avoids demo targets or pasted summary values. It
delegates plot creation to ``generate_reward_curves.py`` and writes a compact
evaluation summary from current repository JSON reports only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
PLOTS_DIR = TRAINING_DIR / "plots"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.generate_reward_curves import TRAINING_CURVE_WARNING, generate_reward_curves  # noqa: E402


PENDING = "PENDING_REAL_ARTIFACT"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_rows() -> list[dict[str, Any]]:
    report = _read_json(TRAINING_DIR / "demo_benchmark_report.json")
    if not isinstance(report, dict):
        return []
    return [row for row in report.get("metrics", []) if isinstance(row, dict)]


def write_evaluation_summary() -> dict[str, Any]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _benchmark_rows()
    summary = {
        "metrics_source": "training/demo_benchmark_report.json" if rows else PENDING,
        "metrics_note": "Current values are loaded only from real repository report artifacts.",
        "training_curve_note": TRAINING_CURVE_WARNING,
        "rows": rows,
        "missing_model_metrics": PENDING,
    }
    (PLOTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "## ShadowOps Evaluation Summary",
        "",
        "> Metrics below are loaded only from real repository artifacts.",
        "",
        f"> {TRAINING_CURVE_WARNING}",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row.get('policy')} | {float(row.get('exact_match', 0.0)):.3f} | "
                f"{float(row.get('safety_accuracy', 0.0)):.3f} | "
                f"{float(row.get('unsafe_decision_rate', 0.0)):.3f} | "
                f"{float(row.get('false_positive_rate', 0.0)):.3f} | "
                f"{float(row.get('reward_mean', 0.0)):.3f} |"
            )
    else:
        lines.append(f"Benchmark metrics: {PENDING}")
    lines.extend(
        [
            "",
            "### Pending Model Metrics",
            "",
            f"- trained checkpoint metrics: {PENDING}",
        ]
    )
    (PLOTS_DIR / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    generate_reward_curves()
    write_evaluation_summary()
    print("Generated plots and evaluation summary from real artifacts only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
