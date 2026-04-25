"""Laptop-safe latency and determinism check for ShadowOps demo policy."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from demo_replays import build_replay_results  # noqa: E402
from training.shadowops_training_common import write_json  # noqa: E402


DEFAULT_LATENCY_REPORT_JSON = TRAINING_DIR / "latency_determinism_report.json"
DEFAULT_LATENCY_REPORT_MD = TRAINING_DIR / "latency_determinism_report.md"


def run_latency_determinism_check(iterations: int = 3) -> dict[str, Any]:
    runs = []
    for _ in range(iterations):
        start = time.perf_counter()
        rows = build_replay_results()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        runs.append(
            {
                "elapsed_ms": elapsed_ms,
                "decisions": {row["name"]: row["supervisor_decision"] for row in rows},
            }
        )

    first = runs[0]["decisions"] if runs else {}
    deterministic = all(run["decisions"] == first for run in runs)
    scenario_count = len(first)
    avg_total_ms = sum(run["elapsed_ms"] for run in runs) / max(len(runs), 1)
    return {
        "iterations": iterations,
        "scenario_count": scenario_count,
        "deterministic": deterministic,
        "average_total_ms": avg_total_ms,
        "average_decision_ms": avg_total_ms / max(scenario_count, 1),
        "decisions": first,
        "runs": runs,
    }


def write_latency_report(
    report: dict[str, Any],
    *,
    output_json: Path = DEFAULT_LATENCY_REPORT_JSON,
    output_md: Path = DEFAULT_LATENCY_REPORT_MD,
) -> None:
    write_json(output_json, report)
    lines = [
        "# ShadowOps Latency And Determinism Report",
        "",
        f"- Iterations: {report['iterations']}",
        f"- Scenarios per run: {report['scenario_count']}",
        f"- Deterministic decisions: {'yes' if report['deterministic'] else 'no'}",
        f"- Average total run time: {report['average_total_ms']:.3f} ms",
        f"- Average decision time: {report['average_decision_ms']:.3f} ms",
        "",
        "## Decisions",
        "",
        "| Scenario | Decision |",
        "| --- | --- |",
    ]
    for name, decision in report["decisions"].items():
        lines.append(f"| {name} | {decision} |")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_latency_determinism_check()
    write_latency_report(report)
    print(f"Deterministic: {'yes' if report['deterministic'] else 'no'}")
    print(f"Average decision time: {report['average_decision_ms']:.3f} ms")
    print(f"Saved: {DEFAULT_LATENCY_REPORT_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_LATENCY_REPORT_MD.relative_to(BACKEND_DIR)}")
    return 0 if report["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
