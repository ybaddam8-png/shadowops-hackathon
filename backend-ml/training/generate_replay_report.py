"""Generate deterministic replay report for ShadowOps demo scenarios."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from demo_replays import build_replay_results  # noqa: E402
from training.shadowops_training_common import write_json  # noqa: E402


DEFAULT_REPLAY_REPORT_JSON = TRAINING_DIR / "replay_report.json"
DEFAULT_REPLAY_REPORT_MD = TRAINING_DIR / "replay_report.md"


def _risk_delta(row: dict[str, Any]) -> float:
    risk_summary = str(row.get("risk_vector_summary", ""))
    base = 0.0
    prefix = "risk_score="
    if prefix in risk_summary:
        try:
            base = float(risk_summary.split(prefix, 1)[1].split(";", 1)[0])
        except (ValueError, IndexError):
            base = 0.0
    return round(float(row.get("cumulative_risk_score", 0.0)) - base, 3)


def build_replay_report() -> dict[str, Any]:
    rows = []
    for row in build_replay_results():
        decision_object = row.get("decision_object", {})
        trace = decision_object.get("decision_trace", {})
        rows.append(
            {
                "scenario_name": row["name"],
                "action_chain": f"{row['domain']}::{row['intent']} -> {row['supervisor_decision']}",
                "expected_behavior": row["expected_safe_action"],
                "actual_decision": row["supervisor_decision"],
                "risk_delta": _risk_delta(row),
                "evidence_requested": row.get("missing_evidence", []),
                "evidence_steps": trace.get("evidence_steps", []),
                "safe_outcome": row.get("safe_outcome", ""),
                "pass_fail": "PASS" if str(row.get("outcome", "")).startswith("PASS") else "REVIEW",
                "decision_trace": trace,
            }
        )
    return {
        "scenario_count": len(rows),
        "pass_count": sum(1 for row in rows if row["pass_fail"] == "PASS"),
        "rows": rows,
    }


def write_replay_report(
    report: dict[str, Any],
    *,
    output_json: Path = DEFAULT_REPLAY_REPORT_JSON,
    output_md: Path = DEFAULT_REPLAY_REPORT_MD,
) -> None:
    write_json(output_json, report)
    lines = [
        "# ShadowOps Replay Report",
        "",
        f"Scenarios: {report['scenario_count']}",
        f"Passed: {report['pass_count']}/{report['scenario_count']}",
        "",
        "| Scenario | Expected | Actual | Risk Delta | Evidence Requests | Result |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['scenario_name']} | {row['expected_behavior']} | {row['actual_decision']} | "
            f"{row['risk_delta']:.3f} | {len(row['evidence_requested'])} | {row['pass_fail']} |"
        )
    lines.extend(["", "## Scenario Details", ""])
    for row in report["rows"]:
        lines.extend(
            [
                f"### {row['scenario_name']}",
                "",
                f"- Action chain: `{row['action_chain']}`",
                f"- Safe outcome: {row['safe_outcome']}",
                f"- Evidence requested: {', '.join(row['evidence_requested']) if row['evidence_requested'] else 'none'}",
                f"- Safety rationale: {row['decision_trace'].get('safety_rationale', 'n/a')}",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def generate_replay_report() -> dict[str, Any]:
    report = build_replay_report()
    write_replay_report(report)
    return report


def main() -> int:
    report = generate_replay_report()
    print(f"Replay report: {report['pass_count']}/{report['scenario_count']} PASS")
    print(f"Saved: {DEFAULT_REPLAY_REPORT_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_REPLAY_REPORT_MD.relative_to(BACKEND_DIR)}")
    return 0 if report["pass_count"] == report["scenario_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
