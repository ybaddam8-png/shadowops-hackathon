"""Calibrate Q-aware policy config using local validation examples only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.analyze_oracle_gap import build_oracle_gap_analysis, write_oracle_gap_analysis  # noqa: E402
from training.shadowops_training_common import (  # noqa: E402
    DEFAULT_QAWARE_POLICY_CONFIG,
    DEFAULT_QAWARE_POLICY_CONFIG_JSON,
    evaluate_outputs,
    load_validation_samples_for_benchmark,
    q_aware_policy_action,
    reset_qaware_policy_config_cache,
    write_json,
)


DEFAULT_CALIBRATION_JSON = REPORTS_DIR / "qaware_calibration_report.json"
DEFAULT_CALIBRATION_MD = REPORTS_DIR / "qaware_calibration_report.md"


def _metrics(samples: list[dict[str, Any]], config: dict[str, Any], label: str) -> dict[str, Any]:
    outputs = [q_aware_policy_action(sample, config=config) for sample in samples]
    return evaluate_outputs(samples, outputs, label)


def _candidate_configs() -> list[dict[str, Any]]:
    base = dict(DEFAULT_QAWARE_POLICY_CONFIG)
    off = dict(base)
    off["medium_ambiguous_network_to_quarantine"] = False
    on = dict(base)
    on["medium_ambiguous_network_to_quarantine"] = True
    on["selected_reason"] = "quarantine medium ambiguous network/vendor exposure with missing evidence"
    narrow = dict(on)
    narrow["ambiguous_network_quarantine_terms"] = [
        "security group",
        "whitelist",
        "alert suppression",
        "vendor claims",
        "port 3389",
    ]
    narrow["selected_reason"] = "narrow network/vendor quarantine terms"
    return [off, narrow, on]


def _safe(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    return (
        metrics["safety_accuracy"] >= float(config.get("safety_floor", 1.0))
        and metrics["unsafe_decision_rate"] <= float(config.get("max_unsafe_decision_rate", 0.0))
        and metrics["false_positive_rate"] <= float(config.get("max_false_positive_rate", 0.01))
    )


def calibrate_qaware_policy() -> dict[str, Any]:
    samples, _ = load_validation_samples_for_benchmark()
    calibration = [sample for index, sample in enumerate(samples) if index % 2 == 0]
    held_out = [sample for index, sample in enumerate(samples) if index % 2 == 1]
    baseline_config = dict(DEFAULT_QAWARE_POLICY_CONFIG)
    baseline_config["medium_ambiguous_network_to_quarantine"] = False
    baseline_cal = _metrics(calibration, baseline_config, "baseline_calibration")
    baseline_hold = _metrics(held_out, baseline_config, "baseline_held_out")

    candidates = []
    best = None
    for config in _candidate_configs():
        cal_metrics = _metrics(calibration, config, "candidate_calibration")
        hold_metrics = _metrics(held_out, config, "candidate_held_out")
        candidate = {
            "config": config,
            "calibration_metrics": {
                key: cal_metrics[key]
                for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")
            },
            "held_out_metrics": {
                key: hold_metrics[key]
                for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")
            },
            "safe": _safe(cal_metrics, config) and _safe(hold_metrics, config),
        }
        candidate["score"] = (
            candidate["held_out_metrics"]["reward_mean"]
            + 0.40 * candidate["held_out_metrics"]["exact_match"]
            - 3.0 * candidate["held_out_metrics"]["false_positive_rate"]
        )
        candidates.append(candidate)
        if candidate["safe"] and (best is None or candidate["score"] > best["score"]):
            best = candidate

    selected = best or candidates[0]
    baseline_score = baseline_hold["reward_mean"] + 0.40 * baseline_hold["exact_match"]
    improved = selected["safe"] and selected["score"] > baseline_score + float(selected["config"].get("min_reward_improvement", 0.0))
    selected_config = selected["config"] if improved else baseline_config
    selected_config = dict(selected_config)
    selected_config["calibrated"] = bool(improved)
    selected_config["selected_metrics"] = selected["held_out_metrics"]
    selected_config["selection_reason"] = (
        selected_config.get("selected_reason", "safe calibration improved held-out reward")
        if improved
        else "No safe held-out improvement exceeded baseline; fallback retained."
    )
    write_json(DEFAULT_QAWARE_POLICY_CONFIG_JSON, selected_config)
    reset_qaware_policy_config_cache()
    gap_report = build_oracle_gap_analysis(samples)
    write_oracle_gap_analysis(gap_report)

    return {
        "sample_count": len(samples),
        "calibration_count": len(calibration),
        "held_out_count": len(held_out),
        "baseline": {
            "calibration": {key: baseline_cal[key] for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")},
            "held_out": {key: baseline_hold[key] for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")},
        },
        "candidates": candidates,
        "selected_config": selected_config,
        "selected_improved": bool(improved),
        "oracle_gap_after_selection": gap_report["after"],
        "output_config": str(DEFAULT_QAWARE_POLICY_CONFIG_JSON.relative_to(BACKEND_DIR)),
    }


def write_calibration_report(report: dict[str, Any]) -> None:
    write_json(DEFAULT_CALIBRATION_JSON, report)
    lines = [
        "# Q-aware Policy Calibration Report",
        "",
        f"Samples: {report['sample_count']}",
        f"Calibration split: {report['calibration_count']}",
        f"Held-out split: {report['held_out_count']}",
        "",
        f"Selected improved: {report['selected_improved']}",
        f"Selected reason: {report['selected_config']['selection_reason']}",
        "",
        "| Candidate | safe | held_out_exact | held_out_safety | held_out_unsafe | held_out_fpr | held_out_reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, candidate in enumerate(report["candidates"], start=1):
        metrics = candidate["held_out_metrics"]
        lines.append(
            f"| {index} | {candidate['safe']} | {metrics['exact_match']:.3f} | {metrics['safety_accuracy']:.3f} | "
            f"{metrics['unsafe_decision_rate']:.3f} | {metrics['false_positive_rate']:.3f} | {metrics['reward_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Safety Gate",
            "",
            f"- Safety floor: {report['selected_config'].get('safety_floor')}",
            f"- Max unsafe decision rate: {report['selected_config'].get('max_unsafe_decision_rate')}",
            f"- Max false-positive rate: {report['selected_config'].get('max_false_positive_rate')}",
        ]
    )
    DEFAULT_CALIBRATION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = calibrate_qaware_policy()
    write_calibration_report(report)
    selected = report["selected_config"]
    print(f"Selected calibrated config: {selected.get('calibrated')}")
    print(f"Reason: {selected['selection_reason']}")
    print(f"Saved: {DEFAULT_QAWARE_POLICY_CONFIG_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_CALIBRATION_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_CALIBRATION_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
