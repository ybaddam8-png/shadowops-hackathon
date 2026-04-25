"""Analyze why Q-aware policy differs from the oracle on validation data."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.shadowops_training_common import (  # noqa: E402
    DEFAULT_QAWARE_POLICY_CONFIG,
    evaluate_oracle,
    evaluate_outputs,
    load_validation_samples_for_benchmark,
    q_aware_policy_action,
    write_json,
)


DEFAULT_ORACLE_GAP_JSON = REPORTS_DIR / "oracle_gap_analysis.json"
DEFAULT_ORACLE_GAP_MD = REPORTS_DIR / "oracle_gap_analysis.md"
DEFAULT_ORACLE_GAP_V2_JSON = REPORTS_DIR / "oracle_gap_report.json"
DEFAULT_ORACLE_GAP_V2_MD = REPORTS_DIR / "oracle_gap_report.md"


def _reason(sample: dict[str, Any], q_action: str, oracle_action: str) -> str:
    text = str(sample.get("raw_payload", "")).lower()
    if sample.get("scenario_type") == "AMBIGUOUS" and q_action == "FORK" and oracle_action == "QUARANTINE":
        if any(term in text for term in ("security group", "whitelist", "open", "port", "vendor", "monitoring", "alert suppression")):
            return "ambiguous network/vendor exposure needs quarantine until evidence is provided"
        return "ambiguous missing-evidence case prefers quarantine over fork"
    if q_action == "BLOCK" and oracle_action == "QUARANTINE":
        return "clear danger should be isolated for evidence preservation"
    if q_action == "FORK" and oracle_action == "BLOCK":
        return "danger evidence is strong enough to block"
    if q_action == "QUARANTINE" and oracle_action == "FORK":
        return "human review is sufficient; quarantine is too conservative"
    if q_action == "ALLOW":
        return "policy allowed a case oracle handled conservatively"
    return "policy threshold differs from oracle reward preference"


def _fix_assessment(sample: dict[str, Any], q_action: str, oracle_action: str, reason: str) -> dict[str, Any]:
    safe_fix = False
    recommendation = "do not fix"
    hidden_eval_risk = "low"
    if q_action == "FORK" and oracle_action == "QUARANTINE" and "network/vendor" in reason:
        safe_fix = True
        recommendation = "generic quarantine rule for medium/high ambiguous exposure with missing evidence"
        hidden_eval_risk = "low if limited to exposure terms and missing evidence"
    elif q_action == "ALLOW":
        safe_fix = True
        recommendation = "raise uncertainty when approval or identity evidence is absent"
        hidden_eval_risk = "medium; validate false-positive traps before changing"
    return {
        "safe_to_fix_generically": safe_fix,
        "recommended_policy_improvement": recommendation,
        "would_hurt_hidden_eval": hidden_eval_risk,
    }


def _outputs(samples: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[str]:
    return [q_aware_policy_action(sample, config=config) for sample in samples]


def build_oracle_gap_analysis(samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if samples is None:
        samples, _ = load_validation_samples_for_benchmark()

    oracle_metrics = evaluate_oracle(samples)
    oracle_outputs = oracle_metrics["predicted_actions"]
    uncalibrated_config = dict(DEFAULT_QAWARE_POLICY_CONFIG)
    uncalibrated_config["medium_ambiguous_network_to_quarantine"] = False
    uncalibrated_outputs = _outputs(samples, uncalibrated_config)
    configured_outputs = _outputs(samples)
    before_metrics = evaluate_outputs(samples, uncalibrated_outputs, "q_aware_uncalibrated")
    after_metrics = evaluate_outputs(samples, configured_outputs, "q_aware_configured")

    misses = []
    domain_distribution: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    confusion = defaultdict(lambda: defaultdict(int))
    false_cases = {"allow": [], "block": [], "quarantine": [], "fork": []}

    for index, (sample, q_action, oracle_action, q_reward, oracle_reward) in enumerate(
        zip(samples, configured_outputs, oracle_outputs, after_metrics["sample_rewards"], oracle_metrics["sample_rewards"])
    ):
        confusion[oracle_action][q_action] += 1
        if q_action == oracle_action:
            continue
        reason = _reason(sample, q_action, oracle_action)
        reason_counts[reason] += 1
        domain = str(sample.get("domain", "unknown"))
        domain_distribution[domain] += 1
        row = {
            "index": index,
            "sample_id": sample.get("sample_id"),
            "domain": domain,
            "scenario_type": sample.get("scenario_type"),
            "severity": sample.get("severity"),
            "risk_score": sample.get("risk_score"),
            "expected_action": sample.get("correct_action"),
            "q_aware_action": q_action,
            "oracle_action": oracle_action,
            "reward_loss": round(float(oracle_reward) - float(q_reward), 6),
            "reason": reason,
            "payload": sample.get("raw_payload", ""),
        }
        row.update(_fix_assessment(sample, q_action, oracle_action, reason))
        misses.append(row)
        false_cases[q_action.lower()].append(row)

    return {
        "title": "ShadowOps Q-aware Oracle Gap Analysis",
        "sample_count": len(samples),
        "before": {
            "exact_match": before_metrics["exact_match"],
            "reward_mean": before_metrics["reward_mean"],
            "miss_count": sum(1 for q, oracle in zip(uncalibrated_outputs, oracle_outputs) if q != oracle),
        },
        "after": {
            "exact_match": after_metrics["exact_match"],
            "safety_accuracy": after_metrics["safety_accuracy"],
            "unsafe_decision_rate": after_metrics["unsafe_decision_rate"],
            "false_positive_rate": after_metrics["false_positive_rate"],
            "reward_mean": after_metrics["reward_mean"],
            "miss_count": len(misses),
        },
        "oracle": {
            "exact_match": oracle_metrics["exact_match"],
            "reward_mean": oracle_metrics["reward_mean"],
        },
        "domain_distribution_of_misses": dict(domain_distribution),
        "action_confusion_matrix": {expected: dict(predicted) for expected, predicted in confusion.items()},
        "false_cases": false_cases,
        "top_reasons": reason_counts.most_common(),
        "missed_samples": sorted(misses, key=lambda row: row["reward_loss"], reverse=True),
    }


def write_oracle_gap_analysis(
    report: dict[str, Any],
    *,
    output_json: Path = DEFAULT_ORACLE_GAP_JSON,
    output_md: Path = DEFAULT_ORACLE_GAP_MD,
) -> None:
    write_json(output_json, report)
    lines = [
        "# ShadowOps Oracle Gap Analysis",
        "",
        f"Samples: {report['sample_count']}",
        "",
        "## Before vs After Calibration",
        "",
        "| Version | exact_match | reward_mean | misses |",
        "| --- | ---: | ---: | ---: |",
        f"| Before | {report['before']['exact_match']:.3f} | {report['before']['reward_mean']:.3f} | {report['before']['miss_count']} |",
        f"| After | {report['after']['exact_match']:.3f} | {report['after']['reward_mean']:.3f} | {report['after']['miss_count']} |",
        f"| Oracle | {report['oracle']['exact_match']:.3f} | {report['oracle']['reward_mean']:.3f} | 0 |",
        "",
        "## Top Reasons Q-aware Differs From Oracle",
        "",
    ]
    if report["top_reasons"]:
        lines.extend(f"- {reason}: {count}" for reason, count in report["top_reasons"])
    else:
        lines.append("- No remaining validation misses.")
    lines.extend(["", "## Domain Distribution Of Misses", ""])
    if report["domain_distribution_of_misses"]:
        lines.extend(f"- {domain}: {count}" for domain, count in report["domain_distribution_of_misses"].items())
    else:
        lines.append("- No remaining validation misses.")
    lines.extend(["", "## Missed Samples", ""])
    for row in report["missed_samples"][:20]:
        lines.extend(
            [
                f"### {row['sample_id']}",
                f"- Domain: {row['domain']}",
                f"- Q-aware: {row['q_aware_action']}",
                f"- Oracle: {row['oracle_action']}",
                f"- Reward loss: {row['reward_loss']:.3f}",
                f"- Reason: {row['reason']}",
                f"- Payload: {row['payload']}",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")

    v2_report = dict(report)
    v2_report["title"] = "ShadowOps Oracle Gap Report v2"
    v2_report["guidance"] = (
        "Do not overfit the final miss. Only apply a policy improvement when the rule is generic, "
        "hidden evaluation does not degrade, and false positives do not increase."
    )
    write_json(DEFAULT_ORACLE_GAP_V2_JSON, v2_report)
    v2_lines = list(lines)
    v2_lines.extend(
        [
            "",
            "## Fix Safety Assessment",
            "",
            v2_report["guidance"],
            "",
        ]
    )
    for row in report["missed_samples"][:20]:
        v2_lines.extend(
            [
                f"- `{row.get('sample_id')}`: safe_to_fix_generically={row.get('safe_to_fix_generically')} | "
                f"hidden_eval_risk={row.get('would_hurt_hidden_eval')} | recommendation={row.get('recommended_policy_improvement')}",
            ]
        )
    DEFAULT_ORACLE_GAP_V2_MD.write_text("\n".join(v2_lines), encoding="utf-8")


def main() -> int:
    report = build_oracle_gap_analysis()
    write_oracle_gap_analysis(report)
    print(f"Oracle gap before: {report['before']['miss_count']} misses")
    print(f"Oracle gap after: {report['after']['miss_count']} misses")
    print(f"Saved: {DEFAULT_ORACLE_GAP_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_ORACLE_GAP_MD.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_ORACLE_GAP_V2_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_ORACLE_GAP_V2_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
