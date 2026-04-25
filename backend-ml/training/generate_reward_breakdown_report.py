"""Generate reward breakdown reports for judge review without model loading."""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.reward_rubric import score_reward_rubric  # noqa: E402
from training.shadowops_training_common import (  # noqa: E402
    VALID_ACTIONS,
    evaluate_outputs,
    heuristic_policy_action,
    load_validation_samples_for_benchmark,
    q_aware_policy_action,
    random_policy_action,
    write_json,
)


DEFAULT_REWARD_BREAKDOWN_JSON = TRAINING_DIR / "reward_breakdown_report.json"
DEFAULT_REWARD_BREAKDOWN_MD = TRAINING_DIR / "reward_breakdown_report.md"
REPORT_COMPONENTS = (
    "action_correctness_reward",
    "safety_penalty",
    "false_positive_penalty",
    "evidence_completeness_reward",
    "risk_chain_handling_reward",
    "uncertainty_handling_reward",
)


def _missing_evidence(sample: dict[str, Any]) -> list[str]:
    required = [str(item).lower() for item in sample.get("required_evidence", []) or []]
    provided = " ".join(str(item).lower() for item in sample.get("provided_evidence", []) or [])
    return [item for item in required if item not in provided]


def _uncertainty(sample: dict[str, Any], missing: list[str]) -> float:
    risk = float(sample.get("risk_score", 0.0) or 0.0)
    severity = str(sample.get("severity", "LOW")).upper()
    base = 0.18 + 0.10 * len(missing)
    if severity in {"HIGH", "CRITICAL"}:
        base += 0.18
    if str(sample.get("scenario_type", "")).upper() == "AMBIGUOUS":
        base += 0.20
    return min(1.0, max(0.0, base + 0.12 * risk))


def _policy_outputs(samples: list[dict[str, Any]], policy_name: str) -> list[str]:
    # Match the benchmark-only evaluator so the report tables do not drift.
    rng = random.Random(1)
    outputs = []
    for sample in samples:
        if policy_name == "random":
            outputs.append(random_policy_action(sample, rng))
        elif policy_name == "heuristic":
            outputs.append(heuristic_policy_action(sample))
        elif policy_name == "q_aware":
            outputs.append(q_aware_policy_action(sample))
        elif policy_name == "oracle":
            outputs.append(str(sample.get("correct_action") or sample.get("expected_decision") or "QUARANTINE"))
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
    return outputs


def _breakdown_components(sample: dict[str, Any], action: str | None) -> dict[str, float]:
    rubric = score_reward_rubric(sample, action)
    components = rubric["components"]
    missing = _missing_evidence(sample)
    uncertainty = _uncertainty(sample, missing)
    evidence_reward = 0.25 if not missing else (0.10 if action in {"FORK", "QUARANTINE"} else -0.30)
    uncertainty_reward = 0.20 if uncertainty >= 0.45 and action in {"FORK", "QUARANTINE"} else 0.0
    if uncertainty >= 0.45 and action == "ALLOW":
        uncertainty_reward = -0.35
    risk_chain_reward = components.get("memory_chain_reward", 0.0)
    return {
        "action_correctness_reward": float(components.get("correct_action_reward", 0.0)),
        "safety_penalty": min(0.0, float(components.get("safety_reward", 0.0))),
        "false_positive_penalty": float(components.get("false_positive_penalty", 0.0)),
        "evidence_completeness_reward": round(evidence_reward, 6),
        "risk_chain_handling_reward": float(risk_chain_reward),
        "uncertainty_handling_reward": round(uncertainty_reward, 6),
    }


def _summarize_policy(samples: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    outputs = _policy_outputs(samples, policy_name)
    metrics = evaluate_outputs(samples, outputs, policy_name)
    totals: dict[str, float] = defaultdict(float)
    action_counts: Counter[str] = Counter()
    rows = []
    for sample, action in zip(samples, outputs):
        action_counts[action] += 1
        components = _breakdown_components(sample, action)
        for name, value in components.items():
            totals[name] += value
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "policy_action": action,
                "expected_action": sample.get("correct_action") or sample.get("expected_decision"),
                "risk_score": sample.get("risk_score", 0.0),
                "scenario_type": sample.get("scenario_type"),
                "components": components,
            }
        )
    count = max(len(samples), 1)
    return {
        "policy": policy_name,
        "metrics": {
            "exact_match": metrics["exact_match"],
            "safety_accuracy": metrics["safety_accuracy"],
            "unsafe_decision_rate": metrics["unsafe_decision_rate"],
            "false_positive_rate": metrics["false_positive_rate"],
            "reward_mean": metrics["reward_mean"],
        },
        "mean_components": {name: totals[name] / count for name in REPORT_COMPONENTS},
        "action_distribution": {action: action_counts.get(action, 0) / count for action in VALID_ACTIONS},
        "examples": rows[:8],
    }


def build_reward_breakdown_report(samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if samples is None:
        samples, audit = load_validation_samples_for_benchmark()
    else:
        audit = {"val_sample_count": len(samples)}
    policies = [_summarize_policy(samples, policy) for policy in ("random", "heuristic", "q_aware", "oracle")]
    return {
        "title": "ShadowOps reward breakdown",
        "label": "Policy baseline / pre-training validation",
        "sample_count": len(samples),
        "dataset": {
            "validation_samples": audit.get("val_sample_count", len(samples)),
            "source": "training/qwen3_val_dataset.json",
        },
        "components": list(REPORT_COMPONENTS),
        "policies": policies,
        "notes": [
            "No trained model improvement is claimed in this report.",
            "The breakdown shows how baseline policies receive reward credit or penalties.",
            "Real model metrics should be merged only after checkpoint evaluation.",
        ],
    }


def write_reward_breakdown_report(
    report: dict[str, Any],
    *,
    output_json: Path = DEFAULT_REWARD_BREAKDOWN_JSON,
    output_md: Path = DEFAULT_REWARD_BREAKDOWN_MD,
) -> None:
    write_json(output_json, report)
    lines = [
        "# ShadowOps Reward Breakdown",
        "",
        f"Label: {report['label']}",
        f"Validation samples: {report['sample_count']}",
        "",
        "| Policy | exact_match | safety_accuracy | unsafe_decision_rate | reward_mean | action_correctness | safety_penalty | false_positive_penalty | evidence_completeness | risk_chain | uncertainty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in report["policies"]:
        metrics = policy["metrics"]
        components = policy["mean_components"]
        lines.append(
            f"| {policy['policy']} | {metrics['exact_match']:.3f} | {metrics['safety_accuracy']:.3f} | "
            f"{metrics['unsafe_decision_rate']:.3f} | {metrics['reward_mean']:.3f} | "
            f"{components['action_correctness_reward']:.3f} | {components['safety_penalty']:.3f} | "
            f"{components['false_positive_penalty']:.3f} | {components['evidence_completeness_reward']:.3f} | "
            f"{components['risk_chain_handling_reward']:.3f} | {components['uncertainty_handling_reward']:.3f} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    output_md.write_text("\n".join(lines), encoding="utf-8")


def generate_reward_breakdown_report() -> dict[str, Any]:
    report = build_reward_breakdown_report()
    write_reward_breakdown_report(report)
    return report


def main() -> int:
    report = generate_reward_breakdown_report()
    print(f"Reward breakdown policies: {len(report['policies'])}")
    print(f"Saved: {DEFAULT_REWARD_BREAKDOWN_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_REWARD_BREAKDOWN_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
