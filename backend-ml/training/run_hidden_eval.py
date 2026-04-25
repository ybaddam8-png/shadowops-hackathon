"""Run hidden-style policy evaluation without loading or training a model."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
HIDDEN_EVAL_PATH = TRAINING_DIR / "hidden_eval_scenarios.json"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from shadowops_env import extract_features  # noqa: E402
from training.shadowops_training_common import (  # noqa: E402
    evaluate_oracle,
    evaluate_policy_on_dataset,
    load_hard_negative_samples,
    load_validation_samples_for_benchmark,
    read_json,
    write_json,
)


DEFAULT_HIDDEN_EVAL_JSON = REPORTS_DIR / "hidden_eval_report.json"
DEFAULT_HIDDEN_EVAL_MD = REPORTS_DIR / "hidden_eval_report.md"


def _legacy_domain(domain: str) -> str:
    lowered = str(domain or "").lower()
    if any(term in lowered for term in ("github", "ci", "deployment")):
        return "GITHUB"
    if any(term in lowered for term in ("iam", "s3", "cloud", "kubernetes", "secrets", "saas", "data")):
        return "AWS"
    return "SOC"


def _scenario_type(expected_action: str, difficulty: str) -> str:
    if expected_action == "ALLOW":
        return "BENIGN_NOISY" if difficulty in {"medium", "hard", "adversarial"} else "BENIGN_CLEAN"
    if expected_action == "BLOCK":
        return "MALICIOUS_HARD" if difficulty in {"hard", "adversarial"} else "MALICIOUS_SOFT"
    return "AMBIGUOUS"


def hidden_scenario_to_sample(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    expected = str(scenario["expected_action"]).upper()
    text = scenario.get("intent", "") + " " + " ".join(scenario.get("indicators", [])) + " " + scenario.get("expected_reason", "")
    legacy = _legacy_domain(scenario.get("domain", "security"))
    risk_vector = extract_features(legacy, scenario.get("intent", "UNKNOWN"), text, random.Random(70_000 + index))
    difficulty = str(scenario.get("difficulty", "medium")).lower()
    severity = "CRITICAL" if difficulty == "adversarial" else ("HIGH" if difficulty == "hard" else "MEDIUM")
    is_malicious = expected == "BLOCK" or any(
        term in text.lower()
        for term in ("exfil", "malicious", "rogue", "no approval", "without approval", "fake", "bypass")
    )
    return {
        "sample_id": scenario.get("id", f"hidden-{index:04d}"),
        "split": "hidden_eval",
        "domain": legacy,
        "policy_domain": scenario.get("domain", "security"),
        "intent": scenario.get("intent", ""),
        "raw_payload": text,
        "action_summary": text,
        "actor": scenario.get("actor", "unknown"),
        "session_id": scenario.get("session_id", f"hidden-session-{index}"),
        "service": scenario.get("domain", "unknown"),
        "environment": scenario.get("environment", "production"),
        "indicators": list(scenario.get("indicators", [])),
        "safe_indicators": list(scenario.get("safe_indicators", [])),
        "provided_evidence": list(scenario.get("safe_indicators", [])),
        "required_evidence": list(scenario.get("missing_evidence", [])),
        "correct_action": expected,
        "expected_decision": expected,
        "is_malicious": is_malicious,
        "scenario_type": _scenario_type(expected, difficulty),
        "severity": severity,
        "risk_level": severity.lower(),
        "risk_score": round(sum(risk_vector[:4]) / max(len(risk_vector[:4]), 1), 6),
        "risk_vector": [round(float(value), 6) for value in risk_vector[:16]],
        "difficulty": difficulty,
        "expected_reason": scenario.get("expected_reason", ""),
        "use_agent_policy": False,
    }


def load_hidden_eval_samples(path: Path = HIDDEN_EVAL_PATH) -> list[dict[str, Any]]:
    rows = read_json(path, default=[])
    return [hidden_scenario_to_sample(row, index) for index, row in enumerate(rows)]


def _chain_samples() -> list[dict[str, Any]]:
    chains = [
        {
            "id": "chain-firewall-iam-export",
            "domain": "network_firewall",
            "intent": "firewall open followed by IAM admin creation and customer data export",
            "actor": "unknown",
            "environment": "production",
            "indicators": ["public_exposure", "admin_privilege", "data_export"],
            "safe_indicators": [],
            "missing_evidence": ["approval chain", "rollback plan", "data classification"],
            "expected_action": "QUARANTINE",
            "expected_reason": "multi-step malicious chain requires isolation",
            "difficulty": "adversarial",
        },
        {
            "id": "chain-ci-secret-prod",
            "domain": "github_ci",
            "intent": "CI workflow edit reads secrets then deploys to production",
            "actor": "new contributor",
            "environment": "production",
            "indicators": ["workflow_modification", "ci_secret_access", "production_deploy"],
            "safe_indicators": [],
            "missing_evidence": ["maintainer identity", "secret permission diff", "approved change ticket"],
            "expected_action": "FORK",
            "expected_reason": "production secret workflow change needs human review",
            "difficulty": "hard",
        },
        {
            "id": "chain-approved-pentest",
            "domain": "pentest",
            "intent": "approved pentest scan exploit simulation and report upload",
            "actor": "trusted maintainer",
            "environment": "staging",
            "indicators": ["scan_activity", "exploit_simulation"],
            "safe_indicators": ["authorization letter", "approved test window", "scope document", "emergency contact"],
            "missing_evidence": [],
            "expected_action": "ALLOW",
            "expected_reason": "valid pentest evidence supports restricted allow",
            "difficulty": "hard",
        },
    ]
    return [hidden_scenario_to_sample(row, index + 10_000) for index, row in enumerate(chains)]


def _ci95(value: float, n: int) -> list[float]:
    radius = 1.96 * math.sqrt(max(value * (1.0 - value), 0.0) / max(n, 1))
    return [max(0.0, value - radius), min(1.0, value + radius)]


def _domain_exact(samples: list[dict[str, Any]], predictions: list[str]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for sample, prediction in zip(samples, predictions):
        domain = str(sample.get("policy_domain") or sample.get("domain") or "unknown")
        totals[domain] += 1
        if prediction == (sample.get("correct_action") or sample.get("expected_decision")):
            hits[domain] += 1
    return {domain: hits[domain] / totals[domain] for domain in sorted(totals)}


def _precision_recall(samples: list[dict[str, Any]], predictions: list[str]) -> dict[str, dict[str, float]]:
    actions = ("ALLOW", "BLOCK", "FORK", "QUARANTINE")
    result = {}
    for action in actions:
        tp = sum(1 for sample, prediction in zip(samples, predictions) if prediction == action and sample.get("correct_action") == action)
        fp = sum(1 for sample, prediction in zip(samples, predictions) if prediction == action and sample.get("correct_action") != action)
        fn = sum(1 for sample, prediction in zip(samples, predictions) if prediction != action and sample.get("correct_action") == action)
        result[action] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
    return result


def _evaluate_dataset(samples: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    rows = {}
    for policy_name in ("random", "heuristic", "q_aware"):
        metrics = evaluate_policy_on_dataset(samples, policy_name, seed=1)
        predictions = metrics["predicted_actions"]
        rows[policy_name] = {
            "metrics": {
                key: metrics[key]
                for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")
            },
            "confidence_intervals": {
                key: _ci95(metrics[key], len(samples))
                for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate")
            },
            "per_domain_exact": _domain_exact(samples, predictions),
            "per_action_precision_recall": _precision_recall(samples, predictions),
        }
    oracle = evaluate_oracle(samples)
    rows["oracle"] = {
        "metrics": {
            key: oracle[key]
            for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")
        },
        "confidence_intervals": {
            key: _ci95(oracle[key], len(samples))
            for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate")
        },
        "per_domain_exact": _domain_exact(samples, oracle["predicted_actions"]),
        "per_action_precision_recall": _precision_recall(samples, oracle["predicted_actions"]),
    }
    return {"dataset": dataset_name, "sample_count": len(samples), "policies": rows}


def build_hidden_eval_report() -> dict[str, Any]:
    validation, _ = load_validation_samples_for_benchmark()
    hard_negative = load_hard_negative_samples()
    hidden = load_hidden_eval_samples()
    chain = _chain_samples()
    return {
        "title": "ShadowOps Hidden Evaluation Report",
        "datasets": {
            "validation": _evaluate_dataset(validation, "validation"),
            "hard_negative": _evaluate_dataset(hard_negative, "hard_negative"),
            "hidden_eval": _evaluate_dataset(hidden, "hidden_eval"),
            "multi_step_chain": _evaluate_dataset(chain, "multi_step_chain"),
        },
        "notes": [
            "No trained model metrics are included.",
            "Q-aware is evaluated as a deterministic policy baseline.",
        ],
    }


def write_hidden_eval_report(report: dict[str, Any]) -> None:
    write_json(DEFAULT_HIDDEN_EVAL_JSON, report)
    lines = ["# ShadowOps Hidden Evaluation Report", ""]
    for dataset_name, dataset in report["datasets"].items():
        lines.extend([f"## {dataset_name}", "", f"Samples: {dataset['sample_count']}", ""])
        lines.extend(
            [
                "| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for policy, row in dataset["policies"].items():
            metrics = row["metrics"]
            lines.append(
                f"| {policy} | {metrics['exact_match']:.3f} | {metrics['safety_accuracy']:.3f} | "
                f"{metrics['unsafe_decision_rate']:.3f} | {metrics['false_positive_rate']:.3f} | {metrics['reward_mean']:.3f} |"
            )
        q = dataset["policies"]["q_aware"]["metrics"]
        lines.extend(["", f"Q-aware exact 95% CI: {dataset['policies']['q_aware']['confidence_intervals']['exact_match']}", ""])
    DEFAULT_HIDDEN_EVAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = build_hidden_eval_report()
    write_hidden_eval_report(report)
    hidden_q = report["datasets"]["hidden_eval"]["policies"]["q_aware"]["metrics"]
    print(
        "Hidden Q-aware: "
        f"exact={hidden_q['exact_match']:.3f} safety={hidden_q['safety_accuracy']:.3f} "
        f"unsafe={hidden_q['unsafe_decision_rate']:.3f} reward={hidden_q['reward_mean']:.3f}"
    )
    print(f"Saved: {DEFAULT_HIDDEN_EVAL_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_HIDDEN_EVAL_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
