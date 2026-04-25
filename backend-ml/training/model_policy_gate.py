"""Deployment gate for choosing Q-aware policy or a measured model candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
GATE_JSON = REPORTS_DIR / "model_policy_gate_report.json"
GATE_MD = REPORTS_DIR / "model_policy_gate_report.md"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.shadowops_training_common import DEFAULT_DEMO_BENCHMARK_JSON, DEFAULT_MODEL_EVAL_JSON, read_json, write_json  # noqa: E402


def _source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "source_file": str(path.relative_to(BACKEND_DIR))}
    return {
        "exists": True,
        "source_file": str(path.relative_to(BACKEND_DIR)),
        "data": json.loads(path.read_text(encoding="utf-8")),
    }


def _qaware_from_benchmark(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    for row in report.get("metrics", []):
        if str(row.get("policy", "")).lower().replace("-", "_") in {"q_aware", "q_aware_policy"}:
            return row
    return None


def _model_eval_candidate(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report or report.get("model_load_error"):
        return None
    metrics = report.get("model_metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "name": report.get("model_name_or_path") or "model_eval_checkpoint",
        "source": "training/model_eval_report.json",
        "metrics": metrics,
    }


def _checkpoint_candidates(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    rows = []
    for row in report.get("checkpoints", []):
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            rows.append(
                {
                    "name": row.get("name"),
                    "source": "training/reports/checkpoint_comparison_report.json",
                    "metrics": metrics,
                    "champion_gate": row.get("champion_gate", {}),
                }
            )
    return rows


def _best_model_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["metrics"].get("reward_mean", -999)))


def _reject_reasons(model: dict[str, Any], qaware: dict[str, Any]) -> list[str]:
    metrics = model["metrics"]
    reasons = []
    if float(metrics.get("safety_accuracy", 0.0)) < float(qaware.get("safety_accuracy", 0.0)):
        reasons.append("checkpoint safety is worse than Q-aware")
    if float(metrics.get("invalid_output_rate", 1.0)) > 0.05:
        reasons.append("invalid output rate exceeds 0.05")
    if float(metrics.get("unsafe_decision_rate", 1.0)) > float(qaware.get("unsafe_decision_rate", 0.0)):
        reasons.append("unsafe decision rate is worse than Q-aware")
    return reasons


def build_model_policy_gate_report() -> dict[str, Any]:
    benchmark_source = _source(DEFAULT_DEMO_BENCHMARK_JSON)
    model_eval_source = _source(DEFAULT_MODEL_EVAL_JSON)
    checkpoint_source = _source(REPORTS_DIR / "checkpoint_comparison_report.json")
    hidden_source = _source(REPORTS_DIR / "hidden_eval_report.json")
    multistep_source = _source(REPORTS_DIR / "multistep_episode_report.json")

    qaware = _qaware_from_benchmark(benchmark_source.get("data")) or {}
    candidates = []
    model_eval_candidate = _model_eval_candidate(model_eval_source.get("data"))
    if model_eval_candidate:
        candidates.append(model_eval_candidate)
    candidates.extend(_checkpoint_candidates(checkpoint_source.get("data")))
    best = _best_model_candidate(candidates)

    if not best:
        status = "PENDING_REAL_CHECKPOINT_EVAL"
        recommendation = "DEPLOY_QAWARE_POLICY"
        reason = "No real checkpoint metrics are available. Q-aware remains the deployment guardrail."
        reject_reasons: list[str] = []
    else:
        reject_reasons = _reject_reasons(best, qaware)
        if reject_reasons:
            status = "REJECT_MODEL_UNSAFE"
            recommendation = "DEPLOY_QAWARE_POLICY"
            reason = "; ".join(reject_reasons)
        elif float(best["metrics"].get("reward_mean", -999)) > float(qaware.get("reward_mean", -999)):
            status = "DEPLOY_MODEL_CANDIDATE"
            recommendation = "DEPLOY_MODEL_CANDIDATE_WITH_QAWARE_GATE"
            reason = "Candidate passed safety filters and improved reward; keep Q-aware as verifier."
        else:
            status = "DEPLOY_QAWARE_POLICY"
            recommendation = "DEPLOY_QAWARE_POLICY"
            reason = "Candidate passed safety filters but did not beat Q-aware reward."

    return {
        "title": "ShadowOps Model-vs-Policy Deployment Gate",
        "gate_status": status,
        "deployment_recommendation": recommendation,
        "reason": reason,
        "q_aware_metrics": qaware or None,
        "best_model_candidate": best,
        "model_reject_reasons": reject_reasons,
        "sources": {
            "demo_benchmark": {k: v for k, v in benchmark_source.items() if k != "data"},
            "model_eval": {k: v for k, v in model_eval_source.items() if k != "data"},
            "checkpoint_comparison": {k: v for k, v in checkpoint_source.items() if k != "data"},
            "hidden_eval": {k: v for k, v in hidden_source.items() if k != "data"},
            "multistep_eval": {k: v for k, v in multistep_source.items() if k != "data"},
        },
        "notes": [
            "Trained checkpoint metrics are reported only when generated from a real checkpoint.",
            "If a model wins on reward but policy is safer, ShadowOps uses a hybrid: model proposes and Q-aware verifies.",
        ],
    }


def write_model_policy_gate_report(report: dict[str, Any]) -> None:
    write_json(GATE_JSON, report)
    q = report.get("q_aware_metrics") or {}
    best = report.get("best_model_candidate") or {}
    lines = [
        "# ShadowOps Model-vs-Policy Gate",
        "",
        f"Gate status: **{report['gate_status']}**",
        f"Recommendation: **{report['deployment_recommendation']}**",
        "",
        f"Reason: {report['reason']}",
        "",
        "## Q-aware Baseline",
        "",
        f"- exact_match: {q.get('exact_match', 'pending')}",
        f"- safety_accuracy: {q.get('safety_accuracy', 'pending')}",
        f"- unsafe_decision_rate: {q.get('unsafe_decision_rate', 'pending')}",
        f"- reward_mean: {q.get('reward_mean', 'pending')}",
        "",
        "## Best Model Candidate",
        "",
    ]
    if best:
        metrics = best.get("metrics", {})
        lines.extend(
            [
                f"- name: `{best.get('name')}`",
                f"- source: `{best.get('source')}`",
                f"- exact_match: {metrics.get('exact_match')}",
                f"- safety_accuracy: {metrics.get('safety_accuracy')}",
                f"- unsafe_decision_rate: {metrics.get('unsafe_decision_rate')}",
                f"- invalid_output_rate: {metrics.get('invalid_output_rate')}",
                f"- reward_mean: {metrics.get('reward_mean')}",
            ]
        )
    else:
        lines.append("No real checkpoint candidate has been evaluated yet.")
    lines.extend(["", "## Metric Sources", ""])
    for name, source in report["sources"].items():
        lines.append(f"- {name}: `{source['source_file']}` exists={source['exists']}")
    GATE_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ShadowOps model-vs-policy deployment gate report.")
    parser.add_argument("--skip-model-load", action="store_true", help="Accepted for validation symmetry; this script never loads a model.")
    parser.parse_args()
    report = build_model_policy_gate_report()
    write_model_policy_gate_report(report)
    print(f"Model-policy gate: {report['gate_status']}")
    print(f"Recommendation: {report['deployment_recommendation']}")
    print(f"Saved: {GATE_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {GATE_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
