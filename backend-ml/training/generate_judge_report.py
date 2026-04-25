"""Generate a laptop-safe judge readiness report for ShadowOps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_memory import ActionMemoryRecord, SessionMemory  # noqa: E402
from evidence_planner import build_evidence_plan  # noqa: E402
from safe_outcome import generate_structured_safe_outcome  # noqa: E402
from training.dataset_audit import run_dataset_audit  # noqa: E402
from training.shadowops_training_common import (  # noqa: E402
    DEFAULT_DEMO_BENCHMARK_JSON,
    DEFAULT_MODEL_EVAL_JSON,
    evaluate_training_gate,
    load_validation_samples_for_benchmark,
    run_demo_benchmark,
    run_reward_diagnostics,
    write_json,
)


DEFAULT_JUDGE_REPORT_JSON = TRAINING_DIR / "judge_readiness_report.json"
DEFAULT_JUDGE_REPORT_MD = TRAINING_DIR / "judge_readiness_report.md"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_summary() -> dict[str, Any]:
    if not DEFAULT_DEMO_BENCHMARK_JSON.exists():
        run_demo_benchmark()
    return _read_json(DEFAULT_DEMO_BENCHMARK_JSON, {})


def _gate_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    model_eval = _read_json(DEFAULT_MODEL_EVAL_JSON, {})
    if model_eval.get("training_gate"):
        return model_eval["training_gate"]
    rows = benchmark.get("metrics") or benchmark.get("rows") or []
    q_aware = next((row for row in rows if row.get("policy") == "Q-aware"), {})
    return evaluate_training_gate(None, q_aware)


def _memory_chain_example() -> dict[str, Any]:
    memory = SessionMemory(persistence_enabled=False)
    session_id = "judge-chain-example"
    for timestamp, summary, risk in (
        (1, "open firewall port 22 to 0.0.0.0/0", 0.68),
        (2, "create IAM admin user with AdministratorAccess", 0.74),
        (3, "export customer data to external transfer destination", 0.81),
    ):
        memory.add_record(
            ActionMemoryRecord(
                actor="unknown",
                session_id=session_id,
                service="aws",
                domain="AWS",
                environment="production",
                timestamp=timestamp,
                decision="QUARANTINE",
                risk_score=risk,
                action_summary=summary,
                indicators=[],
            )
        )
    return memory.summarize_memory_context(session_id)


def _evidence_examples() -> dict[str, Any]:
    plan = build_evidence_plan(
        "iam",
        ["admin_privilege", "unknown_actor"],
        "FORK",
        "production",
        ["requester identity", "approval chain", "privilege diff", "rollback plan"],
        risk_score=0.82,
    )
    safe_outcome = generate_structured_safe_outcome(
        "FORK",
        "iam",
        ["admin_privilege", "unknown_actor"],
        [],
        ["requester identity", "approval chain"],
        "production",
        evidence_plan=plan,
    )
    return {"evidence_plan": plan, "structured_safe_outcome": safe_outcome}


def build_judge_report() -> dict[str, Any]:
    val_samples, _ = load_validation_samples_for_benchmark()
    benchmark = _benchmark_summary()
    reward_diagnostics = run_reward_diagnostics(val_samples)
    dataset_audit = run_dataset_audit()
    evidence_examples = _evidence_examples()
    gate = _gate_summary(benchmark)
    return {
        "status": "judge_ready_laptop_safe",
        "model_claim": "No model improvement is claimed unless model_eval_report proves it.",
        "q_aware_guardrail": "q_aware_demo_policy remains the production guardrail.",
        "baseline_benchmark": benchmark,
        "reward_diagnostics": {
            "sample_count": reward_diagnostics["sample_count"],
            "reward_mean": reward_diagnostics["reward_mean"],
            "reward_std": reward_diagnostics["reward_std"],
            "percent_zero_std_groups": reward_diagnostics["percent_zero_std_groups"],
            "invalid_output_rate": reward_diagnostics["invalid_output_rate"],
            "action_distribution": reward_diagnostics["action_distribution"],
        },
        "dataset_audit": dataset_audit,
        "model_vs_policy_gate": gate,
        "evidence_planner_example": evidence_examples["evidence_plan"],
        "safe_outcome_example": evidence_examples["structured_safe_outcome"],
        "memory_risk_chain_example": _memory_chain_example(),
    }


def write_judge_report(
    report: dict[str, Any],
    *,
    output_json: Path = DEFAULT_JUDGE_REPORT_JSON,
    output_md: Path = DEFAULT_JUDGE_REPORT_MD,
) -> None:
    write_json(output_json, report)
    rows = report.get("baseline_benchmark", {}).get("metrics") or report.get("baseline_benchmark", {}).get("rows", [])
    lines = [
        "# ShadowOps Judge Readiness Report",
        "",
        f"Status: `{report['status']}`",
        "",
        report["model_claim"],
        "",
        f"Guardrail: {report['q_aware_guardrail']}",
        "",
        "## Baseline Benchmark",
        "",
        "| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['exact_match']:.3f} | {row['safety_accuracy']:.3f} | "
            f"{row['unsafe_decision_rate']:.3f} | {row['false_positive_rate']:.3f} | {row['reward_mean']:.3f} |"
        )

    reward = report["reward_diagnostics"]
    audit = report["dataset_audit"]
    gate = report["model_vs_policy_gate"]
    lines.extend(
        [
            "",
            "## Reward Diagnostics",
            "",
            f"- Samples: {reward['sample_count']}",
            f"- Reward mean/std: {reward['reward_mean']:.3f} / {reward['reward_std']:.3f}",
            f"- Zero-std groups: {reward['percent_zero_std_groups']:.1f}%",
            f"- Invalid output rate: {reward['invalid_output_rate']:.3f}",
            "",
            "## Dataset Audit",
            "",
            f"- Train samples: {audit['train_sample_count']}",
            f"- Validation samples: {audit['val_sample_count']}",
            f"- Hard negatives: {audit['hard_negative_count']}",
            f"- Duplicate prompts: {audit['duplicate_prompt_count']}",
            f"- Train/validation overlap: {audit['train_val_overlap_count']}",
            f"- Missing labels: {audit['missing_label_count']}",
            f"- Invalid action labels: {audit['invalid_action_label_count']}",
            "",
            "## Model-vs-Policy Gate",
            "",
            f"- Status: {gate.get('training_gate_status', 'UNKNOWN')}",
            f"- Reason: {gate.get('reason', 'n/a')}",
            f"- Recommended next action: {gate.get('recommended_next_action', 'n/a')}",
            "",
            "## Evidence Planner Example",
            "",
        ]
    )
    for item in report["evidence_planner_example"]:
        lines.append(f"- Step {item['step']} [{item['priority']}]: {item['ask']}")
    lines.extend(
        [
            "",
            "## Safe Outcome Example",
            "",
            f"- Outcome: {report['safe_outcome_example']['outcome']}",
            f"- Human review required: {report['safe_outcome_example']['human_review_required']}",
            f"- Remediation steps: {len(report['safe_outcome_example']['remediation_steps'])}",
            "",
            "## Memory/Risk Chain Example",
            "",
            f"- Session risk: {report['memory_risk_chain_example']['session_risk']:.3f}",
            f"- Detected chains: {', '.join(report['memory_risk_chain_example']['risky_chains'])}",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def generate_judge_report() -> dict[str, Any]:
    report = build_judge_report()
    write_judge_report(report)
    return report


def main() -> int:
    generate_judge_report()
    print(f"Saved: {DEFAULT_JUDGE_REPORT_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_JUDGE_REPORT_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
