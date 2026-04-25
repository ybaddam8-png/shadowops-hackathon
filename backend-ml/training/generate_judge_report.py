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
from training.generate_replay_report import build_replay_report  # noqa: E402
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
FINAL_JUDGE_CLAIM = (
    "ShadowOps currently proves production-safe behavior through the Q-aware policy guardrail. "
    "Trained model improvement is only claimed after checkpoint evaluation passes the "
    "model-vs-policy safety gate."
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_summary() -> dict[str, Any]:
    if not DEFAULT_DEMO_BENCHMARK_JSON.exists():
        run_demo_benchmark()
    return _read_json(DEFAULT_DEMO_BENCHMARK_JSON, {})


def _benchmark_rows(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    return list(benchmark.get("metrics") or benchmark.get("rows") or [])


def _gate_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    model_eval = _read_json(DEFAULT_MODEL_EVAL_JSON, {})
    if model_eval.get("training_gate"):
        return model_eval["training_gate"]
    q_aware = next((row for row in _benchmark_rows(benchmark) if row.get("policy") == "Q-aware"), {})
    return evaluate_training_gate(None, q_aware)


def _memory_chain_example() -> dict[str, Any]:
    memory = SessionMemory(persistence_enabled=False)
    session_id = "judge-chain-example"
    actions = (
        (1, "open firewall port 22 to 0.0.0.0/0", 0.68),
        (2, "create IAM admin user with AdministratorAccess", 0.74),
        (3, "export customer data to external transfer destination", 0.81),
    )
    for timestamp, summary, risk in actions:
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
    context = memory.summarize_memory_context(session_id)
    context["action_chain"] = [summary for _, summary, _ in actions]
    return context


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
    replay_report = build_replay_report()
    gate = _gate_summary(benchmark)
    return {
        "status": "judge_ready_laptop_safe",
        "executive_summary": (
            "ShadowOps is a laptop-safe autonomous incident response supervisor that evaluates "
            "high-risk cloud, CI, IAM, network, and pentest actions before they reach production."
        ),
        "what_shadowops_does": [
            "Scores action risk from payload signals and domain policy.",
            "Accumulates session memory to catch multi-step attack chains.",
            "Requests missing operational evidence before allowing risky actions.",
            "Returns safe outcomes such as quarantine, fork-to-human, rollback, or restricted allow.",
        ],
        "why_it_is_different": [
            "Uses a deterministic Q-aware guardrail by default, so the demo is safe without a GPU.",
            "Compares future model checkpoints against policy baselines before trusting them.",
            "Explains decisions through audit traces, evidence plans, and safe outcomes.",
            "Optimizes for false-positive reduction when trusted evidence is present.",
        ],
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
        "safety_guardrail": {
            "policy": "q_aware_demo_policy",
            "explanation": (
                "The Q-aware policy is the production guardrail until a trained checkpoint "
                "beats reference metrics and passes the safety gate."
            ),
        },
        "evidence_planner_example": evidence_examples["evidence_plan"],
        "safe_outcome_example": evidence_examples["structured_safe_outcome"],
        "memory_risk_chain_example": _memory_chain_example(),
        "model_vs_policy_gate": gate,
        "replay_summary": {
            "scenario_count": replay_report["scenario_count"],
            "pass_count": replay_report["pass_count"],
        },
        "gpu_training_handoff": [
            "Pull shadowops-backend-agent-upgrade.",
            "Run laptop-safe baseline/eval first.",
            "Run SFT smoke, then evaluate SFT checkpoint.",
            "Run GRPO smoke, then evaluate GRPO checkpoint.",
            "Only run longer GRPO after model_eval_report proves safety and reward progress.",
        ],
        "honest_limitations": [
            "No model improvement is claimed without a real checkpoint evaluation report.",
            "Laptop validation verifies policy, parser, evidence planning, memory, reports, and tests only.",
            "HF GPU training must be launched manually and monitored for credit safety.",
        ],
        "final_judge_claim": FINAL_JUDGE_CLAIM,
    }


def _write_benchmark_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['exact_match']:.3f} | {row['safety_accuracy']:.3f} | "
            f"{row['unsafe_decision_rate']:.3f} | {row['false_positive_rate']:.3f} | {row['reward_mean']:.3f} |"
        )


def write_judge_report(
    report: dict[str, Any],
    *,
    output_json: Path = DEFAULT_JUDGE_REPORT_JSON,
    output_md: Path = DEFAULT_JUDGE_REPORT_MD,
) -> None:
    write_json(output_json, report)
    rows = _benchmark_rows(report.get("baseline_benchmark", {}))
    reward = report["reward_diagnostics"]
    audit = report["dataset_audit"]
    gate = report["model_vs_policy_gate"]
    memory = report["memory_risk_chain_example"]

    lines = [
        "# ShadowOps Judge Readiness Report",
        "",
        "## 1. Executive Summary",
        "",
        report["executive_summary"],
        "",
        "## 2. What ShadowOps Does",
        "",
    ]
    lines.extend(f"- {item}" for item in report["what_shadowops_does"])
    lines.extend(["", "## 3. Why It Is Different", ""])
    lines.extend(f"- {item}" for item in report["why_it_is_different"])
    lines.extend(["", "## 4. Benchmark Results", ""])
    _write_benchmark_table(lines, rows)
    lines.extend(
        [
            "",
            "## 5. Dataset Audit",
            "",
            f"- Train samples: {audit['train_sample_count']}",
            f"- Validation samples: {audit['val_sample_count']}",
            f"- Hard negatives: {audit['hard_negative_count']}",
            f"- Duplicate prompts: {audit['duplicate_prompt_count']}",
            f"- Train/validation overlap: {audit['train_val_overlap_count']}",
            f"- Missing labels: {audit['missing_label_count']}",
            f"- Invalid action labels: {audit['invalid_action_label_count']}",
            f"- False-positive challenges: {audit['false_positive_challenge_count']}",
            "",
            "## 6. Reward Diagnostics",
            "",
            f"- Samples: {reward['sample_count']}",
            f"- Reward mean/std: {reward['reward_mean']:.3f} / {reward['reward_std']:.3f}",
            f"- Zero-std groups: {reward['percent_zero_std_groups']:.1f}%",
            f"- Invalid output rate: {reward['invalid_output_rate']:.3f}",
            "",
            "## 7. Safety Guardrail Explanation",
            "",
            f"- Policy: `{report['safety_guardrail']['policy']}`",
            f"- {report['safety_guardrail']['explanation']}",
            "",
            "## 8. Evidence Planning Example",
            "",
        ]
    )
    for item in report["evidence_planner_example"]:
        lines.append(f"- Step {item['step']} [{item['priority']}]: {item['ask']}")
    lines.extend(
        [
            "",
            "## 9. Memory + Risk Chain Example",
            "",
            f"- Action chain: {' -> '.join(memory['action_chain'])}",
            f"- Session risk: {memory['session_risk']:.3f}",
            f"- Detected chains: {', '.join(memory['risky_chains'])}",
            "",
            "## 10. Model-vs-Policy Gate",
            "",
            f"- Status: {gate.get('training_gate_status', 'UNKNOWN')}",
            f"- Reason: {gate.get('reason', 'n/a')}",
            f"- Recommended next action: {gate.get('recommended_next_action', 'n/a')}",
            "",
            "## 11. GPU Training Handoff",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["gpu_training_handoff"])
    lines.extend(["", "## 12. Honest Limitations", ""])
    lines.extend(f"- {item}" for item in report["honest_limitations"])
    lines.extend(
        [
            "",
            "## 13. Final Judge Claim",
            "",
            f"> {report['final_judge_claim']}",
            "",
            "## Replay Summary",
            "",
            f"- Replay scenarios passed: {report['replay_summary']['pass_count']}/{report['replay_summary']['scenario_count']}",
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
