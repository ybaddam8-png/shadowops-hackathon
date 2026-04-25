from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_memory import ActionMemoryRecord, SessionMemory
from evidence_planner import build_evidence_plan
from safe_outcome import generate_structured_safe_outcome
from training.dataset_audit import build_dataset_audit
from training.shadowops_training_common import (
    analyze_action_output,
    build_q_aware_decision,
    evaluate_outputs,
    evaluate_training_gate,
    parse_action,
)


def _record(session_id: str, action_summary: str, risk_score: float, timestamp: int) -> ActionMemoryRecord:
    return ActionMemoryRecord(
        actor="unknown",
        session_id=session_id,
        service="aws",
        domain="AWS",
        environment="production",
        timestamp=timestamp,
        decision="QUARANTINE",
        risk_score=risk_score,
        action_summary=action_summary,
        indicators=[],
    )


def test_firewall_iam_export_chain_accumulates_risk() -> None:
    memory = SessionMemory(persistence_enabled=False)
    session_id = "chain-firewall-iam-export"

    memory.add_record(_record(session_id, "open firewall port 22 to 0.0.0.0/0", 0.64, 1))
    first_risk = memory.compute_session_risk(session_id)
    memory.add_record(_record(session_id, "create IAM admin user with AdministratorAccess", 0.72, 2))
    second_risk = memory.compute_session_risk(session_id)
    memory.add_record(_record(session_id, "export customer data to external transfer destination", 0.80, 3))
    third_risk = memory.compute_session_risk(session_id)

    assert first_risk < second_risk < third_risk
    assert "firewall open -> IAM admin creation -> data export" in memory.detect_risky_chains(session_id)


def test_ci_secret_workflow_deploy_uses_stricter_decision_and_plan() -> None:
    memory_context = {
        "risky_chains": ["CI secret access -> workflow modification -> production deploy"],
        "session_risk": 0.82,
        "actor_risk": 0.75,
        "service_risk": 0.70,
        "recent_indicators": ["ci_secret_access", "workflow_modification", "production_deploy"],
    }

    decision = build_q_aware_decision(
        "GITHUB",
        "MERGE_PR",
        "CI change reads secrets, modifies workflow, then performs production deploy.",
        [0.0] * 16,
        actor="unknown",
        session_id="ci-chain",
        service="github-actions",
        environment="production",
        provided_evidence=[],
        memory_context=memory_context,
    )

    assert decision["decision"] in {"FORK", "QUARANTINE", "BLOCK"}
    assert decision["missing_evidence"]
    assert decision["evidence_plan"]


def test_public_bucket_transfer_iam_escalation_is_not_allowed() -> None:
    memory_context = {
        "risky_chains": ["public bucket -> external transfer -> permission escalation"],
        "session_risk": 0.90,
        "actor_risk": 0.80,
        "service_risk": 0.78,
        "recent_indicators": ["public_bucket", "external_transfer", "iam_admin"],
    }

    decision = build_q_aware_decision(
        "AWS",
        "MODIFY_S3",
        "Set customer PII bucket public-read, external transfer enabled, then create IAM admin role.",
        [0.0] * 16,
        actor="unknown",
        session_id="s3-chain",
        service="s3",
        environment="production",
        provided_evidence=[],
        memory_context=memory_context,
    )

    assert decision["decision"] in {"QUARANTINE", "BLOCK", "FORK"}


def test_authorized_pentest_reduces_false_positive_behavior() -> None:
    decision = build_q_aware_decision(
        "SOC",
        "PENTEST",
        "Authorized pentest port scanning against in-scope target list during approved test window.",
        [0.0] * 16,
        actor="trusted maintainer",
        session_id="pentest-safe",
        service="security",
        environment="staging",
        provided_evidence=[
            "authorization letter",
            "approved test window",
            "scope document",
            "target list",
            "emergency contact",
        ],
    )

    assert decision["decision"] in {"ALLOW", "FORK", "QUARANTINE"}
    assert decision["decision"] != "BLOCK"
    assert "approved pentest" in decision["safe_outcome"].lower() or decision["safe_indicators"]


def test_evidence_planner_orders_identity_approval_impact_rollback() -> None:
    plan = build_evidence_plan(
        "iam",
        ["admin_privilege"],
        "FORK",
        "production",
        ["rollback plan", "approval chain", "requester identity", "privilege diff"],
        risk_score=0.82,
    )
    asks = " | ".join(item["ask"].lower() for item in plan)

    assert asks.index("requester identity") < asks.index("approval chain")
    assert asks.index("approval chain") < asks.index("privilege diff")
    assert asks.index("privilege diff") < asks.index("rollback plan")


def test_structured_safe_outcome_has_remediation_and_human_review() -> None:
    outcome = generate_structured_safe_outcome(
        "FORK",
        "iam",
        ["admin_privilege", "unknown_actor"],
        [],
        ["approval chain", "requester identity"],
        "production",
        evidence_plan=[
            {
                "step": 1,
                "ask": "Verify requester identity before approval.",
                "priority": "critical",
                "reason": "Identity required.",
                "blocks_decision": True,
            }
        ],
    )

    assert len(outcome["remediation_steps"]) > 0
    assert outcome["human_review_required"] is True
    assert outcome["evidence_needed"]


def test_model_vs_policy_gate_reports_baseline_only_no_model() -> None:
    q_aware_metrics = {
        "safety_accuracy": 1.0,
        "unsafe_decision_rate": 0.0,
        "reward_mean": 1.6,
    }
    gate = evaluate_training_gate(None, q_aware_metrics)

    assert gate["training_gate_status"] == "FAIL"
    assert gate["training_gate_passed"] is False
    assert "No model metrics" in gate["reason"]


def test_unknown_domain_does_not_crash() -> None:
    decision = build_q_aware_decision(
        "UNKNOWN_VENDOR",
        "DO_THING",
        "Routine staging change with no public exposure.",
        [0.0] * 16,
        actor="unknown",
        session_id="unknown-domain",
        service="unknown",
        environment="staging",
        provided_evidence=[],
    )

    assert decision["decision"] in {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}
    assert decision["policy_name"] == "q_aware_demo_policy"
    assert isinstance(decision["missing_evidence"], list)


def test_empty_evidence_does_not_crash() -> None:
    decision = build_q_aware_decision(
        "AWS",
        "UPDATE_IAM",
        "Create temporary admin role for production recovery.",
        [0.0] * 16,
        actor="unknown",
        session_id="empty-evidence",
        service="iam",
        environment="production",
        provided_evidence=[],
    )

    assert decision["required_evidence"]
    assert decision["missing_evidence"]
    assert decision["structured_safe_outcome"]["remediation_steps"]


def test_invalid_model_output_is_rejected() -> None:
    analysis = analyze_action_output("please allow this because it is fine")

    assert analysis["parsed_action"] is None
    assert analysis["invalid_output"] is True


def test_action_normalization_accepts_clear_labels_and_synonyms() -> None:
    assert parse_action(" allow. ") == "ALLOW"
    assert parse_action("decision=quarantine") == "QUARANTINE"
    assert parse_action("deny") == "BLOCK"
    assert parse_action("human_review") == "FORK"
    assert parse_action("please allow this") is None


def test_evaluate_outputs_reports_normalized_action_distribution() -> None:
    sample = {
        "sample_id": "unit-1",
        "correct_action": "ALLOW",
        "is_malicious": False,
        "severity": "LOW",
        "scenario_type": "BENIGN_CLEAN",
        "risk_score": 0.0,
        "raw_payload": "Routine approved staging change.",
        "provided_evidence": ["approved ticket"],
        "required_evidence": [],
    }

    metrics = evaluate_outputs([sample], [" allow. "], "unit")

    assert metrics["invalid_output_rate"] == 0.0
    assert metrics["parse_failure_rate"] == 0.0
    assert metrics["normalized_action_distribution"]["ALLOW"] == 1.0


def test_dataset_audit_detects_duplicates_and_invalid_labels() -> None:
    train = [
        {
            "prompt": "same prompt",
            "correct_action": "ALLOW",
            "domain": "AWS",
            "is_malicious": False,
            "scenario_type": "BENIGN_CLEAN",
            "severity": "LOW",
        }
    ]
    val = [
        {
            "prompt": "same prompt",
            "correct_action": "BOGUS",
            "domain": "AWS",
            "is_malicious": False,
            "scenario_type": "BENIGN_NOISY",
            "severity": "MEDIUM",
        },
        {
            "prompt": "missing label",
            "domain": "GITHUB",
            "is_malicious": False,
            "scenario_type": "BENIGN_CLEAN",
            "severity": "LOW",
        },
    ]

    audit = build_dataset_audit(train, val, hard_negative_samples=[])

    assert audit["duplicate_prompt_count"] == 1
    assert audit["train_val_overlap_count"] == 1
    assert audit["missing_label_count"] == 1
    assert audit["invalid_action_label_count"] == 1
    assert audit["passed_preflight"] is False


def test_structured_safe_outcome_always_includes_remediation_steps() -> None:
    outcome = generate_structured_safe_outcome(
        "ALLOW",
        "github_ci",
        [],
        ["approved_change_ticket"],
        [],
        "staging",
    )

    assert outcome["remediation_steps"]
    assert isinstance(outcome["allowed_next_actions"], list)


def test_evidence_plan_step_numbers_are_ordered() -> None:
    plan = build_evidence_plan(
        "network_firewall",
        ["public_exposure"],
        "QUARANTINE",
        "production",
        ["rollback plan", "change ticket", "source/destination justification"],
        risk_score=0.7,
    )

    assert [item["step"] for item in plan] == list(range(1, len(plan) + 1))
