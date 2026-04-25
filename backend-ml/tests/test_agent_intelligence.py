from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_memory import ActionMemoryRecord, SessionMemory
from evidence_planner import build_evidence_plan
from safe_outcome import generate_structured_safe_outcome
from training.shadowops_training_common import build_q_aware_decision, evaluate_training_gate


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
