from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main as api_main  # noqa: E402
from training.shadowops_training_common import build_q_aware_decision  # noqa: E402


def _assert_safe_contract(decision: dict) -> None:
    assert decision["decision"] in {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}
    assert not (
        decision["decision"] == "ALLOW"
        and decision.get("cumulative_risk_score", 0.0) >= 0.70
        and decision.get("missing_evidence")
    )
    assert isinstance(decision.get("structured_safe_outcome"), dict)
    assert decision["structured_safe_outcome"].get("remediation_steps")
    assert isinstance(decision.get("evidence_plan"), list)
    assert isinstance(decision.get("decision_trace"), dict)
    assert decision["decision_trace"].get("final_decision") in {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}


def test_unknown_domain_contract_does_not_crash() -> None:
    decision = build_q_aware_decision(
        "UNKNOWN_VENDOR",
        "CHANGE",
        "Routine non-production configuration update.",
        [0.0] * 16,
        actor="unknown",
        session_id="api-unknown-domain",
        service="unknown",
        environment="staging",
        provided_evidence=[],
        memory_context={},
    )

    _assert_safe_contract(decision)


def test_empty_evidence_and_empty_memory_contract() -> None:
    decision = build_q_aware_decision(
        "AWS",
        "UPDATE_IAM",
        "Create IAM admin role for production recovery.",
        [0.0] * 16,
        actor="unknown",
        session_id="api-empty-memory",
        service="iam",
        environment="production",
        provided_evidence=[],
        memory_context={},
    )

    _assert_safe_contract(decision)
    assert decision["missing_evidence"]


def test_missing_environment_contract_uses_safe_defaults() -> None:
    decision = build_q_aware_decision(
        "SOC",
        "UPDATE_FIREWALL",
        "Open port 22 to 0.0.0.0/0 for emergency access.",
        [0.8] * 16,
        actor="unknown",
        session_id="api-missing-env",
        service="firewall",
        environment="",
        provided_evidence=[],
        memory_context={},
    )

    _assert_safe_contract(decision)
    assert decision["decision"] != "ALLOW"


def test_malformed_optional_fields_are_normalized() -> None:
    decision = api_main._safe_decision_details(
        {
            "decision": "FORK",
            "confidence": "bad",
            "uncertainty": None,
            "risk_score": "0.8",
            "missing_evidence": "approval chain",
            "required_evidence": None,
            "risk_indicators": "admin_privilege",
            "safe_indicators": None,
            "structured_safe_outcome": {"remediation_steps": "fix"},
            "decision_trace": {"risk_signals": "bad"},
        },
        domain="AWS",
        risk_vector=[0.0] * 16,
        ambiguity=0.4,
    )

    _assert_safe_contract(decision)
    assert decision["confidence"] == 0.5
    assert decision["missing_evidence"] == ["approval chain"]
    assert decision["risk_indicators"] == ["admin_privilege"]


def test_unknown_action_label_becomes_safe_default() -> None:
    decision = api_main._safe_decision_details(
        {
            "decision": "DEPLOY_NOW",
            "risk_score": 0.9,
            "cumulative_risk_score": 0.9,
            "missing_evidence": ["approval chain"],
        },
        domain="AWS",
        risk_vector=[0.0] * 16,
        ambiguity=0.7,
    )

    _assert_safe_contract(decision)
    assert decision["decision"] == "QUARANTINE"


def test_missing_policy_config_falls_back_safely(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "_Q_AWARE_POLICY_AVAILABLE", False)
    monkeypatch.setattr(api_main, "build_q_aware_decision", None)

    decision = api_main._decide(
        "AWS",
        "UPDATE_IAM",
        "Create new IAM admin and export secrets.",
        "prompt",
        [0.9] * 16,
        actor="unknown",
        session_id="api-fallback",
        service="iam",
        environment="production",
        provided_evidence=[],
    )

    _assert_safe_contract(decision)
    assert decision["decision"] != "ALLOW"
