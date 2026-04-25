"""Domain-specific ShadowOps policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence_planner import DOMAIN_REQUIRED_EVIDENCE


def _norm(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _provided_text(scenario: dict[str, Any]) -> str:
    return " ".join(_norm(item) for item in scenario.get("provided_evidence", []))


@dataclass(frozen=True)
class DomainPolicy:
    name: str
    risk_terms: dict[str, tuple[str, ...]]
    safe_terms: dict[str, tuple[str, ...]]
    mitre_tactic: str
    mitre_technique: str
    escalation_terms: tuple[str, ...]
    safe_outcome_template: str


POLICIES = {
    "github_ci": DomainPolicy(
        name="github_ci",
        risk_terms={
            "ci_secret_access": ("secret", "token", "credential"),
            "workflow_modification": ("workflow", "pipeline", "ci"),
            "external_fetch": ("curl", "wget", "http://", "| bash"),
            "new_contributor": ("new contributor", "unknown maintainer", "fork"),
            "policy_override": ("ignore security", "override", "bypass"),
            "production_deploy": ("production deploy", "deploy"),
        },
        safe_terms={
            "trusted_maintainer": ("trusted maintainer", "maintainer identity verified", "maintainer identity"),
            "approved_change_ticket": ("approved change ticket", "change ticket", "approval ticket"),
            "package_provenance_verified": ("package provenance", "signed package"),
            "no_secret_scope_change": ("no secret", "no secret permission diff"),
        },
        mitre_tactic="Defense Evasion",
        mitre_technique="T1554 Compromise Client Software Binary",
        escalation_terms=("secret", "curl", "| bash", "production deploy"),
        safe_outcome_template="Block production merge until package provenance is verified.",
    ),
    "aws_s3": DomainPolicy(
        name="aws_s3",
        risk_terms={
            "public_bucket": ("public-read", "public bucket", "public access", "set public"),
            "sensitive_data": ("customer", "pii", "finance", "classified", "sensitive"),
            "data_export": ("export", "transfer", "download all", "exfil"),
            "disable_encryption": ("disable encryption", "unencrypted"),
            "external_destination": ("external", "attacker", "http://"),
        },
        safe_terms={
            "public_data_classification": ("classification public", "public website", "public content"),
            "approved_ticket": ("approval ticket", "approved ticket", "change ticket"),
            "bounded_exposure": ("exposure duration", "ttl", "temporary"),
            "bucket_policy_reviewed": ("bucket policy diff", "policy reviewed"),
        },
        mitre_tactic="Exfiltration",
        mitre_technique="T1537 Transfer Data to Cloud Account",
        escalation_terms=("pii", "disable encryption", "export", "external"),
        safe_outcome_template="Roll back public bucket access.",
    ),
    "iam": DomainPolicy(
        name="iam",
        risk_terms={
            "admin_privilege": ("admin", "administratoraccess", "full privileges", "privilege escalation"),
            "new_identity": ("new iam user", "create user", "new role"),
            "missing_mfa": ("mfa disabled", "without mfa", "no mfa"),
            "break_glass": ("break glass", "break-glass"),
            "external_principal": ("external", "vendor", "contractor"),
        },
        safe_terms={
            "approval_chain": ("approval chain", "two person approval", "security approval"),
            "requester_verified": ("requester identity", "requester verified"),
            "break_glass_approval": ("break glass approval", "break-glass approval", "incident commander"),
            "mfa_verified": ("mfa status verified", "mfa enabled"),
        },
        mitre_tactic="Privilege Escalation",
        mitre_technique="T1098 Account Manipulation",
        escalation_terms=("administratoraccess", "full privileges", "new iam user"),
        safe_outcome_template="Require two-person approval before privilege escalation.",
    ),
    "network_firewall": DomainPolicy(
        name="network_firewall",
        risk_terms={
            "public_exposure": ("0.0.0.0/0", "internet", "public", "open port"),
            "admin_port": ("port 22", "port 3389", "ssh", "rdp"),
            "review_bypass": ("bypass", "override", "ignore"),
            "no_rollback": ("no rollback", "missing rollback"),
            "source_unknown": ("unknown source", "any source"),
        },
        safe_terms={
            "approved_maintenance_window": ("approved maintenance window", "maintenance window"),
            "change_ticket": ("change ticket", "approved ticket", "approval ticket"),
            "source_destination_justified": ("source destination justification", "source/destination justification"),
            "rollback_plan": ("rollback plan", "rollback approved"),
        },
        mitre_tactic="Initial Access",
        mitre_technique="T1190 Exploit Public-Facing Application",
        escalation_terms=("0.0.0.0/0", "port 22", "port 3389"),
        safe_outcome_template="Require rollback plan before firewall exposure.",
    ),
    "pentest": DomainPolicy(
        name="pentest",
        risk_terms={
            "scan_activity": ("scan", "port scanning", "exploit", "payload"),
            "outside_window": ("outside window", "expired", "unauthorized"),
            "unknown_target": ("unknown target", "out of scope"),
            "production_target": ("production", "prod"),
        },
        safe_terms={
            "authorization_letter": ("authorization letter", "authorized pentest"),
            "approved_test_window": ("approved test window", "valid pentest window"),
            "scope_document": ("scope document", "in scope"),
            "emergency_contact": ("emergency contact",),
        },
        mitre_tactic="Reconnaissance",
        mitre_technique="T1595 Active Scanning",
        escalation_terms=("outside window", "unauthorized", "out of scope"),
        safe_outcome_template="Allow only during approved pentest window.",
    ),
}


def normalize_policy_domain(scenario: dict[str, Any]) -> str:
    explicit = str(scenario.get("policy_domain") or "").strip()
    if explicit:
        return explicit
    domain = str(scenario.get("domain") or "").upper()
    domain_key = _norm(scenario.get("domain") or "").replace(" ", "_")
    if domain_key in POLICIES:
        return domain_key
    intent = str(scenario.get("intent") or "").upper()
    text = _norm((scenario.get("action_summary") or scenario.get("raw_payload") or ""))
    if "pentest" in text or "pen test" in text or "port scanning" in text:
        return "pentest"
    if domain == "GITHUB" or "workflow" in text or "ci" in text:
        return "github_ci"
    if intent in {"MODIFY_S3"} or "bucket" in text or "s3" in text:
        return "aws_s3"
    if intent in {"UPDATE_IAM"} or "iam" in text or "administratoraccess" in text:
        return "iam"
    if intent in {"UPDATE_FIREWALL", "MODIFY_VPC", "UPDATE_SECURITY_GROUP"} or "firewall" in text or "security group" in text or "open port" in text:
        return "network_firewall"
    if domain == "AWS":
        return "iam" if "admin" in text else "aws_s3"
    if domain == "SOC":
        return "network_firewall"
    return "github_ci"


def _detect_terms(text: str, term_map: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for name, terms in term_map.items() if _has(text, *terms)]


def evaluate_domain_policy(scenario: dict[str, Any], memory_context: dict[str, Any] | None = None) -> dict[str, Any]:
    memory_context = memory_context or {}
    policy_domain = normalize_policy_domain(scenario)
    policy = POLICIES.get(policy_domain, POLICIES["github_ci"])
    text = " ".join(
        [
            _norm(scenario.get("action_summary") or scenario.get("raw_payload")),
            " ".join(_norm(item) for item in scenario.get("indicators", [])),
        ]
    )
    evidence_text = _provided_text(scenario)
    environment = _norm(scenario.get("environment") or "production")
    actor = _norm(scenario.get("actor") or "unknown")

    risk_indicators = _detect_terms(text, policy.risk_terms)
    safe_indicators = _detect_terms(evidence_text + " " + text, policy.safe_terms)

    if "no secret" in text:
        risk_indicators = [indicator for indicator in risk_indicators if indicator != "ci_secret_access"]
    if "no workflow" in text:
        risk_indicators = [indicator for indicator in risk_indicators if indicator != "workflow_modification"]
    if "no deploy" in text:
        risk_indicators = [indicator for indicator in risk_indicators if indicator != "production_deploy"]

    if environment == "production":
        risk_indicators.append("production_environment")
    if actor in {"unknown", "unknown actor", ""}:
        risk_indicators.append("unknown_actor")
    if environment in {"staging", "dev", "development"}:
        safe_indicators.append("non_production_environment")
    if actor.startswith("trusted maintainer"):
        safe_indicators.append("trusted_maintainer")
    if _has(evidence_text, "automation account", "known automation"):
        safe_indicators.append("known_automation_account")
    if _has(evidence_text, "trusted maintainer"):
        safe_indicators.append("trusted_maintainer")

    risk_indicators = list(dict.fromkeys(risk_indicators))
    safe_indicators = list(dict.fromkeys(safe_indicators))

    base_risk = min(1.0, 0.12 + 0.14 * len(risk_indicators))
    if any(term in text for term in policy.escalation_terms):
        base_risk = min(1.0, base_risk + 0.22)
    if memory_context.get("risky_chains"):
        base_risk = min(1.0, base_risk + 0.18)

    safe_evidence_score = min(1.0, 0.18 * len(safe_indicators))
    adjusted_risk = max(0.0, base_risk - safe_evidence_score * 0.55)

    if adjusted_risk <= 0.22:
        hint = "ALLOW"
    elif adjusted_risk >= 0.82 and safe_evidence_score < 0.25:
        hint = "BLOCK"
    elif adjusted_risk >= 0.62 and safe_evidence_score >= 0.25:
        hint = "FORK"
    elif adjusted_risk >= 0.45:
        hint = "QUARANTINE"
    else:
        hint = "FORK"

    reason_bits = []
    if risk_indicators:
        reason_bits.append("risk=" + ", ".join(risk_indicators))
    if safe_indicators:
        reason_bits.append("trusted=" + ", ".join(safe_indicators))
    if not reason_bits:
        reason_bits.append("no strong domain-specific indicators")

    return {
        "domain": policy_domain,
        "base_risk": round(base_risk, 6),
        "safe_evidence_score": round(safe_evidence_score, 6),
        "risk_indicators": risk_indicators,
        "safe_indicators": safe_indicators,
        "required_evidence": list(DOMAIN_REQUIRED_EVIDENCE.get(policy_domain, [])),
        "mitre_tactic": policy.mitre_tactic,
        "mitre_technique": policy.mitre_technique,
        "recommended_decision_hint": hint,
        "policy_reason": "; ".join(reason_bits),
    }
