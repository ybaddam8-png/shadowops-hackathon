"""Operational safe-outcome generator for ShadowOps decisions."""

from __future__ import annotations

from typing import Iterable, Any


def _contains(values: Iterable[str], *needles: str) -> bool:
    text = " ".join(str(value).lower() for value in values or [])
    return any(needle.lower() in text for needle in needles)


def generate_safe_outcome(
    decision: str,
    domain: str,
    risk_indicators: Iterable[str],
    safe_indicators: Iterable[str],
    missing_evidence: Iterable[str],
    environment: str,
) -> str:
    decision = str(decision or "").upper()
    domain = str(domain or "").lower()
    environment = str(environment or "production").lower()
    risk_indicators = list(risk_indicators or [])
    safe_indicators = list(safe_indicators or [])
    missing_evidence = list(missing_evidence or [])

    if decision == "ALLOW":
        if "staging" in environment or "dev" in environment:
            return "Allow in staging/dev only."
        if _contains(safe_indicators, "valid pentest"):
            return "Allow only during approved pentest window."
        if safe_indicators:
            return "ALLOW_WITH_RESTRICTIONS: proceed only under documented approval and monitoring."
        return "Allow low-risk action."

    if decision == "BLOCK":
        if _contains(risk_indicators, "secret"):
            return "Rotate exposed secrets and block merge."
        if _contains(risk_indicators, "public bucket", "public exposure"):
            return "Roll back public bucket access."
        if _contains(risk_indicators, "unknown actor", "production deploy"):
            return "Deny production deployment by unknown actor."
        if domain == "github_ci":
            return "Block production merge until package provenance is verified."
        return "Block action before production impact."

    if decision == "FORK":
        if domain == "iam" or _contains(risk_indicators, "admin", "privilege"):
            return "Fork to human reviewer with IAM approval chain."
        if domain == "github_ci":
            return "Fork to human reviewer before production workflow changes."
        return "Fork to human reviewer before execution."

    if decision == "QUARANTINE":
        if missing_evidence:
            return "Quarantine action until missing evidence is provided."
        if domain == "network_firewall":
            return "Require rollback plan before firewall exposure."
        if _contains(risk_indicators, "admin", "privilege"):
            return "Require two-person approval before privilege escalation."
        return "Quarantine action pending additional verification."

    return "Hold action until a valid supervisor decision is available."


def generate_structured_safe_outcome(
    decision: str,
    domain: str,
    risk_indicators: Iterable[str],
    safe_indicators: Iterable[str],
    missing_evidence: Iterable[str],
    environment: str,
    evidence_plan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an operational outcome object while keeping the string API stable."""

    decision = str(decision or "").upper()
    domain = str(domain or "").lower()
    risk_indicators = list(risk_indicators or [])
    safe_indicators = list(safe_indicators or [])
    missing_evidence = list(missing_evidence or [])
    evidence_plan = list(evidence_plan or [])
    outcome = generate_safe_outcome(
        decision,
        domain,
        risk_indicators,
        safe_indicators,
        missing_evidence,
        environment,
    )

    remediation_steps: list[str] = []
    allowed_next_actions: list[str] = []
    rollback_required = False
    human_review_required = decision in {"FORK", "QUARANTINE"} or bool(missing_evidence)
    monitoring_required = decision in {"ALLOW", "FORK", "QUARANTINE"}

    if _contains(risk_indicators, "secret"):
        remediation_steps.append("Rotate exposed secrets and review CI secret access.")
    if _contains(risk_indicators, "public", "bucket", "0.0.0.0/0", "open port"):
        remediation_steps.append("Remove public exposure until approval and rollback evidence are verified.")
        rollback_required = True
    if _contains(risk_indicators, "admin", "privilege"):
        remediation_steps.append("Require two-person approval before privilege escalation.")
        human_review_required = True
    if missing_evidence:
        remediation_steps.append("Collect missing evidence in plan order before allowing production impact.")
    if not remediation_steps:
        remediation_steps.append("Proceed with monitoring and retain the approval trail.")

    if decision == "ALLOW":
        allowed_next_actions.extend(["execute under approved scope", "monitor change"])
    elif decision == "BLOCK":
        allowed_next_actions.extend(["deny action", "open incident review"])
    elif decision == "FORK":
        allowed_next_actions.extend(["route to human reviewer", "run in shadow workflow only"])
    elif decision == "QUARANTINE":
        allowed_next_actions.extend(["hold action", "collect missing evidence", "preserve forensic context"])
    else:
        allowed_next_actions.append("request valid supervisor decision")

    evidence_needed = [item.get("ask", "") for item in evidence_plan] or missing_evidence
    return {
        "outcome": outcome,
        "remediation_steps": remediation_steps,
        "rollback_required": rollback_required,
        "human_review_required": human_review_required,
        "monitoring_required": monitoring_required,
        "explanation": (
            "Outcome is based on decision, risk indicators, trusted evidence, "
            "environment, and ordered evidence gaps."
        ),
        "evidence_needed": evidence_needed,
        "allowed_next_actions": allowed_next_actions,
    }
