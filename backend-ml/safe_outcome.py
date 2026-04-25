"""Operational safe-outcome generator for ShadowOps decisions."""

from __future__ import annotations

from typing import Iterable


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
