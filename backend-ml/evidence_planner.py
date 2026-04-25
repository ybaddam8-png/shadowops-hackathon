"""Evidence planning helpers for ShadowOps policy decisions."""

from __future__ import annotations

from typing import Iterable


DOMAIN_REQUIRED_EVIDENCE = {
    "github_ci": [
        "maintainer identity",
        "package provenance",
        "CI workflow diff",
        "secret permission diff",
        "approved change ticket",
    ],
    "aws_s3": [
        "data classification",
        "bucket policy diff",
        "public access justification",
        "approval ticket",
        "exposure duration",
    ],
    "iam": [
        "approval chain",
        "requester identity",
        "privilege diff",
        "break-glass justification",
        "MFA status",
    ],
    "network_firewall": [
        "change ticket",
        "approved maintenance window",
        "source/destination justification",
        "exposed port risk",
        "rollback plan",
    ],
    "pentest": [
        "authorization letter",
        "approved test window",
        "scope document",
        "target list",
        "emergency contact",
    ],
}


def _norm(value: str) -> str:
    return " ".join(str(value).replace("_", " ").replace("-", " ").lower().split())


def _provided_tokens(provided_evidence: Iterable[str]) -> set[str]:
    tokens = set()
    for item in provided_evidence or []:
        normalized = _norm(item)
        tokens.add(normalized)
        words = normalized.split()
        for size in range(2, min(5, len(words)) + 1):
            for start in range(0, len(words) - size + 1):
                tokens.add(" ".join(words[start : start + size]))
    return tokens


def get_required_evidence(
    domain: str,
    indicators: Iterable[str],
    decision: str,
    environment: str,
) -> list[str]:
    domain_key = _norm(domain).replace(" ", "_")
    required = list(DOMAIN_REQUIRED_EVIDENCE.get(domain_key, []))
    indicator_text = " ".join(_norm(item) for item in indicators or [])
    env = _norm(environment)
    decision = str(decision or "").upper()

    if "secret" in indicator_text and "secret permission diff" not in required:
        required.append("secret permission diff")
    if "admin" in indicator_text and "approval chain" not in required:
        required.append("approval chain")
    if "public" in indicator_text and "public access justification" not in required:
        required.append("public access justification")
    if "data export" in indicator_text and "data classification" not in required:
        required.append("data classification")
    if env == "production" and decision in {"ALLOW", "FORK"}:
        required.append("production approval")

    return list(dict.fromkeys(required))


def get_missing_evidence(required_evidence: Iterable[str], provided_evidence: Iterable[str]) -> list[str]:
    provided = _provided_tokens(provided_evidence)
    missing = []
    for item in required_evidence or []:
        normalized = _norm(item)
        if normalized not in provided and not any(normalized in token or token in normalized for token in provided):
            missing.append(item)
    return missing


def explain_evidence_gap(missing_evidence: Iterable[str]) -> str:
    missing = list(missing_evidence or [])
    if not missing:
        return "Required evidence is present."
    if len(missing) == 1:
        return f"Missing required evidence: {missing[0]}."
    return "Missing required evidence: " + ", ".join(missing) + "."
