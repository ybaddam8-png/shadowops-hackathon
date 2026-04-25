"""Evidence planning helpers for ShadowOps policy decisions."""

from __future__ import annotations

from typing import Iterable, Any


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


EVIDENCE_PRIORITY_RULES = (
    (
        "critical",
        ("identity", "maintainer", "requester", "mfa", "authorization letter"),
        "Identity verification must happen before trusting the request.",
        "identity provider / repository owner / security lead",
        "15 minutes",
    ),
    (
        "high",
        ("approval", "ticket", "change", "break-glass", "break glass"),
        "Approval evidence determines whether this is legitimate work or policy bypass.",
        "ticketing system / incident commander",
        "30 minutes",
    ),
    (
        "high",
        ("privilege", "secret", "public", "bucket policy", "data classification", "exposed port"),
        "Privilege, secret, public exposure, and data impact evidence blocks unsafe execution.",
        "cloud audit logs / CI settings / data owner",
        "30 minutes",
    ),
    (
        "medium",
        ("scope", "window", "asset", "target", "source/destination", "duration"),
        "Scope, timing, and asset ownership reduce false positives for approved work.",
        "change request / asset inventory / pentest plan",
        "1 hour",
    ),
    (
        "low",
        ("rollback", "monitoring", "emergency contact"),
        "Rollback and monitoring evidence controls blast radius after approval.",
        "runbook / monitoring owner",
        "before execution",
    ),
)


def _classify_plan_item(evidence: str) -> tuple[str, str, str, str]:
    normalized = _norm(evidence)
    for priority, needles, reason, expected_source, timeout_hint in EVIDENCE_PRIORITY_RULES:
        if any(needle in normalized for needle in needles):
            return priority, reason, expected_source, timeout_hint
    return (
        "medium",
        "This evidence is required before the supervisor can reduce uncertainty.",
        "request owner",
        "before execution",
    )


def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    normalized = _norm(item["ask"])
    category_order = 9
    ordered_groups = (
        ("identity", ("identity", "maintainer", "requester", "mfa", "authorization letter")),
        ("approval", ("approval", "ticket", "change", "break-glass", "break glass")),
        ("impact", ("privilege", "secret", "public", "bucket policy", "data classification", "exposed port")),
        ("scope", ("scope", "window", "asset", "target", "source/destination", "duration")),
        ("rollback", ("rollback", "monitoring", "emergency contact")),
    )
    for index, (_, needles) in enumerate(ordered_groups):
        if any(needle in normalized for needle in needles):
            category_order = index
            break
    return (category_order * 10 + priority_order.get(item["priority"], 9), normalized)


def build_evidence_plan(
    domain: str,
    indicators: Iterable[str],
    decision: str,
    environment: str,
    missing_evidence: Iterable[str],
    memory_context: dict[str, Any] | None = None,
    risk_score: float | None = None,
) -> list[dict[str, Any]]:
    """Build an ordered evidence plan from flat missing evidence.

    Ordering is intentionally deterministic for demos and tests:
    identity -> approval -> privilege/secret/public exposure -> scope/window -> rollback.
    """

    memory_context = memory_context or {}
    missing = list(dict.fromkeys(missing_evidence or []))
    plan: list[dict[str, Any]] = []
    high_risk = (risk_score or 0.0) >= 0.60 or bool(memory_context.get("risky_chains"))
    decision = str(decision or "").upper()

    for evidence in missing:
        priority, reason, expected_source, timeout_hint = _classify_plan_item(evidence)
        blocks_decision = priority in {"critical", "high"} or decision in {"ALLOW", "FORK"} or high_risk
        ask = f"Provide {evidence} for {domain} {environment} action."
        if evidence.lower() == "mfa status":
            ask = "Verify MFA status for the requesting identity."
        elif "identity" in evidence.lower() or "maintainer" in evidence.lower():
            ask = f"Verify {evidence} before reviewing approval evidence."
        elif "rollback" in evidence.lower():
            ask = "Provide rollback plan and monitoring owner before exposure is allowed."
        plan.append(
            {
                "step": 0,
                "ask": ask,
                "priority": priority,
                "reason": reason,
                "blocks_decision": blocks_decision,
                "expected_source": expected_source,
                "timeout_hint": timeout_hint,
            }
        )

    plan.sort(key=_sort_key)
    for index, item in enumerate(plan, start=1):
        item["step"] = index
    return plan
