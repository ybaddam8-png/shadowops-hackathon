"""Cumulative risk scoring for ShadowOps agent decisions."""

from __future__ import annotations

from typing import Any, Iterable


RISK_WEIGHTS = {
    "repeated_suspicious": 0.14,
    "risky_chain": 0.24,
    "production": 0.14,
    "secret_access": 0.18,
    "admin_privilege": 0.22,
    "public_exposure": 0.18,
    "data_export": 0.20,
    "unknown_actor": 0.12,
    "unknown_maintainer": 0.10,
}

TRUSTED_REDUCTIONS = {
    "approved_ticket": 0.14,
    "valid_pentest_window": 0.22,
    "trusted_maintainer": 0.18,
    "non_production": 0.14,
    "break_glass_approval": 0.18,
    "known_automation": 0.16,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _norm(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())


def _text_blob(values: Iterable[Any]) -> str:
    return " ".join(_norm(value) for value in values or [])


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _indicator_text(scenario: dict[str, Any]) -> str:
    return " ".join(
        [
            _norm(scenario.get("action_summary") or scenario.get("raw_payload")),
            _text_blob(scenario.get("indicators", [])),
            _text_blob(scenario.get("risk_indicators", [])),
        ]
    )


def _evidence_text(scenario: dict[str, Any]) -> str:
    return " ".join(
        [
            _text_blob(scenario.get("provided_evidence", [])),
            _text_blob(scenario.get("safe_indicators", [])),
            _norm(scenario.get("actor")),
        ]
    )


def compute_cumulative_risk(
    scenario: dict[str, Any],
    memory_context: dict[str, Any] | None = None,
    base_risk: float | None = None,
) -> dict[str, Any]:
    memory_context = memory_context or {}
    indicators = _indicator_text(scenario)
    evidence = _evidence_text(scenario)
    environment = _norm(scenario.get("environment") or "production")
    actor = _norm(scenario.get("actor") or "unknown")
    service = _norm(scenario.get("service") or scenario.get("domain") or "unknown")

    actor_risk_score = clamp(memory_context.get("actor_risk", 0.0))
    session_risk_score = clamp(memory_context.get("session_risk", 0.0))
    service_risk_score = clamp(memory_context.get("service_risk", 0.0))
    environment_risk_score = 0.30 if environment == "production" else 0.10 if environment in {"staging", "dev", "development"} else 0.20

    score = clamp(base_risk if base_risk is not None else scenario.get("risk_score", 0.0))
    reasons: list[str] = []

    repeated = len(set(scenario.get("indicators", []))) >= 3 or len(memory_context.get("recent_indicators", [])) >= 3
    risky_chains = list(memory_context.get("risky_chains", []))
    risk_conditions = {
        "repeated_suspicious": repeated,
        "risky_chain": bool(risky_chains),
        "production": environment == "production",
        "secret_access": _has(indicators, "secret", "credential", "token"),
        "admin_privilege": _has(indicators, "admin", "administratoraccess", "privilege escalation", "privilege"),
        "public_exposure": _has(indicators, "public", "0.0.0.0/0", "open port", "internet"),
        "data_export": _has(indicators, "export", "exfil", "transfer", "download all"),
        "unknown_actor": actor in {"", "unknown", "unknown actor"},
        "unknown_maintainer": _has(indicators, "unknown maintainer", "new contributor", "unknown actor"),
    }
    for key, active in risk_conditions.items():
        if active:
            score += RISK_WEIGHTS[key]
            reasons.append(key)

    trusted_conditions = {
        "approved_ticket": _has(evidence, "approved ticket", "approval ticket", "change ticket", "approved change"),
        "valid_pentest_window": _has(evidence, "valid pentest", "approved test window", "authorization letter"),
        "trusted_maintainer": _has(evidence, "trusted maintainer", "maintainer identity verified"),
        "non_production": environment in {"staging", "dev", "development"},
        "break_glass_approval": _has(evidence, "break glass approval", "break-glass approval", "incident commander"),
        "known_automation": _has(evidence, "known automation", "automation account", "deployment pattern"),
    }
    for key, active in trusted_conditions.items():
        if active:
            score -= TRUSTED_REDUCTIONS[key]
            reasons.append(f"trusted:{key}")

    cumulative = clamp(
        score * 0.72
        + actor_risk_score * 0.08
        + session_risk_score * 0.10
        + service_risk_score * 0.06
        + environment_risk_score * 0.04
    )

    return {
        "actor_risk_score": actor_risk_score,
        "session_risk_score": session_risk_score,
        "service_risk_score": service_risk_score,
        "environment_risk_score": environment_risk_score,
        "cumulative_risk_score": cumulative,
        "cumulative_risk_reason": ", ".join(reasons) if reasons else "No cumulative risk modifiers triggered.",
        "service": service,
        "actor": actor,
    }
