"""Strict and tolerant action parsing for ShadowOps model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


VALID_ACTIONS = ("ALLOW", "BLOCK", "FORK", "QUARANTINE")
VALID_ACTION_SET = set(VALID_ACTIONS)

ACTION_RE = re.compile(r"\b(ALLOW|BLOCK|FORK|QUARANTINE)\b", re.IGNORECASE)
ACTION_PREFIX_RE = re.compile(r"^\s*(?:action|decision|supervisor decision|recommended action)\s*[:=\-]\s*", re.IGNORECASE)
MAYBE_RE = re.compile(r"\b(maybe|possibly|probably|could|might|not sure)\b", re.IGNORECASE)
SAFE_PREFIX_RE = re.compile(
    r"^(?:action|decision|supervisor decision|recommended action|i recommend|recommendation)\b",
    re.IGNORECASE,
)

JSON_ACTION_KEYS = (
    "action",
    "decision",
    "supervisor_action",
    "supervisor_decision",
    "recommended_action",
)

TOLERANT_SYNONYMS = {
    "HUMAN REVIEW": "FORK",
    "HUMAN_REVIEW": "FORK",
    "MANUAL REVIEW": "FORK",
    "MANUAL_REVIEW": "FORK",
    "ESCALATE": "FORK",
    "REVIEW": "FORK",
    "ISOLATE": "QUARANTINE",
    "HOLD": "QUARANTINE",
    "APPROVE": "ALLOW",
    "APPROVED": "ALLOW",
    "DENY": "BLOCK",
    "DENIED": "BLOCK",
    "REJECT": "BLOCK",
    "REJECTED": "BLOCK",
}


@dataclass(frozen=True)
class ActionParseResult:
    raw: str
    valid: bool
    action: str | None
    mode: str
    reason: str
    mapped_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_phrase(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").split()).upper()


def _canonical_token(value: str, *, allow_synonyms: bool) -> tuple[str | None, str | None]:
    normalized = _normalize_phrase(value)
    if normalized in VALID_ACTION_SET:
        return normalized, None
    if allow_synonyms and normalized in TOLERANT_SYNONYMS:
        return TOLERANT_SYNONYMS[normalized], normalized
    return None, None


def parse_strict_train_action(output: Any) -> ActionParseResult:
    raw = "" if output is None else str(output)
    if raw in VALID_ACTION_SET:
        return ActionParseResult(raw=raw, valid=True, action=raw, mode="strict_train", reason="exact_action")
    if raw == "":
        return ActionParseResult(raw=raw, valid=False, action=None, mode="strict_train", reason="empty_output")
    return ActionParseResult(
        raw=raw,
        valid=False,
        action=None,
        mode="strict_train",
        reason="strict_mode_requires_exact_single_action_token",
    )


def _json_action(raw: str) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(
            r'"(?:action|decision|supervisor_action|supervisor_decision|recommended_action)"\s*:\s*"([^"]+)"',
            raw,
            flags=re.IGNORECASE,
        )
        if not match:
            return None, None
        return _canonical_token(match.group(1), allow_synonyms=True)

    if isinstance(payload, dict):
        for key in JSON_ACTION_KEYS:
            if key in payload:
                return _canonical_token(str(payload[key]), allow_synonyms=True)
    if isinstance(payload, list) and len(payload) == 1:
        return _canonical_token(str(payload[0]), allow_synonyms=True)
    return None, None


def _action_matches(raw: str) -> list[tuple[str, str, int, int]]:
    synonym_words = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in sorted(TOLERANT_SYNONYMS, key=len, reverse=True))
    pattern = re.compile(
        rf"\b(ALLOW|BLOCK|FORK|QUARANTINE|{synonym_words})\b",
        re.IGNORECASE,
    )
    matches: list[tuple[str, str, int, int]] = []
    for match in pattern.finditer(raw):
        action, mapped = _canonical_token(match.group(1), allow_synonyms=True)
        if action is not None:
            matches.append((action, mapped or action, match.start(), match.end()))
    return matches


def parse_tolerant_eval_action(output: Any) -> ActionParseResult:
    raw = "" if output is None else str(output)
    stripped = raw.strip()
    if not stripped:
        return ActionParseResult(raw=raw, valid=False, action=None, mode="tolerant_eval", reason="empty_output")

    json_action, json_mapped = _json_action(stripped)
    if json_action is not None:
        return ActionParseResult(
            raw=raw,
            valid=True,
            action=json_action,
            mode="tolerant_eval",
            reason="json_action",
            mapped_from=json_mapped,
        )

    direct, direct_mapped = _canonical_token(stripped.strip("`'\".,;:!?[](){}"), allow_synonyms=True)
    if direct is not None:
        return ActionParseResult(
            raw=raw,
            valid=True,
            action=direct,
            mode="tolerant_eval",
            reason="direct_action",
            mapped_from=direct_mapped,
        )

    if MAYBE_RE.search(stripped):
        return ActionParseResult(raw=raw, valid=False, action=None, mode="tolerant_eval", reason="ambiguous_uncertainty_text")

    matches = _action_matches(stripped)
    unique_actions = {match[0] for match in matches}
    if len(unique_actions) > 1:
        return ActionParseResult(raw=raw, valid=False, action=None, mode="tolerant_eval", reason="conflicting_actions")
    if not matches:
        return ActionParseResult(raw=raw, valid=False, action=None, mode="tolerant_eval", reason="no_action_found")

    action, mapped_from, start, end = matches[0]
    prefix = stripped[:start].strip(" \t\r\n`'\".,;:!?[](){}")
    suffix = stripped[end:].strip()
    if start == 0:
        if suffix and not suffix.lower().startswith(("because", "-", ":", ".")):
            return ActionParseResult(raw=raw, valid=False, action=None, mode="tolerant_eval", reason="untrusted_free_text_suffix")
        return ActionParseResult(raw=raw, valid=True, action=action, mode="tolerant_eval", reason="leading_action", mapped_from=None if mapped_from == action else mapped_from)
    if SAFE_PREFIX_RE.match(prefix):
        return ActionParseResult(
            raw=raw,
            valid=True,
            action=action,
            mode="tolerant_eval",
            reason="prefixed_action",
            mapped_from=None if mapped_from == action else mapped_from,
        )
    return ActionParseResult(raw=raw, valid=False, action=None, mode="tolerant_eval", reason="untrusted_free_text")


def parse_action_output(output: Any, mode: str = "tolerant_eval") -> ActionParseResult:
    if mode in {"strict_train", "strict_train_mode"}:
        return parse_strict_train_action(output)
    if mode in {"tolerant_eval", "tolerant_eval_mode"}:
        return parse_tolerant_eval_action(output)
    raise ValueError(f"Unknown action parser mode: {mode}")
