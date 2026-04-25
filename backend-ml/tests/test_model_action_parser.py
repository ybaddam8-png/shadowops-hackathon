from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_action_parser import parse_action_output


def test_strict_train_accepts_exact_action_only() -> None:
    result = parse_action_output("ALLOW", mode="strict_train")

    assert result.valid is True
    assert result.action == "ALLOW"
    assert result.reason == "exact_action"


def test_strict_train_rejects_non_exact_outputs() -> None:
    bad_outputs = [
        " allow ",
        "ALLOW.",
        '{"action":"ALLOW"}',
        "ALLOW because it is safe",
        "I recommend BLOCK",
        "",
    ]

    for output in bad_outputs:
        assert parse_action_output(output, mode="strict_train").valid is False


def test_tolerant_eval_accepts_safe_common_formats() -> None:
    assert parse_action_output("allow", mode="tolerant_eval").action == "ALLOW"
    assert parse_action_output('{"action": "ALLOW"}', mode="tolerant_eval").action == "ALLOW"
    assert parse_action_output("action: block", mode="tolerant_eval").action == "BLOCK"
    assert parse_action_output("Action: QUARANTINE because evidence is missing.", mode="tolerant_eval").action == "QUARANTINE"
    assert parse_action_output("isolate", mode="tolerant_eval").action == "QUARANTINE"
    assert parse_action_output("human review", mode="tolerant_eval").action == "FORK"
    assert parse_action_output("manual review", mode="tolerant_eval").action == "FORK"


def test_tolerant_eval_rejects_ambiguous_or_unrelated_text() -> None:
    assert parse_action_output("ALLOW then BLOCK", mode="tolerant_eval").valid is False
    assert parse_action_output("maybe allow", mode="tolerant_eval").valid is False
    assert parse_action_output("please allow because it is fine", mode="tolerant_eval").valid is False
    assert parse_action_output("hello", mode="tolerant_eval").valid is False
    assert parse_action_output("", mode="tolerant_eval").valid is False
