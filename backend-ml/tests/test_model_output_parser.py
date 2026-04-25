from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.shadowops_training_common import analyze_action_output, parse_action


def test_model_output_parser_accepts_common_safe_formats() -> None:
    assert parse_action(" allow ") == "ALLOW"
    assert parse_action('{"action": "block"}') == "BLOCK"
    assert parse_action('{"decision": "human review", "explanation": "approval needed"}') == "FORK"
    assert parse_action("Action: quarantine because evidence is missing.") == "QUARANTINE"
    assert parse_action("I recommend isolate until the ticket is verified.") == "QUARANTINE"


def test_model_output_parser_rejects_ambiguous_or_dangerous_free_text() -> None:
    assert parse_action("please allow this because it is fine") is None
    assert parse_action("ALLOW then BLOCK") is None
    assert analyze_action_output("ALLOW then BLOCK")["multi_action_warning"] is True
    assert analyze_action_output("ALLOW then BLOCK")["invalid_output"] is True
