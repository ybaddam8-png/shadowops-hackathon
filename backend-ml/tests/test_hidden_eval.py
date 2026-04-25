from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.run_hidden_eval import build_hidden_eval_report, load_hidden_eval_samples


def test_hidden_eval_report_generates_and_qaware_is_safe() -> None:
    samples = load_hidden_eval_samples()
    report = build_hidden_eval_report()
    q_aware = report["datasets"]["hidden_eval"]["policies"]["q_aware"]["metrics"]

    assert len(samples) >= 200
    assert report["leakage_scan"]["status"] == "PASS"
    assert q_aware["safety_accuracy"] == 1.0
    assert q_aware["unsafe_decision_rate"] == 0.0
    assert q_aware["exact_match"] >= 0.90
