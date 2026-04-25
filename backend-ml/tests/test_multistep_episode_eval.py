from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.multistep_episode_eval import evaluate_multistep_episodes


def test_multistep_memory_affects_chain_decisions() -> None:
    report = evaluate_multistep_episodes()

    assert report["chain_detection_accuracy"] >= 0.80
    assert report["safe_chain_allow_rate"] >= 0.50
    assert report["malicious_chain_block_or_quarantine_rate"] >= 0.66
    assert report["memory_risk_improvement"] > 0.0
