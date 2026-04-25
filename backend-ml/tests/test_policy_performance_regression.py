from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.shadowops_training_common import evaluate_policy_on_dataset, load_validation_samples_for_benchmark


def test_qaware_policy_does_not_regress_on_validation() -> None:
    samples, _ = load_validation_samples_for_benchmark()

    metrics = evaluate_policy_on_dataset(samples, "q_aware", seed=3)

    assert metrics["exact_match"] >= 0.95
    assert metrics["safety_accuracy"] == 1.0
    assert metrics["unsafe_decision_rate"] == 0.0
    assert metrics["false_positive_rate"] <= 0.01
    assert metrics["reward_mean"] >= 1.866
