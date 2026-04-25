from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.shadowops_training_common import evaluate_policy_on_dataset, load_validation_samples_for_benchmark
from training.model_policy_gate import build_model_policy_gate_report


def test_qaware_policy_does_not_regress_on_validation() -> None:
    samples, _ = load_validation_samples_for_benchmark()

    metrics = evaluate_policy_on_dataset(samples, "q_aware", seed=3)

    assert metrics["exact_match"] >= 0.95
    assert metrics["safety_accuracy"] == 1.0
    assert metrics["unsafe_decision_rate"] == 0.0
    assert metrics["false_positive_rate"] <= 0.01
    assert metrics["reward_mean"] >= 1.866


def test_model_policy_gate_is_pending_without_real_checkpoint_metrics() -> None:
    report = build_model_policy_gate_report()

    assert report["deployment_recommendation"] in {"DEPLOY_QAWARE_POLICY", "DEPLOY_MODEL_CANDIDATE_WITH_QAWARE_GATE"}
    if report["best_model_candidate"] is None:
        assert report["gate_status"] == "PENDING_REAL_CHECKPOINT_EVAL"
