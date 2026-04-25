from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.build_action_only_sft_dataset import build_action_only_dataset
from training.submission_values import build_submission_summary


def test_action_only_dataset_uses_exact_labels() -> None:
    rows, audit = build_action_only_dataset()

    assert rows
    assert audit["invalid_label_count"] == 0
    assert set(audit["action_distribution"]["train"]).issubset({"ALLOW", "BLOCK", "FORK", "QUARANTINE"})


def test_submission_values_mark_missing_metrics_pending() -> None:
    summary = build_submission_summary()

    assert "benchmark_metrics" in summary
    assert summary["note"].startswith("Missing values")
