from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.check_submission_artifacts import check_submission_artifacts
from training.generate_reward_curves import generate_reward_curves
from training.post_training_artifacts import generate_post_training_artifacts


def test_reward_curve_generator_does_not_fake_missing_metrics() -> None:
    report = generate_reward_curves()
    assert report["status"] in {"PASS", "PENDING_REAL_TRAINING_LOGS"}
    if report["status"] == "PENDING_REAL_TRAINING_LOGS":
        assert report["steps"] == []
        assert "PENDING_REAL_TRAINING_LOGS" in report["notes"]


def test_post_training_artifacts_runs_without_checkpoint() -> None:
    report = generate_post_training_artifacts()
    assert report["status"] in {"PASS", "WARN", "FAIL", "PENDING_REAL_TRAINING_LOGS"}
    assert (BACKEND_DIR / "training/reports/post_training_artifacts_report.json").exists()
    assert (BACKEND_DIR / "training/reports/post_training_artifacts_report.md").exists()


def test_artifact_checker_warns_for_pending_real_logs_not_fail() -> None:
    reward_data_path = BACKEND_DIR / "training/reports/reward_curve_data.json"
    if reward_data_path.exists():
        payload = json.loads(reward_data_path.read_text(encoding="utf-8"))
        if payload.get("status") == "PENDING_REAL_TRAINING_LOGS":
            report = check_submission_artifacts()
            assert report["status"] in {"WARN", "PASS"}


def test_gitignore_blocks_checkpoints_and_cache() -> None:
    gitignore = (BACKEND_DIR.parent / ".gitignore").read_text(encoding="utf-8")
    required_entries = [
        "backend-ml/training/checkpoints/",
        "backend-ml/training/cloud_runs/",
        "node_modules/",
        "dist/",
        "build/",
        ".venv/",
        "__pycache__/",
        "*.safetensors",
        "*.pt",
        "*.ckpt",
    ]
    for entry in required_entries:
        assert entry in gitignore


def test_cleanup_report_exists() -> None:
    assert (BACKEND_DIR / "training/reports/repo_cleanup_report.json").exists()
    assert (BACKEND_DIR / "training/reports/repo_cleanup_report.md").exists()
