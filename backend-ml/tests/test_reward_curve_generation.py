from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.generate_reward_curves import (  # noqa: E402
    CURVE_REPORT_MD,
    collect_curve_data,
    generate_reward_curves,
    parse_trainer_state,
)


def test_parse_trainer_state_reads_common_metrics(tmp_path: Path) -> None:
    trainer_state = tmp_path / "trainer_state.json"
    trainer_state.write_text(
        json.dumps(
            {
                "log_history": [
                    {
                        "step": 1,
                        "loss": 0.7,
                        "rewards/reward_fn/mean": 1.2,
                        "rewards/reward_fn/std": 0.3,
                        "grad_norm": 2.0,
                        "completions/mean_length": 1.0,
                        "invalid_output_rate": 0.04,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_trainer_state(trainer_state)

    assert parsed["points"]["reward_mean"][0]["value"] == 1.2
    assert parsed["points"]["loss"][0]["value"] == 0.7
    assert parsed["points"]["grad_norm"][0]["value"] == 2.0


def test_collect_curve_data_handles_missing_logs_without_faking(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"

    data = collect_curve_data(paths=[missing])

    assert data["source_files_used"] == []
    assert data["metrics_found"]["reward_mean"] == 0
    assert "reward_mean" in data["missing_metrics"]


def test_generate_reward_curves_writes_report_and_comparison_plot(tmp_path: Path) -> None:
    report = generate_reward_curves(tmp_path)

    assert CURVE_REPORT_MD.exists()
    assert any(path.endswith("model_policy_comparison.png") for path in report["generated_plot_paths"])
    assert any(path.endswith("safety_reward_comparison.png") for path in report["generated_plot_paths"])
    assert report["warning"] is None or "No trainer_state.json" in report["warning"]
