"""Merge real checkpoint metrics into submission summary artifacts.

This script does not train or load a model. It only reads a user-provided JSON
metrics file and writes README-ready summary files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
DOCS_DIR = BACKEND_DIR.parent / "docs"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.generate_submission_plots import generate_submission_plots  # noqa: E402
from training.shadowops_training_common import write_json  # noqa: E402


TEMPLATE_PATH = TRAINING_DIR / "training_results_template.json"
DEFAULT_OUTPUT_JSON = TRAINING_DIR / "real_training_results_summary.json"
DEFAULT_OUTPUT_MD = TRAINING_DIR / "real_training_results_summary.md"
DEFAULT_README_INSERT = DOCS_DIR / "TRAINING_RESULTS_INSERT.md"
REQUIRED_FIELDS = (
    "raw_model_exact",
    "sft_model_exact",
    "grpo_model_exact",
    "q_aware_exact",
    "raw_model_safety",
    "sft_model_safety",
    "grpo_model_safety",
    "q_aware_safety",
    "reward_curve_path",
    "loss_curve_path",
    "checkpoint_path",
    "training_gate_passed",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def merge_training_results(input_path: Path = TEMPLATE_PATH) -> dict[str, Any]:
    payload = _read_json(input_path)
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields in {input_path}: {', '.join(missing)}")
    has_real_model_metrics = any(payload.get(field) is not None for field in ("raw_model_exact", "sft_model_exact", "grpo_model_exact"))
    summary = {
        "source": str(input_path),
        "has_real_model_metrics": has_real_model_metrics,
        "training_gate_passed": bool(payload.get("training_gate_passed")),
        "metrics": {field: payload.get(field) for field in REQUIRED_FIELDS},
        "honest_claim": (
            "Real trained checkpoint metrics are available."
            if has_real_model_metrics
            else "Real trained checkpoint metrics are pending; Q-aware supervisor is the verified baseline."
        ),
    }
    return summary


def write_training_results(summary: dict[str, Any]) -> None:
    write_json(DEFAULT_OUTPUT_JSON, summary)
    metrics = summary["metrics"]
    lines = [
        "# ShadowOps Real Training Results Insert",
        "",
        summary["honest_claim"],
        "",
        "| Metric | Raw model | SFT model | GRPO model | Q-aware policy |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Exact match | {_format_value(metrics['raw_model_exact'])} | "
            f"{_format_value(metrics['sft_model_exact'])} | {_format_value(metrics['grpo_model_exact'])} | "
            f"{_format_value(metrics['q_aware_exact'])} |"
        ),
        (
            f"| Safety accuracy | {_format_value(metrics['raw_model_safety'])} | "
            f"{_format_value(metrics['sft_model_safety'])} | {_format_value(metrics['grpo_model_safety'])} | "
            f"{_format_value(metrics['q_aware_safety'])} |"
        ),
        "",
        f"- Checkpoint path: `{_format_value(metrics['checkpoint_path'])}`",
        f"- Reward curve path: `{_format_value(metrics['reward_curve_path'])}`",
        f"- Loss curve path: `{_format_value(metrics['loss_curve_path'])}`",
        f"- Training gate passed: `{summary['training_gate_passed']}`",
        "",
        "Do not claim model improvement unless this table contains real checkpoint metrics and the gate passed.",
    ]
    DEFAULT_OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    DEFAULT_README_INSERT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge real ShadowOps training metrics into submission artifacts.")
    parser.add_argument("--input", type=Path, default=TEMPLATE_PATH, help="JSON file containing real training metrics.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not regenerate baseline submission plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = merge_training_results(args.input)
    write_training_results(summary)
    if not args.skip_plots:
        generate_submission_plots()
    print(f"Saved: {DEFAULT_OUTPUT_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_OUTPUT_MD.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_README_INSERT.relative_to(BACKEND_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
