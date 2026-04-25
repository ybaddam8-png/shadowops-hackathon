"""Compare SFT/GRPO run artifacts without loading models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR_FOR_IMPORT))

from training.generate_reward_curves import (
    BACKEND_DIR,
    PLOTS_DIR,
    _matplotlib,
    _read_json,
    _rel,
    parse_reward_curve_file,
    parse_trainer_state,
)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name or "run", path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def _run_metrics(path: Path) -> dict[str, Any] | None:
    candidates = [
        path / "model_eval_report.json",
        path / "training_metrics.json",
        path / "eval_metrics.json",
    ]
    for candidate in candidates:
        data = _read_json(candidate)
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("model_metrics"), dict):
            return data["model_metrics"]
        return data
    return None


def collect_run(name: str, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else (BACKEND_DIR / path)
    row = {
        "name": name,
        "path": _rel(resolved),
        "exists": resolved.exists(),
        "warnings": [],
        "curves": {},
        "metrics": None,
    }
    if not resolved.exists():
        row["warnings"].append(f"Run folder not found: {_rel(resolved)}")
        return row
    trainer_state = resolved / "trainer_state.json"
    if trainer_state.exists():
        row["curves"] = parse_trainer_state(trainer_state)["points"]
    reward_curves = resolved / "reward_curves.json"
    if reward_curves.exists():
        parsed = parse_reward_curve_file(reward_curves)["points"]
        for key, points in parsed.items():
            row["curves"].setdefault(key, [])
            row["curves"][key].extend(points)
    row["metrics"] = _run_metrics(resolved)
    if not row["curves"] and not row["metrics"]:
        row["warnings"].append("No trainer_state.json, reward_curves.json, model_eval_report.json, or training_metrics.json found.")
    return row


def _plot_metric(path: Path, title: str, ylabel: str, runs: list[dict[str, Any]], metric_name: str) -> bool:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for run in runs:
        points = (run.get("curves") or {}).get(metric_name, [])
        if not points:
            continue
        ax.plot(
            [float(point["step"]) for point in points],
            [float(point["value"]) for point in points],
            marker="o",
            label=run["name"],
        )
    if not ax.lines:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_eval(path: Path, runs: list[dict[str, Any]]) -> bool:
    metrics = [run for run in runs if isinstance(run.get("metrics"), dict)]
    if not metrics:
        return False
    plt = _matplotlib()
    labels = [run["name"] for run in metrics]
    reward = [float((run["metrics"] or {}).get("reward_mean", 0.0) or 0.0) for run in metrics]
    safety = [float((run["metrics"] or {}).get("safety_accuracy", 0.0) or 0.0) for run in metrics]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - width / 2 for i in x], safety, width, label="safety_accuracy", color="#059669")
    ax.bar([i + width / 2 for i in x], reward, width, label="reward_mean", color="#7c3aed")
    ax.set_xticks(x, labels)
    ax.set_title("Run evaluation comparison from real artifacts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def compare_training_runs(run_args: list[str], output_dir: Path = PLOTS_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [collect_run(name, path) for name, path in (_parse_run(value) for value in run_args)]
    generated: list[str] = []
    skipped: list[str] = []
    for filename, metric, title, ylabel in (
        ("run_reward_comparison.png", "reward_mean", "Run reward comparison", "reward mean"),
        ("run_loss_comparison.png", "loss", "Run loss comparison", "training loss"),
    ):
        path = output_dir / filename
        if _plot_metric(path, title, ylabel, runs, metric):
            generated.append(_rel(path))
        else:
            skipped.append(filename)
    eval_path = output_dir / "run_eval_comparison.png"
    if _plot_eval(eval_path, runs):
        generated.append(_rel(eval_path))
    else:
        skipped.append("run_eval_comparison.png")
    report = {
        "runs": runs,
        "generated_plot_paths": generated,
        "skipped_plots": skipped,
        "warnings": [warning for run in runs for warning in run.get("warnings", [])],
    }
    (output_dir / "run_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ShadowOps SFT/GRPO training run artifacts.")
    parser.add_argument("--run", action="append", default=[], help="Run in name=path form.")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = compare_training_runs(args.run, args.output_dir)
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    print(f"Run comparison plots generated: {len(report['generated_plot_paths'])}")
    for path in report["generated_plot_paths"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
