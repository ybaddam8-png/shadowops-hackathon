"""Generate reward and training comparison plots from real ShadowOps artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPO_ROOT = BACKEND_DIR.parent
PLOTS_DIR = TRAINING_DIR / "plots"
REPORTS_DIR = TRAINING_DIR / "reports"
CURVE_DATA_JSON = REPORTS_DIR / "reward_curve_data.json"
CURVE_REPORT_MD = REPORTS_DIR / "reward_curve_report.md"

CURVE_ALIASES = {
    "reward_mean": ("reward_mean", "reward", "eval_reward", "rewards/reward_fn/mean", "rewards/mean", "train/reward"),
    "loss": ("loss", "train_loss", "training_loss"),
    "reward_std": ("reward_std", "rewards/reward_fn/std", "rewards/std"),
    "grad_norm": ("grad_norm", "gradient_norm"),
    "completion_length": ("completions/mean_length", "completion_mean_length", "avg_completion_length"),
    "invalid_output_rate": ("invalid_output_rate", "parse_failure_rate", "invalid_action_rate"),
}

PLOT_NAMES = {
    "reward_mean": "reward_curve.png",
    "loss": "loss_curve.png",
    "reward_std": "reward_std_curve.png",
    "grad_norm": "grad_norm_curve.png",
    "completion_length": "completion_length_curve.png",
    "invalid_output_rate": "invalid_output_rate_curve.png",
}

COMPARISON_METRICS = (
    "exact_match",
    "safety_accuracy",
    "unsafe_decision_rate",
    "false_positive_rate",
    "reward_mean",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BACKEND_DIR))
    except ValueError:
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _metric(row: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        value = row.get(key)
        if _is_number(value):
            return float(value)
    return None


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_trainer_state(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    points: dict[str, list[dict[str, float]]] = {name: [] for name in CURVE_ALIASES}
    if not isinstance(data, dict):
        return {"source_file": _rel(path), "points": points, "point_count": 0}
    for index, row in enumerate(data.get("log_history", []) or []):
        if not isinstance(row, dict):
            continue
        step = row.get("step", index)
        if not _is_number(step):
            step = index
        for metric_name, aliases in CURVE_ALIASES.items():
            value = _metric(row, aliases)
            if value is not None:
                points[metric_name].append({"step": float(step), "value": value})
    return {
        "source_file": _rel(path),
        "points": points,
        "point_count": sum(len(values) for values in points.values()),
    }


def parse_reward_curve_file(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    points: dict[str, list[dict[str, float]]] = {name: [] for name in CURVE_ALIASES}
    if not isinstance(data, dict):
        return {"source_file": _rel(path), "points": points, "point_count": 0}
    for split in ("train", "val", "validation"):
        split_data = data.get(split)
        if not isinstance(split_data, dict):
            continue
        rewards = split_data.get("rewards")
        if isinstance(rewards, list):
            for index, value in enumerate(rewards):
                if _is_number(value):
                    points["reward_mean"].append({"step": float(index + 1), "value": float(value), "series": split})
    curve = data.get("grpo_val_curve")
    if isinstance(curve, list):
        for index, value in enumerate(curve):
            if _is_number(value):
                points["reward_mean"].append({"step": float(index + 1), "value": float(value), "series": "grpo_val_curve"})
            elif isinstance(value, dict):
                step = value.get("step", index + 1)
                reward = _metric(value, CURVE_ALIASES["reward_mean"])
                if reward is not None and _is_number(step):
                    points["reward_mean"].append({"step": float(step), "value": reward, "series": "grpo_val_curve"})
    return {
        "source_file": _rel(path),
        "points": points,
        "point_count": sum(len(values) for values in points.values()),
    }


def parse_report_eval_points(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    points: dict[str, list[dict[str, float]]] = {name: [] for name in CURVE_ALIASES}
    if not isinstance(data, dict):
        return {"source_file": _rel(path), "points": points, "point_count": 0}

    candidates: list[dict[str, Any]] = []
    if isinstance(data.get("model_metrics"), dict):
        candidates.append(data["model_metrics"])
    if isinstance(data.get("q_aware_baseline"), dict):
        candidates.append(data["q_aware_baseline"])
    if isinstance(data.get("metrics"), list):
        candidates.extend(row for row in data["metrics"] if isinstance(row, dict))
    for index, row in enumerate(candidates, start=1):
        for metric_name in ("reward_mean", "reward_std", "completion_length", "invalid_output_rate"):
            value = _metric(row, CURVE_ALIASES[metric_name])
            if value is not None:
                points[metric_name].append({"step": float(index), "value": value})
    return {
        "source_file": _rel(path),
        "points": points,
        "point_count": sum(len(values) for values in points.values()),
    }


def discover_curve_sources() -> list[Path]:
    sources: list[Path] = []
    sources.extend(TRAINING_DIR.rglob("trainer_state.json"))
    sources.extend(TRAINING_DIR.glob("reward_curves*.json"))
    sources.extend(BACKEND_DIR.glob("reward_curves*.json"))
    sources.extend(
        path
        for path in (
            REPORTS_DIR / "reward_diagnostics_report.json",
            REPORTS_DIR / "reward_diagnostics.json",
            TRAINING_DIR / "model_eval_report.json",
            REPORTS_DIR / "checkpoint_comparison_report.json",
            TRAINING_DIR / "demo_benchmark_report.json",
            TRAINING_DIR / "model_policy_comparison.json",
        )
        if path.exists()
    )
    return sorted(dict.fromkeys(path.resolve() for path in sources))


def collect_curve_data(paths: list[Path] | None = None) -> dict[str, Any]:
    paths = paths or discover_curve_sources()
    series: dict[str, list[dict[str, Any]]] = {name: [] for name in CURVE_ALIASES}
    source_files: list[str] = []
    source_summaries: list[dict[str, Any]] = []
    training_log_sources = 0
    for path in paths:
        if not path.exists():
            continue
        if path.name == "trainer_state.json":
            parsed = parse_trainer_state(path)
            training_log_sources += 1
        elif path.name.startswith("reward_curves"):
            parsed = parse_reward_curve_file(path)
        else:
            parsed = parse_report_eval_points(path)
        source_files.append(parsed["source_file"])
        source_summaries.append(parsed)
        for metric_name, points in parsed["points"].items():
            if points:
                series[metric_name].append({"source_file": parsed["source_file"], "points": points})
    missing_metrics = [name for name, values in series.items() if not values]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files_used": source_files,
        "source_summaries": source_summaries,
        "metrics_found": {name: sum(len(item["points"]) for item in values) for name, values in series.items()},
        "missing_metrics": missing_metrics,
        "series": series,
        "has_real_training_logs": training_log_sources > 0,
        "warning": None if training_log_sources else "No trainer_state.json logs found; training curves are pending unless another real curve artifact exists.",
    }


def load_policy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark = _read_json(TRAINING_DIR / "demo_benchmark_report.json")
    if isinstance(benchmark, dict):
        rows.extend(row for row in benchmark.get("metrics", []) if isinstance(row, dict))
    comparison = _read_json(REPORTS_DIR / "checkpoint_comparison_report.json")
    if isinstance(comparison, dict):
        for row in comparison.get("checkpoints", []):
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                merged = {"policy": row.get("name", "checkpoint")}
                merged.update(metrics)
                rows.append(merged)
    return [
        row
        for row in rows
        if row.get("policy") and any(_is_number(row.get(metric)) for metric in COMPARISON_METRICS)
    ]


def _plot_curve(path: Path, title: str, ylabel: str, series_items: list[dict[str, Any]]) -> bool:
    if not series_items:
        return False
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for item in series_items:
        points = item["points"]
        if not points:
            continue
        x = [float(point["step"]) for point in points]
        y = [float(point["value"]) for point in points]
        label = item["source_file"]
        if len(label) > 42:
            label = "..." + label[-39:]
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
    if not ax.lines:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_model_policy_comparison(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    plt = _matplotlib()
    labels = [str(row["policy"]) for row in rows]
    reward = [float(row.get("reward_mean", 0.0) or 0.0) for row in rows]
    exact = [float(row.get("exact_match", 0.0) or 0.0) for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar([i - width / 2 for i in x], exact, width, label="exact_match", color="#2563eb")
    ax2.bar([i + width / 2 for i in x], reward, width, label="reward_mean", color="#059669")
    ax1.set_xticks(x, labels, rotation=20, ha="right")
    ax1.set_ylabel("exact_match")
    ax2.set_ylabel("reward_mean")
    ax1.set_title("Policy/model comparison from real evaluation artifacts")
    ax1.grid(axis="y", alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_safety_reward_comparison(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    plt = _matplotlib()
    labels = [str(row["policy"]) for row in rows]
    metrics = list(COMPARISON_METRICS)
    x = list(range(len(labels)))
    width = 0.15
    colors = ["#2563eb", "#059669", "#dc2626", "#f97316", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for offset, metric in enumerate(metrics):
        values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
        positions = [index + (offset - 2) * width for index in x]
        ax.bar(positions, values, width, label=metric, color=colors[offset])
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Safety and reward comparison from real evaluation artifacts")
    ax.set_ylabel("metric value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def generate_reward_curves(output_dir: Path = PLOTS_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_curve_data()
    generated: list[str] = []
    skipped: list[str] = []
    for metric_name, filename in PLOT_NAMES.items():
        path = output_dir / filename
        made = _plot_curve(path, filename.replace("_", " ").replace(".png", "").title(), metric_name, data["series"][metric_name])
        if made:
            generated.append(_rel(path))
        else:
            skipped.append(filename)
    rows = load_policy_rows()
    for filename, plotter in (
        ("model_policy_comparison.png", _plot_model_policy_comparison),
        ("safety_reward_comparison.png", _plot_safety_reward_comparison),
    ):
        path = output_dir / filename
        if plotter(path, rows):
            generated.append(_rel(path))
        else:
            skipped.append(filename)
    data["generated_plot_paths"] = generated
    data["skipped_plots"] = skipped
    data["policy_rows_used"] = [row.get("policy") for row in rows]
    CURVE_DATA_JSON.write_text(json.dumps({k: v for k, v in data.items() if k != "series"}, indent=2), encoding="utf-8")
    write_markdown_report(data)
    return data


def write_markdown_report(data: dict[str, Any]) -> None:
    lines = [
        "# ShadowOps Reward Curve Report",
        "",
        "Plots are generated only from real training/evaluation artifacts.",
        "",
        f"Generated at: `{data['generated_at']}`",
        f"Training logs present: `{data['has_real_training_logs']}`",
    ]
    if data.get("warning"):
        lines.extend(["", f"Warning: {data['warning']}"])
    lines.extend(["", "## Generated Plots", ""])
    lines.extend(f"- `{path}`" for path in data.get("generated_plot_paths", []) or ["none"])
    lines.extend(["", "## Pending / Skipped Plots", ""])
    lines.extend(f"- `{name}`" for name in data.get("skipped_plots", []) or ["none"])
    lines.extend(["", "## Source Files", ""])
    lines.extend(f"- `{path}`" for path in data.get("source_files_used", []) or ["none"])
    lines.extend(["", "## Missing Metrics", ""])
    lines.extend(f"- `{metric}`" for metric in data.get("missing_metrics", []) or ["none"])
    CURVE_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ShadowOps reward/training plots from real artifacts.")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = generate_reward_curves(args.output_dir)
    print(f"Reward curve plots generated: {len(report['generated_plot_paths'])}")
    for path in report["generated_plot_paths"]:
        print(f"  {path}")
    print(f"Saved: {CURVE_DATA_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {CURVE_REPORT_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
