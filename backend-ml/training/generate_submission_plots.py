"""Generate judge-ready submission plots from existing local reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
PLOTS_DIR = TRAINING_DIR / "plots"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.generate_reward_breakdown_report import (  # noqa: E402
    DEFAULT_REWARD_BREAKDOWN_JSON,
    generate_reward_breakdown_report,
)
from training.shadowops_training_common import DEFAULT_DEMO_BENCHMARK_JSON, run_demo_benchmark  # noqa: E402


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_benchmark() -> dict[str, Any]:
    report = _read_json(DEFAULT_DEMO_BENCHMARK_JSON, None)
    if report is None:
        report = run_demo_benchmark()
    return report


def _load_reward_breakdown() -> dict[str, Any]:
    # Regenerate to keep breakdown metrics aligned with the current benchmark code.
    return generate_reward_breakdown_report()


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_bar_plot(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#6b7280", "#2563eb", "#059669", "#7c3aed"][: len(values)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Policy baseline / pre-training validation")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_grouped_benchmark_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    plt = _matplotlib()
    labels = [row["policy"] for row in rows]
    exact = [float(row["exact_match"]) for row in rows]
    reward = [float(row["reward_mean"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar([i - width / 2 for i in x], exact, width=width, color="#2563eb", label="exact_match")
    ax2.bar([i + width / 2 for i in x], reward, width=width, color="#059669", label="reward_mean")
    ax1.set_title("Policy baseline / pre-training validation")
    ax1.set_ylabel("Exact match")
    ax2.set_ylabel("Reward mean")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_reward_breakdown_plot(path: Path, report: dict[str, Any]) -> None:
    plt = _matplotlib()
    policies = [policy["policy"] for policy in report["policies"]]
    components = [
        "action_correctness_reward",
        "safety_penalty",
        "false_positive_penalty",
        "evidence_completeness_reward",
        "uncertainty_handling_reward",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = list(range(len(policies)))
    width = 0.14
    colors = ["#2563eb", "#dc2626", "#f97316", "#059669", "#7c3aed"]
    for offset, component in enumerate(components):
        values = [float(policy["mean_components"].get(component, 0.0)) for policy in report["policies"]]
        positions = [index + (offset - 2) * width for index in x]
        ax.bar(positions, values, width=width, label=component.replace("_", " "), color=colors[offset])
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Reward breakdown: policy baseline / pre-training validation")
    ax.set_ylabel("Mean component value")
    ax.set_xticks(x, policies)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_submission_plots() -> dict[str, Any]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = _load_benchmark()
    breakdown = _load_reward_breakdown()
    rows = list(benchmark.get("metrics") or [])
    labels = [row["policy"] for row in rows]

    _save_grouped_benchmark_plot(PLOTS_DIR / "baseline_policy_comparison.png", rows)
    _save_bar_plot(
        PLOTS_DIR / "safety_accuracy_comparison.png",
        "Safety accuracy comparison",
        labels,
        [float(row["safety_accuracy"]) for row in rows],
        "Safety accuracy",
        (0, 1.05),
    )
    _save_bar_plot(
        PLOTS_DIR / "unsafe_decision_rate.png",
        "Unsafe decision rate",
        labels,
        [float(row["unsafe_decision_rate"]) for row in rows],
        "Unsafe decision rate",
        (0, max(0.25, max(float(row["unsafe_decision_rate"]) for row in rows) + 0.05)),
    )
    _save_reward_breakdown_plot(PLOTS_DIR / "reward_breakdown.png", breakdown)
    index = {
        "label": "Policy baseline / pre-training validation",
        "plots_dir": str(PLOTS_DIR.relative_to(BACKEND_DIR)),
        "plots": [
            "baseline_policy_comparison.png",
            "reward_breakdown.png",
            "safety_accuracy_comparison.png",
            "unsafe_decision_rate.png",
        ],
        "notes": "No trained model improvement is claimed by these plots.",
    }
    (PLOTS_DIR / "plot_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> int:
    index = generate_submission_plots()
    print(f"Submission plots saved to {index['plots_dir']}")
    for plot in index["plots"]:
        print(f"  {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
