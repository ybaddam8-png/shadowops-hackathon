"""Generate reward curve artifacts from real training logs only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
PLOTS_DIR = TRAINING_DIR / "plots"

REAL_LOG_CANDIDATES = (
    TRAINING_DIR / "trainer_state.json",
    TRAINING_DIR / "metrics.jsonl",
    REPORTS_DIR / "reward_curve_data.json",
    REPORTS_DIR / "model_eval_report.json",
    TRAINING_DIR / "model_eval_report.json",
)


@dataclass
class CurveSeries:
    steps: list[float]
    reward_mean: list[float]
    reward_std: list[float]
    completion_length: list[float]
    invalid_output_rate: list[float]
    source_files: list[str]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_from_trainer_state(path: Path, data: dict[str, Any]) -> CurveSeries | None:
    history = data.get("log_history")
    if not isinstance(history, list) or not history:
        return None
    steps: list[float] = []
    reward_mean: list[float] = []
    reward_std: list[float] = []
    completion_length: list[float] = []
    invalid_output_rate: list[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        step = _safe_float(row.get("step"))
        if step is None:
            continue
        mean = _safe_float(row.get("reward_mean") or row.get("train/reward_mean"))
        if mean is None:
            continue
        steps.append(step)
        reward_mean.append(mean)
        reward_std.append(_safe_float(row.get("reward_std") or row.get("train/reward_std") or 0.0) or 0.0)
        completion_length.append(
            _safe_float(row.get("completion_length") or row.get("train/completion_length") or 0.0) or 0.0
        )
        invalid_output_rate.append(
            _safe_float(row.get("invalid_output_rate") or row.get("train/invalid_output_rate") or 0.0) or 0.0
        )
    if not steps:
        return None
    return CurveSeries(steps, reward_mean, reward_std, completion_length, invalid_output_rate, [str(path)])


def _extract_from_metrics_jsonl(path: Path) -> CurveSeries | None:
    if not path.exists():
        return None
    steps: list[float] = []
    reward_mean: list[float] = []
    reward_std: list[float] = []
    completion_length: list[float] = []
    invalid_output_rate: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        step = _safe_float(row.get("step"))
        mean = _safe_float(row.get("reward_mean") or row.get("train/reward_mean"))
        if step is None or mean is None:
            continue
        steps.append(step)
        reward_mean.append(mean)
        reward_std.append(_safe_float(row.get("reward_std") or row.get("train/reward_std") or 0.0) or 0.0)
        completion_length.append(
            _safe_float(row.get("completion_length") or row.get("train/completion_length") or 0.0) or 0.0
        )
        invalid_output_rate.append(
            _safe_float(row.get("invalid_output_rate") or row.get("train/invalid_output_rate") or 0.0) or 0.0
        )
    if not steps:
        return None
    return CurveSeries(steps, reward_mean, reward_std, completion_length, invalid_output_rate, [str(path)])


def _extract_from_reward_curve_data(path: Path, data: dict[str, Any]) -> CurveSeries | None:
    steps_raw = data.get("steps")
    reward_raw = data.get("reward_mean")
    if not isinstance(steps_raw, list) or not isinstance(reward_raw, list) or not steps_raw or not reward_raw:
        return None
    n = min(len(steps_raw), len(reward_raw))
    steps: list[float] = []
    reward_mean: list[float] = []
    reward_std: list[float] = []
    completion_length: list[float] = []
    invalid_output_rate: list[float] = []
    std_raw = data.get("reward_std") if isinstance(data.get("reward_std"), list) else []
    comp_raw = data.get("completion_length") if isinstance(data.get("completion_length"), list) else []
    invalid_raw = data.get("invalid_output_rate") if isinstance(data.get("invalid_output_rate"), list) else []
    for idx in range(n):
        step = _safe_float(steps_raw[idx])
        mean = _safe_float(reward_raw[idx])
        if step is None or mean is None:
            continue
        steps.append(step)
        reward_mean.append(mean)
        reward_std.append(_safe_float(std_raw[idx]) if idx < len(std_raw) else 0.0 or 0.0)
        completion_length.append(_safe_float(comp_raw[idx]) if idx < len(comp_raw) else 0.0 or 0.0)
        invalid_output_rate.append(_safe_float(invalid_raw[idx]) if idx < len(invalid_raw) else 0.0 or 0.0)
    if not steps:
        return None
    return CurveSeries(steps, reward_mean, reward_std, completion_length, invalid_output_rate, [str(path)])


def collect_real_curve_series() -> tuple[CurveSeries | None, list[str]]:
    notes: list[str] = []
    trainer_path = TRAINING_DIR / "trainer_state.json"
    if trainer_path.exists():
        extracted = _extract_from_trainer_state(trainer_path, _read_json(trainer_path))
        if extracted:
            return extracted, notes
        notes.append("trainer_state.json present but did not include usable reward history.")

    metrics_path = TRAINING_DIR / "metrics.jsonl"
    extracted = _extract_from_metrics_jsonl(metrics_path)
    if extracted:
        return extracted, notes
    if metrics_path.exists():
        notes.append("metrics.jsonl present but did not include usable reward history.")

    curve_data_path = REPORTS_DIR / "reward_curve_data.json"
    if curve_data_path.exists():
        extracted = _extract_from_reward_curve_data(curve_data_path, _read_json(curve_data_path))
        if extracted:
            return extracted, notes
        notes.append("reward_curve_data.json present but was not usable for plotting.")

    notes.append("PENDING_REAL_TRAINING_LOGS")
    return None, notes


def _plot_line(path: Path, x: list[float], y: list[float], title: str, y_label: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_bar(path: Path, labels: list[str], values: list[float], title: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(labels, values, color=["#2563eb", "#059669"])
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _non_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_policy_comparison() -> tuple[dict[str, Any], str]:
    candidates = [
        TRAINING_DIR / "model_policy_comparison.json",
        REPORTS_DIR / "model_policy_comparison.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_json(candidate), str(candidate)
    return {}, "missing"


def _load_reward_diagnostics() -> tuple[dict[str, Any], str]:
    candidates = [
        TRAINING_DIR / "reward_diagnostics_report.json",
        REPORTS_DIR / "reward_diagnostics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_json(candidate), str(candidate)
    return {}, "missing"


def _write_pending_stub_pngs() -> None:
    x = [0.0, 1.0]
    y = [0.0, 0.0]
    _plot_line(PLOTS_DIR / "reward_curve.png", x, y, "Reward Curve (Pending Real Logs)", "Reward mean", "#6b7280")
    _plot_line(PLOTS_DIR / "reward_std_curve.png", x, y, "Reward Std (Pending Real Logs)", "Reward std", "#6b7280")
    _plot_line(
        PLOTS_DIR / "completion_length_curve.png",
        x,
        y,
        "Completion Length (Pending Real Logs)",
        "Completion length",
        "#6b7280",
    )
    _plot_line(
        PLOTS_DIR / "invalid_output_rate_curve.png",
        x,
        y,
        "Invalid Output Rate (Pending Real Logs)",
        "Invalid output rate",
        "#6b7280",
    )


def generate_reward_curves() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    series, notes = collect_real_curve_series()

    if series is None:
        _write_pending_stub_pngs()
        status = "PENDING_REAL_TRAINING_LOGS"
        payload = {
            "status": status,
            "notes": notes,
            "required_real_log_candidates": [str(path.relative_to(BACKEND_DIR)) for path in REAL_LOG_CANDIDATES],
            "source_files": [],
            "steps": [],
            "reward_mean": [],
            "reward_std": [],
            "completion_length": [],
            "invalid_output_rate": [],
        }
    else:
        _plot_line(PLOTS_DIR / "reward_curve.png", series.steps, series.reward_mean, "Reward Mean Curve", "Reward mean", "#2563eb")
        _plot_line(PLOTS_DIR / "reward_std_curve.png", series.steps, series.reward_std, "Reward Std Curve", "Reward std", "#7c3aed")
        _plot_line(
            PLOTS_DIR / "completion_length_curve.png",
            series.steps,
            series.completion_length,
            "Completion Length Curve",
            "Completion length",
            "#059669",
        )
        _plot_line(
            PLOTS_DIR / "invalid_output_rate_curve.png",
            series.steps,
            series.invalid_output_rate,
            "Invalid Output Rate Curve",
            "Invalid output rate",
            "#dc2626",
        )
        status = "PASS"
        payload = {
            "status": status,
            "notes": notes,
            "required_real_log_candidates": [str(path.relative_to(BACKEND_DIR)) for path in REAL_LOG_CANDIDATES],
            "source_files": [str(Path(path).relative_to(BACKEND_DIR)) for path in series.source_files],
            "steps": series.steps,
            "reward_mean": series.reward_mean,
            "reward_std": series.reward_std,
            "completion_length": series.completion_length,
            "invalid_output_rate": series.invalid_output_rate,
        }

    policy_payload, policy_source = _load_policy_comparison()
    diagnostics_payload, diagnostics_source = _load_reward_diagnostics()
    labels = ["model", "q_aware"]
    model_reward = _safe_float(policy_payload.get("model", {}).get("reward_mean"))
    q_reward = _safe_float(policy_payload.get("q_aware", {}).get("reward_mean"))
    if model_reward is None or q_reward is None:
        labels = ["pending", "q_aware"]
        model_reward = 0.0
        q_reward = q_reward or 0.0
    _plot_bar(PLOTS_DIR / "model_policy_comparison.png", labels, [model_reward, q_reward], "Model vs Q-aware Reward", "Reward mean")

    safety = _safe_float(diagnostics_payload.get("safety_reward"))
    if safety is None:
        safety = 0.0
    baseline = _safe_float(diagnostics_payload.get("reward_mean")) or 0.0
    _plot_bar(
        PLOTS_DIR / "safety_reward_comparison.png",
        ["safety_component", "reward_mean"],
        [safety, baseline],
        "Safety Reward Comparison",
        "Reward component",
    )

    payload["comparison_sources"] = {
        "model_policy_comparison": policy_source,
        "reward_diagnostics": diagnostics_source,
    }
    payload["plots"] = [
        "training/plots/reward_curve.png",
        "training/plots/reward_std_curve.png",
        "training/plots/completion_length_curve.png",
        "training/plots/invalid_output_rate_curve.png",
        "training/plots/model_policy_comparison.png",
        "training/plots/safety_reward_comparison.png",
    ]
    payload["plot_non_empty"] = {
        plot_name: _non_empty(BACKEND_DIR / plot_name) for plot_name in payload["plots"]
    }

    report_json = REPORTS_DIR / "reward_curve_data.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md = REPORTS_DIR / "reward_curve_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Reward Curve Report",
                "",
                f"- status: `{payload['status']}`",
                f"- source files: {', '.join(payload['source_files']) if payload['source_files'] else 'none'}",
                f"- notes: {', '.join(payload['notes']) if payload['notes'] else 'none'}",
                "",
                "## Plot Outputs",
                "",
                *[f"- `{plot}`" for plot in payload["plots"]],
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    payload = generate_reward_curves()
    print(f"reward_curve_status={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
