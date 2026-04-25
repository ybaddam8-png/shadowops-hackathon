"""Generate README-ready judge artifacts without loading a model."""

from __future__ import annotations

import json
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

from training.openenv_eval import generate_openenv_report
from training.shadowops_training_common import (
    DEFAULT_DEMO_BENCHMARK_JSON,
    DEFAULT_MODEL_POLICY_COMPARISON_JSON,
    DEFAULT_MODEL_POLICY_COMPARISON_MD,
    DEFAULT_VAL_DATASET_PATH,
    load_validation_samples_for_benchmark,
    run_demo_benchmark,
    run_model_policy_comparison,
    run_reward_diagnostics,
    write_json,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BACKEND_DIR))
    except ValueError:
        return str(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_benchmark_table(report: dict[str, Any], output_dir: Path) -> None:
    rows = report.get("metrics") or []
    lines = [
        "# ShadowOps Benchmark Table",
        "",
        f"Validation source: `{DEFAULT_VAL_DATASET_PATH.relative_to(BACKEND_DIR)}`",
        "",
        "| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean | invalid_output_rate | zero_std_reward_group_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    diagnostics = _read_json(output_dir / "reward_diagnostics.json", {})
    zero_std = float(diagnostics.get("frac_reward_zero_std", 0.0) or 0.0)
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['exact_match']:.3f} | {row['safety_accuracy']:.3f} | "
            f"{row['unsafe_decision_rate']:.3f} | {row['false_positive_rate']:.3f} | "
            f"{row['reward_mean']:.3f} | 0.000 | {zero_std:.3f} |"
        )
    (output_dir / "benchmark_table.md").write_text("\n".join(lines), encoding="utf-8")


def _write_reward_diagnostics(diagnostics: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "reward_diagnostics.json", diagnostics)
    lines = [
        "# ShadowOps Reward Diagnostics",
        "",
        f"- Samples: {diagnostics['sample_count']}",
        f"- Reward mean/std: {diagnostics['reward_mean']:.3f} / {diagnostics['reward_std']:.3f}",
        f"- Zero-std reward groups: {diagnostics['percent_zero_std_groups']:.1f}%",
        f"- Invalid output rate: {diagnostics['invalid_output_rate']:.3f}",
        "",
        "## Action Distribution",
        "",
    ]
    for action, value in diagnostics["action_distribution"].items():
        lines.append(f"- {action}: {value:.3f}")
    (output_dir / "reward_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def _write_reward_curve_note(output_dir: Path) -> None:
    candidate_logs = [
        BACKEND_DIR / "reward_curves_qwen3.json",
        TRAINING_DIR / "reward_curves.json",
    ]
    existing = [path for path in candidate_logs if path.exists()]
    lines = ["# Reward Curve Artifact", ""]
    if not existing:
        lines.append("No real training logs were found. Reward curve PNG is intentionally not generated.")
    else:
        png_path = output_dir / "reward_curve.png"
        _write_reward_curve_png(existing[0], png_path)
        lines.append("Real training log candidates were found:")
        lines.extend(f"- `{path.relative_to(BACKEND_DIR)}`" for path in existing)
        lines.append("")
        lines.append(f"Generated PNG plot: `{png_path.name}`")
    (output_dir / "reward_curve_status.md").write_text("\n".join(lines), encoding="utf-8")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png_rgb(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> None:
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def _draw_line(
    pixels: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    width = len(pixels[0])
    height = len(pixels)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _extract_reward_series(log_payload: dict[str, Any]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    baselines = log_payload.get("baselines", {})
    for name in ("random", "heuristic", "quarantine_aware"):
        rewards = baselines.get(name, {}).get("rewards")
        if isinstance(rewards, list) and rewards:
            series[name] = [float(value) for value in rewards[:120]]
    for name in ("train", "validation", "grpo_val_curve"):
        payload = log_payload.get(name, {})
        rewards = payload.get("rewards") if isinstance(payload, dict) else payload
        if isinstance(rewards, list) and rewards:
            series[name] = [float(value) for value in rewards[:120]]
    return series


def _write_reward_curve_png(log_path: Path, output_path: Path) -> None:
    payload = _read_json(log_path, {})
    series = _extract_reward_series(payload)
    width, height = 720, 420
    margin = 42
    pixels = [[(250, 252, 255) for _ in range(width)] for _ in range(height)]
    axis_color = (80, 90, 110)
    for x in range(margin, width - margin):
        pixels[height - margin][x] = axis_color
    for y in range(margin, height - margin):
        pixels[y][margin] = axis_color
    if not series:
        _write_png_rgb(output_path, width, height, pixels)
        return
    values = [value for rewards in series.values() for value in rewards]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1.0)
    colors = {
        "random": (220, 70, 70),
        "heuristic": (40, 120, 220),
        "quarantine_aware": (40, 160, 95),
        "train": (230, 145, 35),
        "validation": (120, 80, 200),
        "grpo_val_curve": (120, 80, 200),
    }
    plot_width = width - margin * 2
    plot_height = height - margin * 2
    for name, rewards in series.items():
        if len(rewards) < 2:
            continue
        color = colors.get(name, (40, 40, 40))
        points = []
        for index, value in enumerate(rewards):
            x = margin + int(index * plot_width / max(len(rewards) - 1, 1))
            y = height - margin - int((value - min_value) * plot_height / span)
            points.append((x, y))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            _draw_line(pixels, x0, y0, x1, y1, color)
    _write_png_rgb(output_path, width, height, pixels)


def generate_report_artifacts(output_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = _read_json(DEFAULT_DEMO_BENCHMARK_JSON, None)
    if benchmark is None:
        benchmark = run_demo_benchmark()
    val_samples, _ = load_validation_samples_for_benchmark()
    diagnostics = run_reward_diagnostics(val_samples)
    _write_reward_diagnostics(diagnostics, output_dir)
    _write_benchmark_table(benchmark, output_dir)

    comparison = run_model_policy_comparison(
        output_json=DEFAULT_MODEL_POLICY_COMPARISON_JSON,
        output_md=DEFAULT_MODEL_POLICY_COMPARISON_MD,
    )
    shutil.copyfile(DEFAULT_MODEL_POLICY_COMPARISON_JSON, output_dir / "model_policy_comparison.json")
    shutil.copyfile(DEFAULT_MODEL_POLICY_COMPARISON_MD, output_dir / "model_policy_comparison.md")
    openenv_report = generate_openenv_report(output_dir)
    _write_reward_curve_note(output_dir)
    index = {
        "output_dir": _display_path(output_dir),
        "artifacts": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "benchmark_policy_count": len(benchmark.get("metrics", [])),
        "model_policy_datasets": sorted(comparison.get("datasets", {}).keys()),
        "openenv_loop": {
            "episodes": openenv_report["episodes"],
            "total_steps": openenv_report["total_steps"],
            "unsafe_allow_rate": openenv_report["unsafe_allow_rate"],
        },
    }
    write_json(output_dir / "artifact_index.json", index)
    return index


def main() -> int:
    index = generate_report_artifacts()
    print(f"Generated judge artifacts in {index['output_dir']}")
    for artifact in index["artifacts"]:
        print(f"  {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
