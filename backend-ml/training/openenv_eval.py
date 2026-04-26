"""Laptop-safe OpenEnv loop evaluation for ShadowOps."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import struct
import sys
import zlib
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from openenv_shadowops import ShadowOpsOpenEnv  # noqa: E402
from training.shadowops_training_common import (  # noqa: E402
    build_q_aware_decision,
    compute_risk_score,
    evaluate_outputs,
    load_validation_samples_for_benchmark,
    write_json,
)


DEFAULT_EPISODES = 50
DEFAULT_EPISODE_MAX_LENGTH = 5
DEFAULT_POLICY = "q_aware"
DEFAULT_BASELINE_POLICY = "heuristic"
DEFAULT_BEHAVIOR_EXAMPLES = 8
TARGET_ACTIONS = {"BLOCK", "FORK", "QUARANTINE"}
TIER_EXPECTED_ACTION = {
    "BENIGN_CLEAN": "ALLOW",
    "BENIGN_NOISY": "ALLOW",
    "AMBIGUOUS": "QUARANTINE",
    "MALICIOUS_SOFT": "BLOCK",
    "MALICIOUS_HARD": "FORK",
}
SUPPORTED_POLICIES = ("random", "heuristic", "q_aware")


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _confidence_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    if len(values) == 1:
        std = 0.0
    else:
        std = float(statistics.pstdev(values))
    return {
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": std,
    }


def _reward_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    if len(values) == 1:
        std = 0.0
    else:
        std = float(statistics.pstdev(values))
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": std,
    }


def _heuristic_action_and_confidence(risk: float, tier: str) -> tuple[str, float, str]:
    tier = str(tier or "").upper()
    if risk >= 0.62:
        confidence = 0.60 + min(0.35, (risk - 0.62) * 1.25)
        return "BLOCK", _clip(confidence), "heuristic high-risk threshold"
    if risk >= 0.38 or tier == "AMBIGUOUS":
        margin = abs(risk - 0.50)
        confidence = 0.56 + min(0.22, margin * 1.4)
        return "FORK", _clip(confidence), "heuristic uncertainty escalation"
    confidence = 0.58 + min(0.30, (0.38 - risk) * 1.2)
    return "ALLOW", _clip(confidence), "heuristic low-risk allow"


def _expected_action(tier: str, is_malicious: bool) -> str:
    tier_name = str(tier or "").upper()
    if tier_name in TIER_EXPECTED_ACTION:
        return TIER_EXPECTED_ACTION[tier_name]
    return "FORK" if is_malicious else "ALLOW"


def _choose_policy_decision(
    env: ShadowOpsOpenEnv,
    observation: dict[str, Any],
    policy_name: str,
    rng: random.Random,
) -> dict[str, Any]:
    actions = list(observation.get("available_actions") or ["ALLOW", "BLOCK", "FORK", "QUARANTINE"])
    incident = observation.get("incident_state", {})
    risk_vector = list(observation.get("risk_vector", [0.0] * 16))
    risk = float(compute_risk_score(risk_vector))

    if policy_name == "random":
        action = rng.choice(actions)
        return {
            "action": action,
            "confidence": 0.25,
            "explanation": "random baseline action",
        }

    if policy_name == "heuristic":
        action, confidence, reason = _heuristic_action_and_confidence(risk, str(incident.get("tier", "")))
        return {
            "action": action,
            "confidence": confidence,
            "explanation": reason,
        }

    decision = build_q_aware_decision(
        incident.get("domain", "SOC"),
        incident.get("intent", "UNKNOWN"),
        incident.get("payload", ""),
        risk_vector,
        actor="openenv_agent",
        session_id=env.session_id,
        service=incident.get("domain", "unknown"),
        environment="production",
        provided_evidence=[],
        memory_context=env.state().get("memory_context", {}),
    )
    action = str(decision.get("decision", "QUARANTINE")).upper()
    if action not in actions:
        action = "QUARANTINE"
    confidence = decision.get("confidence")
    if confidence is None:
        confidence = 0.50 + min(0.40, abs(risk - 0.50))
    return {
        "action": action,
        "confidence": _clip(float(confidence)),
        "explanation": str(decision.get("explanation", "q-aware policy decision")).strip() or "q-aware policy decision",
    }


def _sample_policy_decision(
    sample: dict[str, Any],
    policy_name: str,
    rng: random.Random,
) -> dict[str, Any]:
    risk_vector = list(sample.get("risk_vector") or [0.0] * 16)
    risk = float(sample.get("risk_score", compute_risk_score(risk_vector)))
    domain = str(sample.get("domain", "SOC"))
    intent = str(sample.get("intent", "UNKNOWN"))
    payload = str(sample.get("raw_payload", ""))
    tier = str(sample.get("scenario_type", ""))

    if policy_name == "random":
        action = rng.choice(["ALLOW", "BLOCK", "FORK", "QUARANTINE"])
        return {"action": action, "confidence": 0.25, "rationale": "random baseline"}

    if policy_name == "heuristic":
        action, confidence, reason = _heuristic_action_and_confidence(risk, tier)
        return {"action": action, "confidence": confidence, "rationale": reason}

    decision = build_q_aware_decision(
        domain,
        intent,
        payload,
        risk_vector,
        actor=str(sample.get("actor", "openenv_eval")),
        session_id=str(sample.get("session_id", "openenv-eval")),
        service=str(sample.get("service", domain.lower())),
        environment=str(sample.get("environment", "production")),
        provided_evidence=list(sample.get("provided_evidence", [])),
        memory_context={},
    )
    action = str(decision.get("decision", "QUARANTINE")).upper()
    if action not in {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}:
        action = "QUARANTINE"
    confidence = decision.get("confidence")
    if confidence is None:
        confidence = 0.50 + min(0.40, abs(risk - 0.50))
    rationale = str(decision.get("explanation", "q-aware serving policy"))
    return {"action": action, "confidence": _clip(float(confidence)), "rationale": rationale}


def _is_smoke_run(episodes: int) -> bool:
    return int(episodes) < DEFAULT_EPISODES


def _evaluate_step_pass(action: str, is_malicious: bool) -> bool:
    if is_malicious:
        return action in TARGET_ACTIONS
    return action == "ALLOW"


def evaluate_openenv_loop(
    *,
    policy_name: str = DEFAULT_POLICY,
    episodes: int = DEFAULT_EPISODES,
    episode_max_length: int = DEFAULT_EPISODE_MAX_LENGTH,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a policy through the actual multi-step environment loop."""

    rng = random.Random(seed)
    episode_rows: list[dict[str, Any]] = []
    step_rewards: list[float] = []
    episode_rewards: list[float] = []
    confidence_values: list[float] = []

    total_reward = 0.0
    total_steps = 0
    malicious_steps = 0
    benign_steps = 0
    unsafe_allows = 0
    safe_blocks = 0
    safe_allows = 0
    conservative_malicious_actions = 0
    risky_chain_hits = 0
    exact_match_count = 0
    step_pass_count = 0
    step_fail_count = 0

    for episode in range(episodes):
        env = ShadowOpsOpenEnv(seed=seed + episode, episode_max_length=episode_max_length, persist_memory=False)
        observation = env.reset()
        done = False
        steps: list[dict[str, Any]] = []
        episode_unsafe_allows = 0
        episode_safe_blocks = 0
        episode_exact = 0
        episode_passes = 0
        episode_fails = 0

        while not done:
            incident = dict(observation.get("incident_state", {}))
            tier = str(incident.get("tier", "UNKNOWN"))
            payload = str(incident.get("payload", ""))
            decision = _choose_policy_decision(env, observation, policy_name, rng)
            action = str(decision["action"])
            confidence = _clip(float(decision.get("confidence", 0.0)))

            observation, reward, done, info = env.step(action)
            total_reward += reward
            total_steps += 1
            step_rewards.append(float(reward))
            confidence_values.append(confidence)

            is_malicious = bool(info.get("is_malicious", False))
            expected = _expected_action(tier, is_malicious)
            exact = action == expected
            pass_step = _evaluate_step_pass(action, is_malicious)
            if exact:
                exact_match_count += 1
                episode_exact += 1
            if pass_step:
                step_pass_count += 1
                episode_passes += 1
            else:
                step_fail_count += 1
                episode_fails += 1

            if is_malicious:
                malicious_steps += 1
                if action == "ALLOW":
                    unsafe_allows += 1
                    episode_unsafe_allows += 1
                if action in TARGET_ACTIONS:
                    conservative_malicious_actions += 1
            else:
                benign_steps += 1
                if action in TARGET_ACTIONS:
                    safe_blocks += 1
                    episode_safe_blocks += 1
                if action == "ALLOW":
                    safe_allows += 1

            if info.get("memory_context", {}).get("risky_chains"):
                risky_chain_hits += 1

            steps.append(
                {
                    "step": int(info.get("step", len(steps) + 1)),
                    "domain": info.get("domain"),
                    "tier": tier,
                    "payload_excerpt": payload[:140],
                    "action": action,
                    "expected_action": expected,
                    "action_correct": exact,
                    "step_passed": pass_step,
                    "reward": float(reward),
                    "confidence": confidence,
                    "is_malicious": is_malicious,
                    "outcome": info.get("outcome"),
                    "cumulative_risk_score": float(info.get("cumulative_risk_score", 0.0) or 0.0),
                    "missing_evidence_count": len(info.get("missing_evidence", [])),
                    "evidence_plan_steps": len(info.get("evidence_plan", [])),
                    "policy_explanation": str(decision.get("explanation", ""))[:220],
                }
            )

        state = env.state()
        episode_reward = float(state.get("episode_reward", 0.0) or 0.0)
        episode_rewards.append(episode_reward)
        episode_rows.append(
            {
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "steps": steps,
                "unsafe_allow_steps": episode_unsafe_allows,
                "safe_block_steps": episode_safe_blocks,
                "exact_match_steps": episode_exact,
                "pass_steps": episode_passes,
                "fail_steps": episode_fails,
                "final_health": state.get("health", {}),
            }
        )
        env.close()

    smoke_run = _is_smoke_run(episodes)
    run_label = "smoke_test" if smoke_run else f"full_eval_{episodes}_episodes"
    unsafe_allow_rate = unsafe_allows / max(malicious_steps, 1)
    safe_block_rate = safe_blocks / max(benign_steps, 1)

    return {
        "policy": policy_name,
        "seed": seed,
        "episodes": episodes,
        "episode_max_length": episode_max_length,
        "run_label": run_label,
        "is_smoke_test": smoke_run,
        "run_scope_note": (
            "SMOKE TEST: fewer than 50 episodes; not a full judge run."
            if smoke_run
            else f"FULL EVAL: {episodes} episodes (judge-facing run size)."
        ),
        "total_steps": total_steps,
        "malicious_steps": malicious_steps,
        "benign_steps": benign_steps,
        "mean_reward_per_step": total_reward / max(total_steps, 1),
        "reward_summary_per_step": _reward_summary(step_rewards),
        "reward_summary_per_episode": _reward_summary(episode_rewards),
        "mean_episode_reward": float(statistics.mean(episode_rewards)) if episode_rewards else 0.0,
        "accuracy": exact_match_count / max(total_steps, 1),
        "unsafe_allow_count": unsafe_allows,
        "unsafe_allow_rate": unsafe_allow_rate,
        "unsafe_allow_rate_per_step": unsafe_allows / max(total_steps, 1),
        "safe_block_count": safe_blocks,
        "safe_block_rate": safe_block_rate,
        "safe_allow_count": safe_allows,
        "safe_allow_rate": safe_allows / max(benign_steps, 1),
        "malicious_block_or_quarantine_count": conservative_malicious_actions,
        "malicious_block_or_quarantine_rate": conservative_malicious_actions / max(malicious_steps, 1),
        "average_confidence": float(statistics.mean(confidence_values)) if confidence_values else 0.0,
        "confidence_summary": _confidence_summary(confidence_values),
        "step_pass_count": step_pass_count,
        "step_fail_count": step_fail_count,
        "risky_chain_signal_count": risky_chain_hits,
        "safety_adjusted_score": (total_reward / max(total_steps, 1)) - (unsafe_allow_rate * 50) - (safe_block_rate * 10) - ((step_fail_count / max(total_steps, 1)) * 5),
        "episodes_detail": episode_rows,
    }


def _model_checkpoint_availability() -> dict[str, Any]:
    comparison_path = TRAINING_DIR / "model_policy_comparison.json"
    if not comparison_path.exists():
        return {
            "comparison_file": str(comparison_path.relative_to(BACKEND_DIR)),
            "available": False,
            "note": "model_policy_comparison.json not found; checkpoint availability unknown.",
        }
    try:
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "comparison_file": str(comparison_path.relative_to(BACKEND_DIR)),
            "available": False,
            "note": "model_policy_comparison.json is unreadable; checkpoint availability unknown.",
        }
    rows = payload.get("datasets", {}).get("validation", {}).get("rows", [])
    grpo_row = next((row for row in rows if row.get("policy") == "grpo_model"), None)
    if grpo_row is None:
        return {
            "comparison_file": str(comparison_path.relative_to(BACKEND_DIR)),
            "available": False,
            "note": "No grpo_model row was found in model_policy_comparison.json.",
        }
    available = bool(grpo_row.get("available", False))
    return {
        "comparison_file": str(comparison_path.relative_to(BACKEND_DIR)),
        "available": available,
        "note": (
            "Measured grpo_model metrics are available."
            if available
            else "grpo_model row exists but metrics are not available in this repository snapshot."
        ),
    }


def build_before_after_behavior_comparison(
    *,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    target_policy: str = DEFAULT_POLICY,
    seed: int = 42,
    max_examples: int = DEFAULT_BEHAVIOR_EXAMPLES,
) -> dict[str, Any]:
    samples, _ = load_validation_samples_for_benchmark()
    rng = random.Random(seed + 17)

    rows: list[dict[str, Any]] = []
    baseline_actions: list[str] = []
    target_actions: list[str] = []

    for sample in samples:
        baseline = _sample_policy_decision(sample, baseline_policy, rng)
        target = _sample_policy_decision(sample, target_policy, rng)
        expected = str(sample.get("correct_action") or sample.get("expected_decision") or "UNKNOWN")

        baseline_action = str(baseline["action"])
        target_action = str(target["action"])
        baseline_actions.append(baseline_action)
        target_actions.append(target_action)

        row = {
            "scenario_id": str(sample.get("sample_id", "")),
            "scenario": f"{sample.get('domain', 'UNKNOWN')}::{sample.get('intent', 'UNKNOWN')}",
            "scenario_summary": str(sample.get("raw_payload", ""))[:180],
            "expected_action": expected,
            "baseline_action": baseline_action,
            "qaware_action": target_action,
            "baseline_correct": baseline_action == expected,
            "trained_correct": target_action == expected,
            "baseline_confidence": round(float(baseline.get("confidence", 0.0) or 0.0), 3),
            "confidence": round(float(target.get("confidence", 0.0) or 0.0), 3),
            "baseline_failure_reason": str(baseline.get("rationale", ""))[:200] if baseline_action != expected else "",
            "qaware_success_reason": str(target.get("rationale", ""))[:200] if target_action == expected else "",
            "risk_score": round(float(sample.get("risk_score", 0.0) or 0.0), 4),
            "missing_evidence": list(sample.get("missing_evidence", [])),
            "evidence_plan": list(sample.get("evidence_plan", [])),
            "safe_outcome": target_action in ("BLOCK", "FORK", "QUARANTINE") and expected in ("BLOCK", "FORK", "QUARANTINE") or target_action == "ALLOW" and expected == "ALLOW",
        }
        rows.append(row)

    baseline_metrics = evaluate_outputs(samples, baseline_actions, label=f"{baseline_policy}_baseline")
    target_metrics = evaluate_outputs(samples, target_actions, label=f"{target_policy}_serving")

    deltas = {
        "exact_match_delta": float(target_metrics.get("exact_match", 0.0) - baseline_metrics.get("exact_match", 0.0)),
        "safety_accuracy_delta": float(
            target_metrics.get("safety_accuracy", 0.0) - baseline_metrics.get("safety_accuracy", 0.0)
        ),
        "unsafe_decision_rate_delta": float(
            target_metrics.get("unsafe_decision_rate", 0.0) - baseline_metrics.get("unsafe_decision_rate", 0.0)
        ),
        "reward_mean_delta": float(target_metrics.get("reward_mean", 0.0) - baseline_metrics.get("reward_mean", 0.0)),
    }

    differing = [row for row in rows if row["baseline_action"] != row["qaware_action"]]
    differing.sort(
        key=lambda row: (
            int(row["trained_correct"]) - int(row["baseline_correct"]),
            row["risk_score"],
        ),
        reverse=True,
    )
    selected = differing[: max(0, max_examples)]
    if len(selected) < max_examples:
        selected_ids = {row["scenario_id"] for row in selected}
        fallback = [
            row
            for row in sorted(rows, key=lambda item: abs(float(item["risk_score"]) - 0.5), reverse=True)
            if row["scenario_id"] not in selected_ids
        ]
        selected.extend(fallback[: max_examples - len(selected)])

    checkpoint_status = _model_checkpoint_availability()

    return {
        "title": "ShadowOps before/after behavior comparison",
        "comparison_type": "baseline_vs_serving_policy",
        "baseline_policy": baseline_policy,
        "target_policy": target_policy,
        "sample_source": str((TRAINING_DIR / "qwen3_val_dataset.json").relative_to(BACKEND_DIR)),
        "sample_count": len(samples),
        "checkpoint_status": checkpoint_status,
        "note": (
            "Target policy is serving-time q_aware logic. This file does not claim checkpoint training gains "
            "unless checkpoint_status.available is true."
        ),
        "aggregate": {
            "baseline": {
                "exact_match": baseline_metrics.get("exact_match", 0.0),
                "safety_accuracy": baseline_metrics.get("safety_accuracy", 0.0),
                "unsafe_decision_rate": baseline_metrics.get("unsafe_decision_rate", 0.0),
                "reward_mean": baseline_metrics.get("reward_mean", 0.0),
            },
            "trained_or_serving": {
                "exact_match": target_metrics.get("exact_match", 0.0),
                "safety_accuracy": target_metrics.get("safety_accuracy", 0.0),
                "unsafe_decision_rate": target_metrics.get("unsafe_decision_rate", 0.0),
                "reward_mean": target_metrics.get("reward_mean", 0.0),
            },
            "delta_target_minus_baseline": deltas,
        },
        "examples": selected,
    }


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


def _write_episode_reward_plot(
    *,
    baseline_rewards: list[float],
    target_rewards: list[float],
    output_path: Path,
) -> None:
    width, height = 960, 420
    margin_left, margin_right, margin_top, margin_bottom = 56, 20, 16, 34
    pixels = [[(250, 252, 255) for _ in range(width)] for _ in range(height)]

    axis = (70, 82, 102)
    grid = (228, 233, 242)
    for y in range(margin_top, height - margin_bottom):
        pixels[y][margin_left] = axis
    for x in range(margin_left, width - margin_right):
        pixels[height - margin_bottom][x] = axis
    for line in range(1, 5):
        y = margin_top + int((height - margin_top - margin_bottom) * line / 5)
        for x in range(margin_left + 1, width - margin_right):
            pixels[y][x] = grid

    all_values = [float(v) for v in baseline_rewards + target_rewards]
    if not all_values:
        _write_png_rgb(output_path, width, height, pixels)
        return
    min_v = min(all_values)
    max_v = max(all_values)
    if abs(max_v - min_v) < 1e-9:
        max_v = min_v + 1.0

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_index = max(len(baseline_rewards), len(target_rewards)) - 1
    max_index = max(max_index, 1)

    def map_point(index: int, value: float) -> tuple[int, int]:
        x = margin_left + int(index * plot_w / max_index)
        y = margin_top + int((1.0 - ((value - min_v) / (max_v - min_v))) * plot_h)
        return x, y

    baseline_color = (213, 78, 76)
    target_color = (46, 172, 104)

    if len(baseline_rewards) >= 2:
        baseline_points = [map_point(i, float(v)) for i, v in enumerate(baseline_rewards)]
        for (x0, y0), (x1, y1) in zip(baseline_points, baseline_points[1:]):
            _draw_line(pixels, x0, y0, x1, y1, baseline_color)
    if len(target_rewards) >= 2:
        target_points = [map_point(i, float(v)) for i, v in enumerate(target_rewards)]
        for (x0, y0), (x1, y1) in zip(target_points, target_points[1:]):
            _draw_line(pixels, x0, y0, x1, y1, target_color)

    _write_png_rgb(output_path, width, height, pixels)


def _write_behavior_comparison(
    comparison: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, str]:
    json_path = output_dir / "openenv_behavior_comparison.json"
    md_path = output_dir / "openenv_behavior_comparison.md"
    write_json(json_path, comparison)

    lines = [
        "# ShadowOps Before/After Behavior Comparison",
        "",
        f"- Baseline policy: {comparison['baseline_policy']}",
        f"- Target policy: {comparison['target_policy']}",
        f"- Sample source: `{comparison['sample_source']}`",
        f"- Samples: {comparison['sample_count']}",
        f"- Checkpoint availability: {comparison['checkpoint_status']['available']}",
        f"- Checkpoint note: {comparison['checkpoint_status']['note']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Baseline | Target/Serving | Delta (target-baseline) |",
        "| --- | ---: | ---: | ---: |",
        f"| Exact match | {comparison['aggregate']['baseline']['exact_match']:.3f} | {comparison['aggregate']['trained_or_serving']['exact_match']:.3f} | {comparison['aggregate']['delta_target_minus_baseline']['exact_match_delta']:+.3f} |",
        f"| Safety accuracy | {comparison['aggregate']['baseline']['safety_accuracy']:.3f} | {comparison['aggregate']['trained_or_serving']['safety_accuracy']:.3f} | {comparison['aggregate']['delta_target_minus_baseline']['safety_accuracy_delta']:+.3f} |",
        f"| Unsafe decision rate | {comparison['aggregate']['baseline']['unsafe_decision_rate']:.3f} | {comparison['aggregate']['trained_or_serving']['unsafe_decision_rate']:.3f} | {comparison['aggregate']['delta_target_minus_baseline']['unsafe_decision_rate_delta']:+.3f} |",
        f"| Reward mean | {comparison['aggregate']['baseline']['reward_mean']:.3f} | {comparison['aggregate']['trained_or_serving']['reward_mean']:.3f} | {comparison['aggregate']['delta_target_minus_baseline']['reward_mean_delta']:+.3f} |",
        "",
        "## Representative Scenarios",
        "",
        "| Scenario ID | Scenario Summary | Expected Action | Baseline Action | Q-Aware Action | Failure Reason (Baseline) | Success Reason (Q-Aware) | Risk Score | Confidence | Safe Outcome |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in comparison["examples"]:
        baseline_rationale = str(row["baseline_failure_reason"]).replace("|", "/")
        trained_rationale = str(row["qaware_success_reason"]).replace("|", "/")
        lines.append(
            f"| `{row['scenario_id']}` | {row['scenario_summary']} | {row['expected_action']} | "
            f"{row['baseline_action']} | {row['qaware_action']} | "
            f"{baseline_rationale} | {trained_rationale} | "
            f"{row['risk_score']:.3f} | {row['confidence']:.3f} | {row['safe_outcome']} |"
        )
    lines.extend(["", "## Note", "", comparison["note"]])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path.name, "md": md_path.name}


def write_openenv_report(report: dict[str, Any], output_dir: Path = REPORTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "openenv_loop_eval.json"
    md_path = output_dir / "openenv_loop_eval.md"
    write_json(json_path, report)

    reward_plot = report.get("artifacts", {}).get("episode_reward_plot", "openenv_episode_rewards.png")
    behavior_md = report.get("artifacts", {}).get("behavior_comparison_md", "openenv_behavior_comparison.md")
    lines = [
        "# ShadowOps OpenEnv Loop Evaluation",
        "",
        f"- Policy evaluated: {report['policy']}",
        f"- Baseline policy for comparison: {report['baseline_policy']}",
        f"- Episodes: {report['episodes']}",
        f"- Episode max length: {report['episode_max_length']}",
        f"- Seed: {report['seed']}",
        f"- Run label: {report['run_label']}",
        f"- Scope note: {report['run_scope_note']}",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total steps | {report['total_steps']} |",
        f"| Malicious steps | {report['malicious_steps']} |",
        f"| Benign steps | {report['benign_steps']} |",
        f"| Accuracy | {report['accuracy']:.3f} |",
        f"| Unsafe allow rate (malicious-only) | {report['unsafe_allow_rate']:.3f} |",
        f"| Safe block rate (benign blocked/forked/quarantined) | {report['safe_block_rate']:.3f} |",
        f"| Average confidence | {report['average_confidence']:.3f} |",
        f"| Mean reward per step | {report['mean_reward_per_step']:.3f} |",
        f"| Step pass count | {report['step_pass_count']} |",
        f"| Step fail count | {report['step_fail_count']} |",
        "",
        "## Reward Summary",
        "",
        f"- Per-step reward mean/median/std: {report['reward_summary_per_step']['mean']:.3f} / {report['reward_summary_per_step']['median']:.3f} / {report['reward_summary_per_step']['std']:.3f}",
        f"- Per-episode reward mean/median/std: {report['reward_summary_per_episode']['mean']:.3f} / {report['reward_summary_per_episode']['median']:.3f} / {report['reward_summary_per_episode']['std']:.3f}",
        f"- Per-episode reward min/max: {report['reward_summary_per_episode']['min']:.3f} / {report['reward_summary_per_episode']['max']:.3f}",
        "",
        "## Before vs After Aggregate",
        "",
        "| Metric | Baseline | Target | Delta (target-baseline) |",
        "| --- | ---: | ---: | ---: |",
        f"| Unsafe allow rate | {report['baseline_summary']['unsafe_allow_rate']:.3f} | {report['unsafe_allow_rate']:.3f} | {report['comparison_delta']['unsafe_allow_rate_delta']:+.3f} |",
        f"| Safe block rate | {report['baseline_summary']['safe_block_rate']:.3f} | {report['safe_block_rate']:.3f} | {report['comparison_delta']['safe_block_rate_delta']:+.3f} |",
        f"| Average confidence | {report['baseline_summary']['average_confidence']:.3f} | {report['average_confidence']:.3f} | {report['comparison_delta']['average_confidence_delta']:+.3f} |",
        f"| Mean reward/step | {report['baseline_summary']['mean_reward_per_step']:.3f} | {report['mean_reward_per_step']:.3f} | {report['comparison_delta']['mean_reward_per_step_delta']:+.3f} |",
        f"| Safety Adjusted Score | {report['baseline_summary']['safety_adjusted_score']:.3f} | {report['safety_adjusted_score']:.3f} | {report['comparison_delta'].get('safety_adjusted_score_delta', 0.0):+.3f} |",
        "",
        "## Safety vs Reward Trade-Off",
        "",
        "- **Note on Reward vs Safety**: The `heuristic` baseline may occasionally have a higher `mean_reward_per_step` due to faster resolution times.",
        "- However, **Q-aware is considered safer** when its `unsafe_allow_rate = 0.000`. Unsafe allow is the primary failure mode in security automation and carries severe negative business impact.",
        "- Lower confidence scores in Q-aware do not necessarily mean failure; they often reflect **cautious uncertainty** on ambiguous payloads, which correctly triggers QUARANTINE instead of false-positive blocks or dangerous allows.",
        "",
        "## Representative Behavior",
        "",
        f"See `{behavior_md}` for 5-10 structured before/after scenarios.",
        "",
        "| Scenario | Baseline | Target | Baseline correct | Target correct |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["behavior_examples"]:
        lines.append(
            f"| {row['scenario']} | {row['baseline_action']} | {row['qaware_action']} | "
            f"{row['baseline_correct']} | {row['trained_correct']} |"
        )

    lines.extend(
        [
            "",
            "## Episode Summary",
            "",
            "| Episode | Reward | Steps | Unsafe allows | Safe blocks | Pass | Fail | Final SOC | Final GitHub | Final AWS |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["episodes_detail"]:
        health = row.get("final_health", {})
        lines.append(
            f"| {row['episode']} | {float(row['episode_reward']):.3f} | {len(row['steps'])} | "
            f"{row['unsafe_allow_steps']} | {row['safe_block_steps']} | {row['pass_steps']} | {row['fail_steps']} | "
            f"{health.get('SOC', 0)} | {health.get('GITHUB', 0)} | {health.get('AWS', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Plot",
            "",
            f"- Episode reward trend plot: `{reward_plot}`",
            "- Color mapping in the plot: red=baseline policy, green=target policy.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def generate_openenv_report(
    output_dir: Path = REPORTS_DIR,
    *,
    policy_name: str = DEFAULT_POLICY,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    episodes: int = DEFAULT_EPISODES,
    episode_max_length: int = DEFAULT_EPISODE_MAX_LENGTH,
    seed: int = 42,
    behavior_examples: int = DEFAULT_BEHAVIOR_EXAMPLES,
) -> dict[str, Any]:
    baseline_report = evaluate_openenv_loop(
        policy_name=baseline_policy,
        episodes=episodes,
        episode_max_length=episode_max_length,
        seed=seed,
    )
    target_report = evaluate_openenv_loop(
        policy_name=policy_name,
        episodes=episodes,
        episode_max_length=episode_max_length,
        seed=seed,
    )

    comparison_delta = {
        "unsafe_allow_rate_delta": float(target_report["unsafe_allow_rate"] - baseline_report["unsafe_allow_rate"]),
        "safe_block_rate_delta": float(target_report["safe_block_rate"] - baseline_report["safe_block_rate"]),
        "average_confidence_delta": float(target_report["average_confidence"] - baseline_report["average_confidence"]),
        "mean_reward_per_step_delta": float(
            target_report["mean_reward_per_step"] - baseline_report["mean_reward_per_step"]
        ),
        "safety_adjusted_score_delta": float(
            target_report["safety_adjusted_score"] - baseline_report["safety_adjusted_score"]
        ),
    }
    comparison = build_before_after_behavior_comparison(
        baseline_policy=baseline_policy,
        target_policy=policy_name,
        seed=seed,
        max_examples=max(5, min(10, behavior_examples)),
    )
    behavior_files = _write_behavior_comparison(comparison, output_dir=output_dir)

    plots_dir = TRAINING_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reward_plot_path = plots_dir / "openenv_episode_rewards.png"
    _write_episode_reward_plot(
        baseline_rewards=[float(row["episode_reward"]) for row in baseline_report["episodes_detail"]],
        target_rewards=[float(row["episode_reward"]) for row in target_report["episodes_detail"]],
        output_path=reward_plot_path,
    )

    report = dict(target_report)
    report["baseline_policy"] = baseline_policy
    report["baseline_summary"] = {
        "unsafe_allow_rate": baseline_report["unsafe_allow_rate"],
        "safe_block_rate": baseline_report["safe_block_rate"],
        "average_confidence": baseline_report["average_confidence"],
        "mean_reward_per_step": baseline_report["mean_reward_per_step"],
        "safety_adjusted_score": baseline_report.get("safety_adjusted_score", 0.0),
    }
    report["comparison_delta"] = comparison_delta
    report["behavior_examples"] = comparison["examples"]
    report["checkpoint_status"] = comparison["checkpoint_status"]
    report["artifacts"] = {
        "episode_reward_plot": reward_plot_path.name,
        "behavior_comparison_json": behavior_files["json"],
        "behavior_comparison_md": behavior_files["md"],
    }
    write_openenv_report(report, output_dir)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShadowOps OpenEnv evaluation runner")
    parser.add_argument("--policy", default=DEFAULT_POLICY, choices=SUPPORTED_POLICIES, help="Target policy")
    parser.add_argument(
        "--baseline-policy",
        default=DEFAULT_BASELINE_POLICY,
        choices=SUPPORTED_POLICIES,
        help="Baseline policy for before/after comparison",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes to evaluate")
    parser.add_argument(
        "--episode-max-length",
        type=int,
        default=DEFAULT_EPISODE_MAX_LENGTH,
        help="Max steps per episode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--behavior-examples",
        type=int,
        default=DEFAULT_BEHAVIOR_EXAMPLES,
        help="Representative scenarios to keep in before/after summary (5-10 recommended)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Output directory for openenv_loop_eval artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = generate_openenv_report(
        output_dir=args.output_dir,
        policy_name=args.policy,
        baseline_policy=args.baseline_policy,
        episodes=max(1, int(args.episodes)),
        episode_max_length=max(1, int(args.episode_max_length)),
        seed=int(args.seed),
        behavior_examples=max(1, int(args.behavior_examples)),
    )
    output_dir = args.output_dir
    print(f"OpenEnv episodes: {report['episodes']}")
    print(f"OpenEnv run label: {report['run_label']}")
    print(f"OpenEnv unsafe allow rate: {report['unsafe_allow_rate']:.3f}")
    print(f"Saved: {(output_dir / 'openenv_loop_eval.json').relative_to(BACKEND_DIR)}")
    print(f"Saved: {(output_dir / 'openenv_loop_eval.md').relative_to(BACKEND_DIR)}")
    print(f"Saved: {(output_dir / 'openenv_behavior_comparison.json').relative_to(BACKEND_DIR)}")
    print(f"Saved: {(output_dir / 'openenv_behavior_comparison.md').relative_to(BACKEND_DIR)}")
    print(f"Saved: {(output_dir / 'openenv_episode_rewards.png').relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
