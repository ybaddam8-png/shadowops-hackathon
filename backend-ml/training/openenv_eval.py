"""Laptop-safe OpenEnv loop evaluation for ShadowOps."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from openenv_shadowops import ShadowOpsOpenEnv  # noqa: E402
from training.shadowops_training_common import build_q_aware_decision, compute_risk_score, write_json  # noqa: E402


def _choose_action(env: ShadowOpsOpenEnv, observation: dict[str, Any], policy_name: str, rng: random.Random) -> str:
    actions = list(observation.get("available_actions") or ["ALLOW", "BLOCK", "FORK", "QUARANTINE"])
    if policy_name == "random":
        return rng.choice(actions)
    incident = observation.get("incident_state", {})
    risk_vector = list(observation.get("risk_vector", [0.0] * 16))
    if policy_name == "heuristic":
        risk = compute_risk_score(risk_vector)
        if risk >= 0.62:
            return "BLOCK"
        if risk >= 0.38 or incident.get("tier") == "AMBIGUOUS":
            return "FORK"
        return "ALLOW"
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
    return str(decision.get("decision", "QUARANTINE"))


def evaluate_openenv_loop(
    *,
    policy_name: str = "q_aware",
    episodes: int = 3,
    episode_max_length: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a policy through the actual multi-step environment loop."""

    rng = random.Random(seed)
    episode_rows = []
    total_reward = 0.0
    unsafe_allows = 0
    total_steps = 0
    risky_chain_hits = 0

    for episode in range(episodes):
        env = ShadowOpsOpenEnv(seed=seed + episode, episode_max_length=episode_max_length, persist_memory=False)
        observation = env.reset()
        done = False
        steps = []
        while not done:
            action = _choose_action(env, observation, policy_name, rng)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            total_steps += 1
            if action == "ALLOW" and info.get("is_malicious"):
                unsafe_allows += 1
            if info.get("memory_context", {}).get("risky_chains"):
                risky_chain_hits += 1
            steps.append(
                {
                    "step": info.get("step"),
                    "domain": info.get("domain"),
                    "action": action,
                    "reward": reward,
                    "outcome": info.get("outcome"),
                    "cumulative_risk_score": info.get("cumulative_risk_score", 0.0),
                    "missing_evidence_count": len(info.get("missing_evidence", [])),
                    "evidence_plan_steps": len(info.get("evidence_plan", [])),
                }
            )
        episode_rows.append(
            {
                "episode": episode + 1,
                "episode_reward": env.state().get("episode_reward", 0.0),
                "steps": steps,
                "final_health": env.state().get("health", {}),
            }
        )
        env.close()

    return {
        "policy": policy_name,
        "episodes": episodes,
        "episode_max_length": episode_max_length,
        "total_steps": total_steps,
        "mean_reward_per_step": total_reward / max(total_steps, 1),
        "unsafe_allow_rate": unsafe_allows / max(total_steps, 1),
        "risky_chain_signal_count": risky_chain_hits,
        "episodes_detail": episode_rows,
    }


def write_openenv_report(report: dict[str, Any], output_dir: Path = REPORTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "openenv_loop_eval.json"
    md_path = output_dir / "openenv_loop_eval.md"
    write_json(json_path, report)
    lines = [
        "# ShadowOps OpenEnv Loop Evaluation",
        "",
        f"- Policy: {report['policy']}",
        f"- Episodes: {report['episodes']}",
        f"- Total steps: {report['total_steps']}",
        f"- Mean reward per step: {report['mean_reward_per_step']:.3f}",
        f"- Unsafe allow rate: {report['unsafe_allow_rate']:.3f}",
        f"- Risky chain signals: {report['risky_chain_signal_count']}",
        "",
        "| Episode | Reward | Steps | Final SOC | Final GitHub | Final AWS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["episodes_detail"]:
        health = row.get("final_health", {})
        lines.append(
            f"| {row['episode']} | {float(row['episode_reward']):.3f} | {len(row['steps'])} | "
            f"{health.get('SOC', 0)} | {health.get('GITHUB', 0)} | {health.get('AWS', 0)} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def generate_openenv_report(output_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    report = evaluate_openenv_loop()
    write_openenv_report(report, output_dir)
    return report


def main() -> int:
    report = generate_openenv_report()
    print(f"OpenEnv episodes: {report['episodes']}")
    print(f"OpenEnv unsafe allow rate: {report['unsafe_allow_rate']:.3f}")
    print(f"Saved: {(REPORTS_DIR / 'openenv_loop_eval.json').relative_to(BACKEND_DIR)}")
    print(f"Saved: {(REPORTS_DIR / 'openenv_loop_eval.md').relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
