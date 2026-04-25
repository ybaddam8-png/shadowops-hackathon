"""OpenEnv-style wrapper for the ShadowOps cybersecurity environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any

from agent_memory import ActionMemoryRecord, SessionMemory
from shadowops_env import ACTIONS, UniversalShadowEnv, compute_ambiguity, extract_features
from training.reward_rubric import score_reward_rubric
from training.shadowops_training_common import build_q_aware_decision, normalize_action_output


VALID_ACTIONS = tuple(ACTIONS.values())
ACTION_TO_INDEX = {label: index for index, label in ACTIONS.items()}


@dataclass
class EpisodeStep:
    step: int
    action: str
    reward: float
    domain: str
    outcome: str
    risk_score: float
    decision_context: dict[str, Any] = field(default_factory=dict)


class ShadowOpsOpenEnv:
    """Small Gym/OpenEnv-compatible wrapper around ``UniversalShadowEnv``.

    The wrapper keeps the model-free demo deterministic while exposing a
    judge-friendly environment contract: ``reset()``, ``step(action)``,
    ``state()``, and ``close()``. Actions affect production/shadow state,
    quarantine holds, memory, accumulated risk, and future observations.
    """

    metadata = {
        "name": "shadowops",
        "render_modes": [],
        "actions": VALID_ACTIONS,
    }

    def __init__(
        self,
        *,
        seed: int = 42,
        malicious_rate: float = 0.5,
        episode_max_length: int = 8,
        memory_path: Path | str | None = None,
        persist_memory: bool = False,
    ) -> None:
        self.seed = seed
        self.episode_max_length = episode_max_length
        self._env = UniversalShadowEnv(
            malicious_rate=malicious_rate,
            episode_max_length=episode_max_length,
            mode="openenv",
            seed=seed,
        )
        self.memory = SessionMemory(
            persistence_enabled=persist_memory,
            storage_path=memory_path or Path(__file__).resolve().parent / "data" / "openenv_session_memory.json",
        )
        self.session_id = f"openenv-{seed}"
        self.history: list[EpisodeStep] = []
        self._last_observation: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}

    def reset(self) -> dict[str, Any]:
        """Reset the episode and return an observation object."""

        obs_text, obs_vec = self._env.reset()
        self.history.clear()
        self._last_info = {}
        self._last_observation = self._format_observation(obs_text, obs_vec)
        return self._last_observation

    def step(self, action: str | int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply an action and return ``(observation, reward, done, info)``."""

        action_label = self._normalize_action(action)
        scenario = dict(self._env._current_scenario or {})
        risk_vector = extract_features(
            scenario.get("domain", "SOC"),
            scenario.get("intent", "UNKNOWN"),
            scenario.get("raw_payload", ""),
            self._env.rng,
        )
        risk_score = float(sum(risk_vector[:4]) / max(len(risk_vector[:4]), 1))
        memory_context = self.memory.summarize_memory_context(self.session_id)
        decision_context = build_q_aware_decision(
            scenario.get("domain", "SOC"),
            scenario.get("intent", "UNKNOWN"),
            scenario.get("raw_payload", ""),
            risk_vector,
            actor="openenv_agent",
            session_id=self.session_id,
            service=scenario.get("domain", "unknown"),
            environment="production",
            provided_evidence=[],
            timestamp=len(self.history) + 1,
            memory_context=memory_context,
        )

        obs_text, obs_vec, reward, done, info = self._env.step(ACTION_TO_INDEX[action_label])
        info = dict(info)
        self.memory.add_record(
            ActionMemoryRecord(
                actor="openenv_agent",
                session_id=self.session_id,
                service=str(info.get("domain", scenario.get("domain", "unknown"))),
                domain=str(info.get("domain", scenario.get("domain", "unknown"))),
                environment="production",
                timestamp=time.time(),
                decision=action_label,
                risk_score=float(decision_context.get("cumulative_risk_score", risk_score)),
                action_summary=str(scenario.get("raw_payload", "")),
                indicators=list(decision_context.get("risk_indicators", [])),
            )
        )
        updated_memory_context = self.memory.summarize_memory_context(self.session_id)
        rubric = score_reward_rubric(
            {
                "correct_action": decision_context.get("decision"),
                "severity": "CRITICAL" if decision_context.get("cumulative_risk_score", 0.0) >= 0.80 else "MEDIUM",
                "risk_score": decision_context.get("cumulative_risk_score", 0.0),
                "raw_payload": scenario.get("raw_payload", ""),
                "required_evidence": decision_context.get("required_evidence", []),
                "provided_evidence": [],
                "is_malicious": decision_context.get("decision") in {"BLOCK", "FORK", "QUARANTINE"},
            },
            action_label,
            decision_context,
            memory_context=updated_memory_context,
        )

        info.update(
            {
                "available_actions": list(VALID_ACTIONS),
                "decision_context": decision_context,
                "memory_context": updated_memory_context,
                "reward_rubric": rubric,
                "risk_score": decision_context.get("risk_score", 0.0),
                "cumulative_risk_score": decision_context.get("cumulative_risk_score", 0.0),
                "missing_evidence": decision_context.get("missing_evidence", []),
                "evidence_plan": decision_context.get("evidence_plan", []),
                "safe_outcome": decision_context.get("safe_outcome", ""),
            }
        )
        self.history.append(
            EpisodeStep(
                step=int(info.get("step", len(self.history) + 1)),
                action=action_label,
                reward=float(reward),
                domain=str(info.get("domain", "unknown")),
                outcome=str(info.get("outcome", "unknown")),
                risk_score=float(decision_context.get("cumulative_risk_score", 0.0)),
                decision_context=decision_context,
            )
        )
        self._last_info = info
        self._last_observation = self._format_observation(obs_text, obs_vec)
        return self._last_observation, float(reward), bool(done), info

    def state(self) -> dict[str, Any]:
        """Return the current incident-response state without mutating it."""

        memory_context = self.memory.summarize_memory_context(self.session_id)
        return {
            "session_id": self.session_id,
            "step_count": self._env.step_count,
            "episode_reward": self._env.episode_reward,
            "available_actions": list(VALID_ACTIONS),
            "production": self._env.get_production_snapshot(),
            "health": self._env.get_health_scores(),
            "forensic_log": self._env.get_forensic_log(),
            "incident_reports": self._env.get_incident_reports(),
            "memory_context": memory_context,
            "history": [step.__dict__ for step in self.history],
            "last_info": self._last_info,
        }

    def close(self) -> None:
        """Close hook for OpenEnv/Gym compatibility."""

        return None

    def clear_memory(self) -> None:
        self.memory.clear()

    def _format_observation(self, obs_text: str, obs_vec: list[float]) -> dict[str, Any]:
        current = dict(self._env._current_scenario or {})
        q_active = bool(obs_vec[16]) if len(obs_vec) > 16 else False
        q_steps = obs_vec[17] if len(obs_vec) > 17 else 0.0
        return {
            "prompt": obs_text,
            "risk_vector": list(obs_vec[:16]),
            "quarantine": {
                "active": q_active,
                "steps_remaining_normalized": q_steps,
            },
            "available_actions": list(VALID_ACTIONS),
            "incident_state": {
                "domain": current.get("domain", "unknown"),
                "intent": current.get("intent", "unknown"),
                "payload": current.get("raw_payload", ""),
                "tier": current.get("tier", "unknown"),
                "ambiguity_score": compute_ambiguity(obs_vec[:16]),
                "step_count": self._env.step_count,
                "health": self._env.get_health_scores(),
                "memory_context": self.memory.summarize_memory_context(self.session_id),
            },
        }

    @staticmethod
    def _normalize_action(action: str | int) -> str:
        if isinstance(action, int):
            if action not in ACTIONS:
                raise ValueError(f"Invalid ShadowOps action index: {action}")
            return ACTIONS[action]
        parsed = normalize_action_output(str(action))
        if parsed not in VALID_ACTIONS:
            raise ValueError(f"Invalid ShadowOps action label: {action}")
        return parsed


def make_env(**kwargs: Any) -> ShadowOpsOpenEnv:
    return ShadowOpsOpenEnv(**kwargs)
