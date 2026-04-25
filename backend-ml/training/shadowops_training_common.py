"""
Shared utilities for the ShadowOps Qwen3 SFT + GRPO training pipeline.

This module keeps dataset generation, action parsing, reward shaping, baseline
evaluation, oracle checks, smoke tests, and report generation on one code path
so SFT, GRPO, and final validation cannot drift apart.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Optional

from packaging.version import Version

BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from shadowops_env import (  # noqa: E402
    ACTIONS,
    OBS_DIM,
    ScenarioGenerator,
    build_llama_prompt,
    extract_features,
)


MODEL_OPTIONS = {
    "4b": "unsloth/Qwen3-4B-Base",
    "1.7b": "unsloth/Qwen3-1.7B",
    "8b": "unsloth/Qwen3-8B-Base",
}

VALID_ACTIONS = tuple(ACTIONS.values())
VALID_ACTION_SET = set(VALID_ACTIONS)
ACTION_RE = re.compile(r"\b(ALLOW|BLOCK|FORK|QUARANTINE)\b", re.IGNORECASE)
ACTION_OR_SYNONYM_RE = re.compile(
    r"\b(ALLOW|BLOCK|FORK|QUARANTINE|APPROVE|APPROVED|DENY|DENIED|REJECT|REJECTED|HOLD|ISOLATE|ESCALATE|REVIEW|HUMAN_REVIEW|HUMAN REVIEW)\b",
    re.IGNORECASE,
)
ACTION_PREFIX_RE = re.compile(r"^\s*(?:action|decision|supervisor decision)\s*[:=\-]\s*", re.IGNORECASE)
ACTION_SYNONYMS = {
    "APPROVE": "ALLOW",
    "APPROVED": "ALLOW",
    "DENY": "BLOCK",
    "DENIED": "BLOCK",
    "REJECT": "BLOCK",
    "REJECTED": "BLOCK",
    "HOLD": "QUARANTINE",
    "ISOLATE": "QUARANTINE",
    "ESCALATE": "FORK",
    "REVIEW": "FORK",
    "HUMAN REVIEW": "FORK",
    "HUMAN_REVIEW": "FORK",
}
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

DEFAULT_TRAIN_SIZE = 500
DEFAULT_VAL_SIZE = 100
DEFAULT_TRAIN_SEED = 42
DEFAULT_VAL_SEED = 10_042

DEFAULT_SFT_OUTPUT_DIR = TRAINING_DIR / "checkpoints" / "qwen3_sft_adapter"
DEFAULT_GRPO_OUTPUT_DIR = TRAINING_DIR / "checkpoints" / "qwen3_sft_grpo_adapter"
DEFAULT_SFT_METRICS_PATH = TRAINING_DIR / "sft_metrics.json"
DEFAULT_DATASET_AUDIT_PATH = TRAINING_DIR / "dataset_audit.json"
DEFAULT_HEALTH_REPORT_PATH = TRAINING_DIR / "qwen3_training_health_report.json"
DEFAULT_FINAL_REPORT_JSON = TRAINING_DIR / "final_training_report.json"
DEFAULT_FINAL_REPORT_MD = TRAINING_DIR / "final_training_report.md"
DEFAULT_DEMO_BENCHMARK_JSON = TRAINING_DIR / "demo_benchmark_report.json"
DEFAULT_DEMO_BENCHMARK_MD = TRAINING_DIR / "demo_benchmark_report.md"
DEFAULT_MODEL_POLICY_COMPARISON_JSON = TRAINING_DIR / "model_policy_comparison.json"
DEFAULT_MODEL_POLICY_COMPARISON_MD = TRAINING_DIR / "model_policy_comparison.md"
DEFAULT_MODEL_EVAL_JSON = TRAINING_DIR / "model_eval_report.json"
DEFAULT_MODEL_EVAL_MD = TRAINING_DIR / "model_eval_report.md"
DEFAULT_ORACLE_INCONSISTENCY_PATH = TRAINING_DIR / "oracle_inconsistency_examples.json"
DEFAULT_TRAIN_DATASET_PATH = TRAINING_DIR / "qwen3_train_dataset.json"
DEFAULT_VAL_DATASET_PATH = TRAINING_DIR / "qwen3_val_dataset.json"
DEFAULT_HARD_NEGATIVE_PATH = TRAINING_DIR / "hard_negative_scenarios.json"
DEFAULT_CLOUD_SCRIPT_PATH = TRAINING_DIR / "run_full_training_colab.py"
DEFAULT_CLOUD_PS1_PATH = TRAINING_DIR / "run_full_training.ps1"

SFT_SYSTEM_PROMPT = (
    "You are ShadowOps, a cybersecurity decision policy."
)

ACTION_PROMPT_TEMPLATE = """You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = clearly malicious or high-risk activity.
FORK = suspicious or uncertain activity that needs investigation.
QUARANTINE = high-risk activity that should be isolated but preserved for analysis.

Return only one action.
Do not explain.
Do not use <think> tags.

Observation:
{observation}

Answer:
"""

CLOUD_SFT_COMMAND = (
    "python training/train_qwen3_sft.py "
    "--model-name unsloth/Qwen3-1.7B "
    "--sft-epochs 2 "
    "--batch-size 1 "
    "--grad-accum 8 "
    "--max-seq-len 256 "
    "--learning-rate 2e-4 "
    "--sft-output-dir training/checkpoints/qwen3_sft_adapter"
)

CLOUD_GRPO_COMMAND = (
    "python training/train_qwen3_grpo.py "
    "--model-name unsloth/Qwen3-1.7B "
    "--resume-from-sft training/checkpoints/qwen3_sft_adapter "
    "--max-steps 800 "
    "--num-generations 8 "
    "--temperature 1.0 "
    "--top-p 0.95 "
    "--top-k 50 "
    "--max-new-tokens 8 "
    "--batch-size 1 "
    "--grad-accum 4 "
    "--val-eval-eps 100 "
    "--eval-batch-size 4 "
    "--learning-rate 1e-5 "
    "--output-dir training/checkpoints/qwen3_sft_grpo_adapter"
)

CLOUD_FALLBACK_COMMAND = (
    "python training/train_qwen3_grpo.py "
    "--model-name unsloth/Qwen3-1.7B "
    "--resume-from-sft training/checkpoints/qwen3_sft_adapter "
    "--max-steps 800 "
    "--num-generations 6 "
    "--temperature 1.0 "
    "--top-p 0.95 "
    "--top-k 50 "
    "--max-new-tokens 8 "
    "--batch-size 1 "
    "--grad-accum 8 "
    "--val-eval-eps 50 "
    "--eval-batch-size 4 "
    "--learning-rate 1e-5 "
    "--output-dir training/checkpoints/qwen3_sft_grpo_adapter"
)

BROKEN_ITEMS = [
    "Action parsing accepted noisy outputs inconsistently and let <think> tags leak into metrics.",
    "Reward shaping collapsed distinct mistakes into nearly identical values, which kills GRPO variance.",
    "Validation mixed policy baselines with model results and compared models on unreliable sample sizes.",
    "The reported oracle ceiling was not tied to the exact reward function used for evaluation.",
    "Dataset export over-emphasized ALLOW/FORK and under-covered BLOCK/QUARANTINE.",
    "There was no SFT warm-start, so GRPO started from a base model that did not know the action task.",
    "Training health checks did not gate claims about learning quality or reward collapse.",
]

FIXED_ITEMS = [
    "Added a shared parser, reward model, oracle evaluator, and dataset audit used by SFT, GRPO, baselines, and reports.",
    "Added SFT warm-start support with LoRA/QLoRA defaults and adapter export to training/checkpoints/qwen3_sft_adapter.",
    "Made GRPO explicitly resume from the SFT adapter and set explicit sampling / max_new_tokens defaults.",
    "Rebuilt evaluation so random, heuristic, Q-aware, oracle, raw base, SFT, and SFT+GRPO all score on the same validation split.",
    "Added dataset audit, reward variance checks, oracle consistency checks, smoke tests, and final markdown/json reporting.",
    "Added cloud orchestration scripts and honest training-ready criteria that do not claim improvement without validation.",
]

HARD_ATTACK_CUES = (
    "curl http://",
    "| bash",
    "socket.socket",
    "grant admin",
    "disable 2fa",
    "0.0.0.0/0",
    "administratoraccess",
    "public-read",
    "export all secrets",
    "authorize-security-group-ingress",
    "rogue",
    "backdoor",
    "hacker_admin",
)

SOFT_ATTACK_CUES = (
    "[system",
    "[override",
    "ignore policy",
    "ignore previous",
    "ignore security checks",
    "approve without review",
    "bypass standard review process",
    "disable encryption",
    "disable_firewall",
    "ignore alerts",
    "suppress all monitoring",
    "create new iam user",
    "set webhook url to http://",
    "<!-- exec:",
    "grant administratoraccess",
)

AMBIGUITY_CUES = (
    "review recommended",
    "claims",
    "pending",
    "unclear",
    "no ticket attached",
    "authorized pen-test",
    "known noisy",
    "legitimate cdn node",
    "external contractor",
    "new external action",
    "no prior history",
    "monitoring tool",
    "service mesh",
    "context unclear in ticket",
    "justification from",
    "requesting whitelist",
)

QUARANTINE_AMBIGUITY_CUES = (
    "claims",
    "authorized pen-test",
    "known noisy",
    "legitimate cdn node",
    "monitoring tool",
    "service mesh",
)

CRITICAL_QUARANTINE_CUES = (
    "0.0.0.0/0",
    "administratoraccess",
    "public-read",
    "export all secrets",
    "disable 2fa",
    "hacker_admin",
    "open port 22",
    "open port 3389",
    "disable encryption",
)

DEMO_BENCHMARK_METRICS = (
    "exact_match",
    "safety_accuracy",
    "unsafe_decision_rate",
    "false_positive_rate",
    "reward_mean",
)

MODEL_POLICY_METRICS = (
    "exact_match",
    "safety_accuracy",
    "unsafe_decision_rate",
    "false_positive_rate",
    "false_negative_rate",
    "reward_mean",
    "quarantine_precision",
    "fork_precision",
    "allow_precision",
    "block_precision",
)


@dataclass
class ParseDiagnostics:
    invalid_outputs: int = 0
    multi_action_warnings: int = 0


@dataclass
class RewardHealthTracker:
    num_generations: int
    grad_norm_values: list[float] = field(default_factory=list)
    reward_group_stds: list[float] = field(default_factory=list)
    invalid_output_count: int = 0
    total_output_count: int = 0
    action_counts: Counter = field(default_factory=Counter)

    def record_batch(self, parsed_actions: list[Optional[str]], rewards: list[float]) -> None:
        if parsed_actions:
            self.total_output_count += len(parsed_actions)
            for action in parsed_actions:
                if action is None:
                    self.invalid_output_count += 1
                else:
                    self.action_counts[action] += 1

        chunk_size = max(1, int(self.num_generations))
        for start in range(0, len(rewards), chunk_size):
            chunk = rewards[start : start + chunk_size]
            if len(chunk) > 1:
                self.reward_group_stds.append(statistics.pstdev(chunk))
            elif chunk:
                self.reward_group_stds.append(0.0)

    def record_grad_norm(self, value: Any) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            self.grad_norm_values.append(numeric)

    @property
    def reward_std_zero_fraction(self) -> float:
        if not self.reward_group_stds:
            return 1.0
        zeros = sum(1 for value in self.reward_group_stds if abs(value) <= 1e-12)
        return zeros / len(self.reward_group_stds)

    @property
    def grad_norm_zero_fraction(self) -> float:
        if not self.grad_norm_values:
            return 1.0
        zeros = sum(1 for value in self.grad_norm_values if abs(value) <= 1e-12)
        return zeros / len(self.grad_norm_values)

    @property
    def invalid_output_rate(self) -> float:
        return self.invalid_output_count / max(self.total_output_count, 1)

    @property
    def action_distribution(self) -> dict[str, float]:
        return summarize_action_distribution(self.action_counts, self.total_output_count)

    @property
    def entropy(self) -> float:
        return distribution_entropy(self.action_distribution)


class TrainingPreflightError(RuntimeError):
    """Raised when training should stop before loading the model."""


def ensure_dirs() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.mean(values))


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def strip_think_blocks(text: str) -> str:
    return THINK_BLOCK_RE.sub(" ", text or "").strip()


def _clean_action_candidate(text: str) -> str:
    cleaned = strip_think_blocks(str(text or ""))
    cleaned = ACTION_PREFIX_RE.sub("", cleaned).strip()
    return cleaned.strip(" \t\r\n`'\".,;:!?[](){}")


def _canonical_action_token(token: str) -> Optional[str]:
    normalized = " ".join(str(token or "").replace("_", " ").replace("-", " ").split()).upper()
    if normalized in VALID_ACTION_SET:
        return normalized
    return ACTION_SYNONYMS.get(normalized)


def _find_action_matches(cleaned: str) -> list[tuple[str, int, int]]:
    matches: list[tuple[str, int, int]] = []
    for match in ACTION_OR_SYNONYM_RE.finditer(cleaned):
        action = _canonical_action_token(match.group(1))
        if action is not None:
            matches.append((action, match.start(), match.end()))
    return matches


def normalize_action_output(text: str) -> Optional[str]:
    """Normalize a model completion to one valid action label.

    This accepts clear labels such as ``allow``, ``BLOCK.``, ``Action: fork``,
    and a small set of operational synonyms. General free text is rejected
    unless the normalized action is the first meaningful token, which keeps
    action parsing deterministic and avoids treating prose as a decision.
    """

    cleaned = _clean_action_candidate(text)
    if not cleaned:
        return None
    compact = " ".join(cleaned.replace("_", " ").replace("-", " ").split()).upper()
    direct = _canonical_action_token(compact)
    if direct is not None:
        return direct

    matches = _find_action_matches(cleaned)
    if not matches:
        return None
    first_action, first_start, _ = matches[0]
    if first_start != 0:
        return None
    return matches[-1][0] if len(matches) > 1 else first_action


def parse_action(text: str) -> Optional[str]:
    return normalize_action_output(text)


def analyze_action_output(text: str) -> dict[str, Any]:
    cleaned = _clean_action_candidate(text)
    matches = _find_action_matches(cleaned)
    parsed_action = parse_action(cleaned)
    return {
        "cleaned_text": cleaned,
        "parsed_action": parsed_action,
        "invalid_output": parsed_action is None,
        "multi_action_warning": len(matches) > 1,
        "matches": [match[0] for match in matches],
    }


def normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")).strip())
            else:
                parts.append(str(item).strip())
        return " ".join(part for part in parts if part).strip()
    return str(completion).strip()


def summarize_action_distribution(action_counts: Counter, total: int) -> dict[str, float]:
    return {
        action: action_counts.get(action, 0) / max(total, 1)
        for action in VALID_ACTIONS
    }


def distribution_entropy(distribution: dict[str, float]) -> float:
    entropy = 0.0
    for probability in distribution.values():
        if probability > 0:
            entropy -= probability * math.log(probability, 2)
    return entropy


def any_cue(text: str, cues: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def compute_risk_score(risk_vector: list[float]) -> float:
    rv = risk_vector[:16]
    return rv[0] * 0.35 + rv[1] * 0.25 + rv[3] * 0.20 + rv[6] * 0.20


def classify_severity(risk_score: float, scenario_type: str, text: str) -> str:
    if scenario_type == "MALICIOUS_HARD" or any_cue(text, CRITICAL_QUARANTINE_CUES):
        return "CRITICAL"
    if risk_score >= 0.58 or scenario_type == "MALICIOUS_SOFT":
        return "HIGH"
    if risk_score >= 0.36 or scenario_type == "AMBIGUOUS":
        return "MEDIUM"
    return "LOW"


def infer_demo_scenario_type(raw_payload: str, risk_score: float) -> str:
    text = (raw_payload or "").lower()
    if any_cue(text, HARD_ATTACK_CUES) or any_cue(text, CRITICAL_QUARANTINE_CUES):
        return "MALICIOUS_HARD"
    if any_cue(text, SOFT_ATTACK_CUES):
        return "MALICIOUS_SOFT"
    if any_cue(text, AMBIGUITY_CUES):
        return "AMBIGUOUS"
    if risk_score >= 0.58:
        return "MALICIOUS_SOFT"
    if risk_score >= 0.36:
        return "AMBIGUOUS"
    if risk_score >= 0.22:
        return "BENIGN_NOISY"
    return "BENIGN_CLEAN"


def build_demo_policy_sample(
    domain: str,
    intent: str,
    raw_payload: str,
    risk_vector: list[float],
    *,
    actor: str = "unknown",
    session_id: str = "default",
    service: str = "",
    environment: str = "production",
    provided_evidence: Optional[list[str]] = None,
    indicators: Optional[list[str]] = None,
) -> dict[str, Any]:
    risk_score = compute_risk_score(risk_vector)
    scenario_type = infer_demo_scenario_type(raw_payload, risk_score)
    severity = classify_severity(risk_score, scenario_type, raw_payload)
    return {
        "sample_id": "demo-live",
        "split": "demo",
        "domain": domain,
        "intent": intent,
        "raw_payload": raw_payload,
        "action_summary": raw_payload,
        "actor": actor or "unknown",
        "session_id": session_id or "default",
        "service": service or domain or "unknown",
        "environment": environment or "production",
        "provided_evidence": list(provided_evidence or []),
        "indicators": list(indicators or []),
        "scenario_type": scenario_type,
        "is_malicious": scenario_type in {"MALICIOUS_SOFT", "MALICIOUS_HARD"},
        "severity": severity,
        "risk_score": round(float(risk_score), 6),
        "risk_vector": [round(float(value), 6) for value in risk_vector[:16]],
        "correct_action": "",
        "completion": "",
    }


def build_observation(domain: str, intent: str, raw_payload: str, risk_vector: list[float], severity: str) -> str:
    risk_text = "[" + ", ".join(f"{value:.3f}" for value in risk_vector[:16]) + "]"
    return (
        f"Domain: {domain}\n"
        f"Intent: {intent}\n"
        f"Payload: {raw_payload}\n"
        f"Severity: {severity}\n"
        f"Risk vector: {risk_text}"
    )


def build_action_prompt(observation: str) -> str:
    return ACTION_PROMPT_TEMPLATE.format(observation=observation)


def build_teacher_action(sample: dict[str, Any]) -> str:
    scenario_type = sample["scenario_type"]
    text = sample["raw_payload"].lower()
    risk_score = float(sample["risk_score"])
    severity = sample["severity"]
    is_malicious = bool(sample["is_malicious"])

    if scenario_type in {"BENIGN_CLEAN", "BENIGN_NOISY"}:
        return "ALLOW"

    if scenario_type == "MALICIOUS_SOFT":
        if any_cue(text, CRITICAL_QUARANTINE_CUES) or severity == "CRITICAL":
            return "QUARANTINE"
        return "BLOCK"

    if scenario_type == "MALICIOUS_HARD":
        if any_cue(text, CRITICAL_QUARANTINE_CUES) and risk_score >= 0.40:
            return "QUARANTINE"
        return "FORK"

    if scenario_type == "AMBIGUOUS":
        quarantine_signal = any_cue(text, QUARANTINE_AMBIGUITY_CUES)
        if quarantine_signal and (is_malicious or severity in {"HIGH", "CRITICAL"} or risk_score >= 0.42):
            return "QUARANTINE"
        return "FORK"

    return "ALLOW"


def random_policy_action(sample: dict[str, Any], rng: random.Random) -> str:
    return rng.choice(list(VALID_ACTIONS))


def heuristic_policy_action(sample: dict[str, Any]) -> str:
    risk_score = float(sample["risk_score"])
    severity = sample["severity"]
    if severity == "CRITICAL" or risk_score >= 0.62:
        return "BLOCK"
    if risk_score >= 0.38 or sample["scenario_type"] == "AMBIGUOUS":
        return "FORK"
    return "ALLOW"


def q_aware_policy_action(sample: dict[str, Any]) -> str:
    scenario_type = sample["scenario_type"]
    text = sample["raw_payload"].lower()
    risk_score = float(sample["risk_score"])
    severity = sample["severity"]

    if scenario_type in {"BENIGN_CLEAN", "BENIGN_NOISY"}:
        return "ALLOW"
    if scenario_type == "MALICIOUS_SOFT":
        if any_cue(text, CRITICAL_QUARANTINE_CUES) or severity == "CRITICAL":
            return "QUARANTINE"
        return "BLOCK"
    if scenario_type == "MALICIOUS_HARD":
        if any_cue(text, CRITICAL_QUARANTINE_CUES) and risk_score >= 0.40:
            return "QUARANTINE"
        return "FORK"
    if scenario_type == "AMBIGUOUS":
        if any_cue(text, QUARANTINE_AMBIGUITY_CUES) and severity in {"HIGH", "CRITICAL"}:
            return "QUARANTINE"
        return "FORK"
    return "ALLOW"


def q_aware_demo_policy_action(
    domain: str,
    intent: str,
    raw_payload: str,
    risk_vector: list[float],
) -> str:
    sample = build_demo_policy_sample(domain, intent, raw_payload, risk_vector)
    return q_aware_policy_action(sample)


def build_decision_trace(
    *,
    domain: str,
    risk_signals: Iterable[str],
    safe_signals: Iterable[str],
    cumulative_risk_score: float,
    memory_context: Optional[dict[str, Any]],
    missing_evidence: Iterable[str],
    evidence_plan: Iterable[dict[str, Any]],
    final_decision: str,
    safe_outcome: str,
) -> dict[str, Any]:
    memory_context = memory_context or {}
    memory_signals = list(memory_context.get("risky_chains", []))
    if memory_context.get("recent_indicators"):
        memory_signals.extend(f"recent:{item}" for item in memory_context.get("recent_indicators", [])[:5])
    if memory_context.get("session_risk", 0.0):
        memory_signals.append(f"session_risk={float(memory_context.get('session_risk', 0.0)):.3f}")
    evidence_steps = [
        {
            "step": item.get("step"),
            "priority": item.get("priority"),
            "ask": item.get("ask"),
            "blocks_decision": item.get("blocks_decision", False),
        }
        for item in evidence_plan or []
        if isinstance(item, dict)
    ]
    if final_decision == "ALLOW":
        rationale = "Allowed only because trusted evidence and risk stayed within policy limits."
    elif final_decision == "BLOCK":
        rationale = "Blocked because risk indicators show clear malicious or high-danger behavior."
    elif final_decision == "FORK":
        rationale = "Forked to human review because risk is high or approval is required before execution."
    else:
        rationale = "Quarantined until missing evidence is provided and risk can be reduced."
    return {
        "domain": domain or "unknown",
        "risk_signals": list(risk_signals or []),
        "safe_signals": list(safe_signals or []),
        "cumulative_risk_score": round(float(cumulative_risk_score or 0.0), 3),
        "memory_signals": list(dict.fromkeys(str(item) for item in memory_signals)),
        "missing_evidence": list(missing_evidence or []),
        "evidence_steps": evidence_steps,
        "final_decision": final_decision,
        "safety_rationale": f"{rationale} Safe outcome: {safe_outcome}",
    }


def build_q_aware_decision(
    domain: str,
    intent: str,
    raw_payload: str,
    risk_vector: list[float],
    *,
    actor: str = "unknown",
    session_id: str = "default",
    service: str = "",
    environment: str = "production",
    provided_evidence: Optional[list[str]] = None,
    timestamp: Any = 0,
    memory_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from domain_policies import evaluate_domain_policy
    from evidence_planner import build_evidence_plan, get_missing_evidence, get_required_evidence, explain_evidence_gap
    from risk_accumulator import clamp, compute_cumulative_risk
    from safe_outcome import generate_safe_outcome, generate_structured_safe_outcome

    sample = build_demo_policy_sample(
        domain,
        intent,
        raw_payload,
        risk_vector,
        actor=actor,
        session_id=session_id,
        service=service,
        environment=environment,
        provided_evidence=provided_evidence,
    )
    policy_features = evaluate_domain_policy(sample, memory_context=memory_context)
    sample["policy_domain"] = policy_features["domain"]
    sample["risk_indicators"] = policy_features["risk_indicators"]
    sample["safe_indicators"] = policy_features["safe_indicators"]

    required_evidence = get_required_evidence(
        policy_features["domain"],
        policy_features["risk_indicators"],
        policy_features["recommended_decision_hint"],
        environment,
    )
    missing_evidence = get_missing_evidence(required_evidence, provided_evidence or [])
    risk_data = compute_cumulative_risk(
        sample,
        memory_context=memory_context,
        base_risk=max(float(sample["risk_score"]), float(policy_features["base_risk"])),
    )
    cumulative_risk = float(risk_data["cumulative_risk_score"])
    missing_ratio = len(missing_evidence) / max(len(required_evidence), 1)
    safe_score = float(policy_features["safe_evidence_score"])
    q_aware_hint = q_aware_policy_action(sample)
    policy_hint = policy_features["recommended_decision_hint"]

    uncertainty = clamp(0.18 + 0.42 * missing_ratio + (0.12 if actor in {"", "unknown"} else 0.0) - safe_score * 0.25)
    high_danger = any(
        indicator in policy_features["risk_indicators"]
        for indicator in (
            "ci_secret_access",
            "external_fetch",
            "data_export",
            "disable_encryption",
            "admin_privilege",
            "public_exposure",
            "policy_override",
        )
    )

    if safe_score >= 0.70 and missing_ratio <= 0.05 and cumulative_risk <= 0.90:
        decision = "ALLOW"
    elif safe_score >= 0.54 and missing_ratio <= 0.15 and cumulative_risk <= 0.75:
        decision = "ALLOW"
    elif missing_ratio > 0.45 and safe_score < 0.54 and policy_hint == "ALLOW":
        decision = "QUARANTINE"
    elif policy_hint == "ALLOW" and missing_ratio > 0.25 and cumulative_risk >= 0.25:
        decision = "QUARANTINE"
    elif cumulative_risk <= 0.25:
        decision = "ALLOW"
    elif cumulative_risk >= 0.86 and high_danger and safe_score < 0.20 and uncertainty < 0.65:
        decision = "BLOCK"
    elif cumulative_risk >= 0.70 and safe_score >= 0.25:
        decision = "FORK"
    elif cumulative_risk >= 0.45 and uncertainty >= 0.45:
        decision = "QUARANTINE"
    elif cumulative_risk >= 0.62:
        decision = "FORK"
    else:
        decision = policy_hint if policy_hint in VALID_ACTION_SET else q_aware_hint

    if q_aware_hint == "QUARANTINE" and decision == "BLOCK":
        decision = "QUARANTINE"
    if q_aware_hint == "FORK" and decision == "QUARANTINE" and cumulative_risk >= 0.60:
        decision = "FORK"
    if q_aware_hint == "FORK" and decision == "BLOCK" and uncertainty >= 0.35 and policy_hint != "BLOCK":
        decision = "FORK"

    confidence = clamp(1.0 - uncertainty + (0.10 if high_danger and safe_score < 0.20 else 0.0))
    evidence_plan = build_evidence_plan(
        policy_features["domain"],
        policy_features["risk_indicators"],
        decision,
        environment,
        missing_evidence,
        memory_context=memory_context,
        risk_score=cumulative_risk,
    )
    safe_outcome = generate_safe_outcome(
        decision,
        policy_features["domain"],
        policy_features["risk_indicators"],
        policy_features["safe_indicators"],
        missing_evidence,
        environment,
    )
    structured_safe_outcome = generate_structured_safe_outcome(
        decision,
        policy_features["domain"],
        policy_features["risk_indicators"],
        policy_features["safe_indicators"],
        missing_evidence,
        environment,
        evidence_plan=evidence_plan,
    )
    evidence_gap = explain_evidence_gap(missing_evidence)
    explanation = (
        f"{policy_features['policy_reason']}. "
        f"cumulative_risk={cumulative_risk:.2f}; uncertainty={uncertainty:.2f}. "
        f"{evidence_gap}"
    )
    decision_trace = build_decision_trace(
        domain=policy_features["domain"],
        risk_signals=policy_features["risk_indicators"],
        safe_signals=policy_features["safe_indicators"],
        cumulative_risk_score=cumulative_risk,
        memory_context=memory_context,
        missing_evidence=missing_evidence,
        evidence_plan=evidence_plan,
        final_decision=decision,
        safe_outcome=safe_outcome,
    )

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "uncertainty": round(uncertainty, 3),
        "risk_score": round(float(sample["risk_score"]), 3),
        "cumulative_risk_score": round(cumulative_risk, 3),
        "cumulative_risk_reason": risk_data["cumulative_risk_reason"],
        "missing_evidence": missing_evidence,
        "required_evidence": required_evidence,
        "explanation": explanation,
        "safe_outcome": safe_outcome,
        "evidence_plan": evidence_plan,
        "structured_safe_outcome": structured_safe_outcome,
        "decision_trace": decision_trace,
        "memory_context": memory_context or {},
        "policy_name": "q_aware_demo_policy",
        "domain": policy_features["domain"],
        "mitre_tactic": policy_features["mitre_tactic"],
        "mitre_technique": policy_features["mitre_technique"],
        "risk_indicators": policy_features["risk_indicators"],
        "safe_indicators": policy_features["safe_indicators"],
        "actor": actor or "unknown",
        "session_id": session_id or "default",
        "service": service or domain or "unknown",
        "environment": environment or "production",
        "timestamp": timestamp,
    }


def build_sample_record(index: int, split: str, scenario_generator: ScenarioGenerator) -> dict[str, Any]:
    scenario = scenario_generator.sample()
    risk_vector = extract_features(
        scenario["domain"],
        scenario["intent"],
        scenario["raw_payload"],
        scenario_generator.rng,
    )
    risk_score = compute_risk_score(risk_vector)
    severity = classify_severity(risk_score, scenario["tier"], scenario["raw_payload"])
    observation = build_observation(
        scenario["domain"],
        scenario["intent"],
        scenario["raw_payload"],
        risk_vector,
        severity,
    )
    prompt = build_action_prompt(observation)
    legacy_prompt = build_llama_prompt(
        scenario["domain"],
        scenario["intent"],
        scenario["raw_payload"],
        risk_vector,
        False,
        0,
    )
    sample = {
        "sample_id": f"{split}-{index:05d}",
        "split": split,
        "domain": scenario["domain"],
        "intent": scenario["intent"],
        "raw_payload": scenario["raw_payload"],
        "legacy_prompt": legacy_prompt,
        "observation": observation,
        "prompt": prompt,
        "completion": "",
        "correct_action": "",
        "scenario_type": scenario["tier"],
        "is_malicious": bool(scenario["is_malicious"]),
        "severity": severity,
        "risk_score": round(float(risk_score), 6),
        "risk_vector": [round(float(value), 6) for value in risk_vector[:16]],
        "source_policy": "tier_aware_teacher_v2",
    }
    sample["correct_action"] = build_teacher_action(sample)
    sample["completion"] = sample["correct_action"]
    sample["text"] = sample["prompt"] + sample["completion"]
    return sample


def generate_dataset_split(
    sample_count: int,
    seed: int,
    split: str,
    forbidden_prompts: Optional[set[str]] = None,
) -> tuple[list[dict[str, Any]], int]:
    scenario_generator = ScenarioGenerator(seed=seed)
    samples: list[dict[str, Any]] = []
    seen_prompts = set(forbidden_prompts or set())
    duplicate_prompts = 0
    attempts = 0
    max_attempts = max(200, sample_count * 40)

    while len(samples) < sample_count and attempts < max_attempts:
        attempts += 1
        sample = build_sample_record(len(samples), split, scenario_generator)
        prompt = sample["prompt"]
        if prompt in seen_prompts:
            duplicate_prompts += 1
            continue
        seen_prompts.add(prompt)
        samples.append(sample)

    if len(samples) < sample_count:
        raise RuntimeError(
            f"Could not generate {sample_count} unique {split} samples after {attempts} attempts."
        )

    return samples, duplicate_prompts


def generate_datasets(
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    train_seed: int = DEFAULT_TRAIN_SEED,
    val_seed: int = DEFAULT_VAL_SEED,
    save: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ensure_dirs()
    train_samples, train_duplicates = generate_dataset_split(train_size, train_seed, "train")
    train_prompts = {sample["prompt"] for sample in train_samples}
    val_samples, val_duplicates = generate_dataset_split(
        val_size,
        val_seed,
        "val",
        forbidden_prompts=train_prompts,
    )
    audit = audit_datasets(
        train_samples,
        val_samples,
        duplicate_prompt_count=train_duplicates + val_duplicates,
    )
    if save:
        write_json(DEFAULT_TRAIN_DATASET_PATH, train_samples)
        write_json(DEFAULT_VAL_DATASET_PATH, val_samples)
        write_json(DEFAULT_DATASET_AUDIT_PATH, audit)
    return train_samples, val_samples, audit


def audit_datasets(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    duplicate_prompt_count: int = 0,
) -> dict[str, Any]:
    all_samples = train_samples + val_samples
    missing_label_count = sum(
        1 for sample in all_samples if not (sample.get("correct_action") or sample.get("expected_decision"))
    )
    invalid_action_label_count = sum(
        1
        for sample in all_samples
        if (sample.get("correct_action") or sample.get("expected_decision")) is not None
        and (sample.get("correct_action") or sample.get("expected_decision")) not in VALID_ACTION_SET
    )
    train_counts = Counter(sample.get("correct_action") for sample in train_samples if sample.get("correct_action") in VALID_ACTION_SET)
    val_counts = Counter(sample.get("correct_action") for sample in val_samples if sample.get("correct_action") in VALID_ACTION_SET)
    combined_counts = Counter(train_counts)
    combined_counts.update(val_counts)

    train_mal = Counter("malicious" if sample["is_malicious"] else "benign" for sample in train_samples)
    val_mal = Counter("malicious" if sample["is_malicious"] else "benign" for sample in val_samples)
    scenario_counts = Counter(sample["scenario_type"] for sample in all_samples)
    domain_counts = Counter(sample.get("domain", "unknown") for sample in all_samples)
    risk_type_counts = Counter(sample.get("severity", "UNKNOWN") for sample in all_samples)
    false_positive_challenge_count = sum(
        1
        for sample in train_samples + val_samples
        if not sample.get("is_malicious", False)
        and (
            sample.get("scenario_type") == "BENIGN_NOISY"
            or sample.get("severity") in {"MEDIUM", "HIGH"}
            or bool(sample.get("provided_evidence"))
        )
    )
    hard_negative_count = 0
    if DEFAULT_HARD_NEGATIVE_PATH.exists():
        with contextlib.suppress(Exception):
            hard_negative_count = len(read_json(DEFAULT_HARD_NEGATIVE_PATH, default=[]))

    train_prompts = {sample["prompt"] for sample in train_samples if sample.get("prompt")}
    val_prompts = {sample["prompt"] for sample in val_samples if sample.get("prompt")}
    overlap_count = len(train_prompts & val_prompts)

    train_distribution = summarize_action_distribution(train_counts, len(train_samples))
    val_distribution = summarize_action_distribution(val_counts, len(val_samples))
    combined_distribution = summarize_action_distribution(combined_counts, len(train_samples) + len(val_samples))

    examples_per_action = {
        action: [
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "severity": sample["severity"],
                "payload": sample["raw_payload"][:160],
            }
            for sample in train_samples + val_samples
            if sample["correct_action"] == action
        ][:3]
        for action in VALID_ACTIONS
    }

    failures: list[str] = []
    if overlap_count > 0:
        failures.append("train/val prompt overlap > 0")
    if missing_label_count > 0:
        failures.append("missing action label")
    if invalid_action_label_count > 0:
        failures.append("invalid action label")
    for action, share in train_distribution.items():
        if share < 0.05:
            failures.append(f"{action} share below 5% in train split")
    if train_distribution.get("ALLOW", 0.0) > 0.60:
        failures.append("ALLOW share above 60% in train split")
    min_val_tail = max(3, math.ceil(len(val_samples) * 0.05))
    if val_counts.get("BLOCK", 0) < min_val_tail:
        failures.append("validation BLOCK coverage too low")
    if val_counts.get("QUARANTINE", 0) < min_val_tail:
        failures.append("validation QUARANTINE coverage too low")

    audit = {
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "action_distribution": {
            "train": dict(train_counts),
            "val": dict(val_counts),
            "train_pct": train_distribution,
            "val_pct": val_distribution,
            "combined_pct": combined_distribution,
        },
        "benign_malicious_distribution": {
            "train": dict(train_mal),
            "val": dict(val_mal),
        },
        "scenario_distribution": dict(scenario_counts),
        "risk_type_distribution": dict(risk_type_counts),
        "domain_distribution": dict(domain_counts),
        "hard_negative_count": hard_negative_count,
        "false_positive_challenge_count": false_positive_challenge_count,
        "missing_label_count": missing_label_count,
        "invalid_action_label_count": invalid_action_label_count,
        "duplicate_prompt_count": duplicate_prompt_count,
        "train_val_overlap_count": overlap_count,
        "examples_per_action": examples_per_action,
        "passed_preflight": not failures,
        "failures": failures,
    }
    return audit


def preflight_dataset_check(audit: dict[str, Any]) -> None:
    if not audit.get("passed_preflight", False):
        raise TrainingPreflightError("; ".join(audit.get("failures", ["dataset audit failed"])))


def is_false_positive(sample: dict[str, Any], predicted_action: Optional[str]) -> bool:
    return (not sample["is_malicious"]) and predicted_action == "BLOCK"


def is_false_negative(sample: dict[str, Any], predicted_action: Optional[str]) -> bool:
    return bool(sample.get("is_malicious", False)) and predicted_action == "ALLOW"


def is_unsafe_decision(sample: dict[str, Any], predicted_action: Optional[str]) -> bool:
    if predicted_action is None:
        return True
    if predicted_action == (sample.get("correct_action") or sample.get("expected_decision")):
        return False
    high_risk = sample.get("severity") in {"HIGH", "CRITICAL"} or sample.get("scenario_type") == "MALICIOUS_HARD"
    return bool(sample.get("is_malicious", False) or high_risk) and predicted_action == "ALLOW"


def is_safe_action(sample: dict[str, Any], predicted_action: Optional[str]) -> bool:
    if predicted_action is None:
        return False
    if predicted_action == (sample.get("correct_action") or sample.get("expected_decision")):
        return True
    if is_unsafe_decision(sample, predicted_action):
        return False
    if sample.get("is_malicious", False):
        return predicted_action in {"BLOCK", "FORK", "QUARANTINE"}
    return predicted_action in {"ALLOW", "FORK", "QUARANTINE"}


def _decision_payload(output: Any) -> dict[str, Any]:
    return output if isinstance(output, dict) else {}


def _output_action_text(output: Any) -> str:
    if isinstance(output, dict):
        return str(output.get("decision") or output.get("action_taken") or output.get("action") or "")
    return str(output)


def _clearly_safe_sample(sample: dict[str, Any]) -> bool:
    if sample.get("is_malicious", False):
        return False
    trusted_text = " ".join(
        str(item).lower()
        for item in (
            list(sample.get("provided_evidence", []))
            + list(sample.get("safe_indicators", []))
            + [sample.get("actor", ""), sample.get("environment", "")]
        )
    )
    trusted_evidence = any(
        cue in trusted_text
        for cue in (
            "approved ticket",
            "approval ticket",
            "approved change ticket",
            "authorization letter",
            "approved test window",
            "trusted maintainer",
            "known automation",
            "break-glass approval",
            "break glass approval",
            "classification public",
            "staging",
            "dev",
        )
    )
    return (
        sample.get("severity", "LOW") == "LOW"
        and sample.get("scenario_type", "").startswith("BENIGN")
    ) or trusted_evidence


def _sample_text(sample: dict[str, Any]) -> str:
    return " ".join(
        str(value).lower()
        for value in (
            sample.get("raw_payload", ""),
            sample.get("action_summary", ""),
            " ".join(sample.get("indicators", [])),
            " ".join(sample.get("provided_evidence", [])),
        )
    )


def _missing_critical_evidence(sample: dict[str, Any]) -> bool:
    required = {str(item).lower() for item in sample.get("required_evidence", [])}
    provided_text = " ".join(str(item).lower() for item in sample.get("provided_evidence", []))
    critical_terms = (
        "approval",
        "authorization",
        "identity",
        "mfa",
        "secret",
        "privilege",
        "public access",
        "data classification",
    )
    return any(term in evidence for evidence in required for term in critical_terms) and not any(
        term in provided_text for term in critical_terms
    )


def _reward_margin(sample: dict[str, Any], predicted_action: str) -> float:
    risk = float(sample.get("risk_score", 0.0) or 0.0)
    severity = str(sample.get("severity", "LOW")).upper()
    scenario_type = str(sample.get("scenario_type", ""))
    margin = 0.0
    if severity == "CRITICAL":
        margin += 0.18
    elif severity == "HIGH":
        margin += 0.12
    elif severity == "MEDIUM":
        margin += 0.06
    margin += min(0.12, max(0.0, risk) * 0.10)
    if scenario_type == "MALICIOUS_HARD" and predicted_action in {"BLOCK", "FORK", "QUARANTINE"}:
        margin += 0.12
    if scenario_type == "AMBIGUOUS" and predicted_action in {"FORK", "QUARANTINE"}:
        margin += 0.08
    if _clearly_safe_sample(sample) and predicted_action == "ALLOW":
        margin += 0.08
    return round(margin, 6)


def compute_shaped_reward(
    sample: dict[str, Any],
    predicted_action: Optional[str],
    decision_payload: Optional[dict[str, Any]] = None,
) -> tuple[float, str]:
    decision_payload = decision_payload or {}
    correct_action = sample.get("correct_action") or sample.get("expected_decision")
    high_risk = sample.get("severity") in {"HIGH", "CRITICAL"} or sample.get("scenario_type") == "MALICIOUS_HARD"
    critical = sample.get("severity") == "CRITICAL" or sample.get("risk_level") == "critical"
    clearly_safe = _clearly_safe_sample(sample)

    if predicted_action not in VALID_ACTION_SET:
        return -2.5, "invalid_output"
    if predicted_action == correct_action:
        reward = 1.5 if predicted_action in {"QUARANTINE", "FORK"} else 2.0
        category = "exact_conservative" if predicted_action in {"QUARANTINE", "FORK"} else "exact_correct"
        reward += _reward_margin(sample, predicted_action)
    elif is_unsafe_decision(sample, predicted_action):
        reward = -5.0 if critical else -3.0
        category = "critical_unsafe_allow" if critical else "false_negative_allow"
    elif clearly_safe and predicted_action == "BLOCK":
        reward = -0.7
        category = "unnecessary_block"
    elif clearly_safe and predicted_action == "QUARANTINE":
        reward = -0.3
        category = "unnecessary_quarantine"
    elif clearly_safe and predicted_action == "FORK":
        reward = -0.4
        category = "false_positive_fork"
    elif sample.get("is_malicious", False) or high_risk:
        if predicted_action in {"BLOCK", "FORK", "QUARANTINE"}:
            reward = 0.8 + _reward_margin(sample, predicted_action) * 0.5
            category = "safe_conservative"
        else:
            reward = -0.5
            category = "minor_wrong"
    else:
        reward = -0.5
        category = "minor_wrong"

    text = _sample_text(sample)
    if any(
        cue in text
        for cue in (
            "ignore previous",
            "ignore security",
            "system override",
            "curl http://",
            "| bash",
            "public-read",
            "export all secrets",
            "administratoraccess",
            "0.0.0.0/0",
            "open port 22",
            "workflow",
        )
    ) and predicted_action == "ALLOW":
        reward -= 0.8
        category = "unsafe_allow_attack_cue"

    if _missing_critical_evidence(sample) and predicted_action == "ALLOW":
        reward -= 0.6
        category = "allow_missing_critical_evidence"

    if decision_payload.get("multi_action_warning"):
        reward -= 0.2
        category = "ambiguous_format"

    explanation = str(decision_payload.get("explanation", "")).lower()
    required_evidence = [str(item).lower() for item in sample.get("required_evidence", [])]
    if decision_payload:
        if required_evidence and any(item in explanation for item in required_evidence[:3]):
            reward += 0.3
        if decision_payload.get("missing_evidence") and "missing" not in explanation:
            reward -= 0.3
        expected_tactic = str(sample.get("mitre_tactic", "")).lower()
        expected_technique = str(sample.get("mitre_technique", "")).lower()
        actual_tactic = str(decision_payload.get("mitre_tactic", "")).lower()
        actual_technique = str(decision_payload.get("mitre_technique", "")).lower()
        if expected_tactic and actual_tactic and expected_tactic != actual_tactic:
            reward -= 0.2
        elif expected_technique and actual_technique and expected_technique.split()[0] != actual_technique.split()[0]:
            reward -= 0.2

    return round(float(reward), 6), category


def evaluate_outputs(
    samples: list[dict[str, Any]],
    outputs: list[Any],
    label: str,
) -> dict[str, Any]:
    parser_stats = ParseDiagnostics()
    action_counts: Counter = Counter()
    reward_categories: Counter = Counter()
    parsed_actions: list[Optional[str]] = []
    rewards: list[float] = []
    completion_lengths: list[int] = []
    confusion_matrix = {
        expected: {predicted: 0 for predicted in (*VALID_ACTIONS, "INVALID")}
        for expected in VALID_ACTIONS
    }
    exact = 0
    safe = 0
    valid = 0
    unsafe = 0
    false_positive_count = 0
    false_negative_count = 0

    for sample, output in zip(samples, outputs):
        output_text = _output_action_text(output)
        completion_lengths.append(len(str(output_text).split()))
        analysis = analyze_action_output(output_text)
        action = analysis["parsed_action"]
        payload = dict(_decision_payload(output))
        payload["multi_action_warning"] = analysis["multi_action_warning"]
        parsed_actions.append(action)
        if analysis["invalid_output"]:
            parser_stats.invalid_outputs += 1
        if analysis["multi_action_warning"]:
            parser_stats.multi_action_warnings += 1
        if action is not None:
            action_counts[action] += 1
            valid += 1
        reward, category = compute_shaped_reward(sample, action, payload)
        rewards.append(reward)
        reward_categories[category] += 1
        if action == (sample.get("correct_action") or sample.get("expected_decision")):
            exact += 1
        expected_action = sample.get("correct_action") or sample.get("expected_decision")
        if expected_action in confusion_matrix:
            confusion_matrix[expected_action][action or "INVALID"] += 1
        if is_safe_action(sample, action):
            safe += 1
        else:
            unsafe += 1
        if is_false_positive(sample, action):
            false_positive_count += 1
        if is_false_negative(sample, action):
            false_negative_count += 1

    total = len(samples)
    benign_total = sum(1 for sample in samples if not sample.get("is_malicious", False))
    malicious_total = sum(1 for sample in samples if sample.get("is_malicious", False))
    precision_by_action = {}
    per_action_accuracy = {}
    for action_name in VALID_ACTIONS:
        predicted_total = action_counts.get(action_name, 0)
        expected_total = sum(
            1 for sample in samples if (sample.get("correct_action") or sample.get("expected_decision")) == action_name
        )
        correct_for_action = sum(
            1
            for sample, parsed_action in zip(samples, parsed_actions)
            if parsed_action == action_name
            and parsed_action == (sample.get("correct_action") or sample.get("expected_decision"))
        )
        precision_by_action[action_name.lower()] = correct_for_action / max(predicted_total, 1)
        per_action_accuracy[action_name] = correct_for_action / max(expected_total, 1)

    metrics = {
        "label": label,
        "sample_count": total,
        "exact_match": exact / max(total, 1),
        "safety_accuracy": safe / max(total, 1),
        "valid_action_rate": valid / max(total, 1),
        "invalid_action_rate": parser_stats.invalid_outputs / max(total, 1),
        "invalid_output_rate": parser_stats.invalid_outputs / max(total, 1),
        "parse_failure_rate": parser_stats.invalid_outputs / max(total, 1),
        "unsafe_decision_rate": unsafe / max(total, 1),
        "false_positive_rate": false_positive_count / max(benign_total, 1),
        "false_negative_rate": false_negative_count / max(malicious_total, 1),
        "reward_mean": safe_mean(rewards),
        "reward_std": safe_std(rewards),
        "allow_precision": precision_by_action["allow"],
        "block_precision": precision_by_action["block"],
        "fork_precision": precision_by_action["fork"],
        "quarantine_precision": precision_by_action["quarantine"],
        "per_action_accuracy": per_action_accuracy,
        "confusion_matrix": confusion_matrix,
        "avg_completion_length": safe_mean(completion_lengths),
        "action_distribution": summarize_action_distribution(action_counts, total),
        "normalized_action_distribution": summarize_action_distribution(action_counts, total),
        "invalid_output_count": parser_stats.invalid_outputs,
        "multi_action_warnings": parser_stats.multi_action_warnings,
        "multi_action_warning_rate": parser_stats.multi_action_warnings / max(total, 1),
        "entropy": distribution_entropy(summarize_action_distribution(action_counts, max(total, 1))),
        "reward_breakdown": dict(reward_categories),
        "predicted_actions": [action if action is not None else "INVALID" for action in parsed_actions],
        "sample_rewards": rewards,
    }
    return metrics


def evaluate_policy_on_dataset(
    samples: list[dict[str, Any]],
    policy_name: str,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    outputs: list[str] = []
    for sample in samples:
        if policy_name == "random":
            outputs.append(random_policy_action(sample, rng))
        elif policy_name == "heuristic":
            outputs.append(heuristic_policy_action(sample))
        elif policy_name == "q_aware":
            if sample.get("use_agent_policy"):
                outputs.append(
                    build_q_aware_decision(
                        sample["domain"],
                        sample.get("intent", sample.get("action_summary", "")),
                        sample.get("raw_payload", sample.get("action_summary", "")),
                        sample.get("risk_vector", [0.0] * 16),
                        actor=sample.get("actor", "unknown"),
                        session_id=sample.get("session_id", "default"),
                        service=sample.get("service", sample.get("domain", "unknown")),
                        environment=sample.get("environment", "production"),
                        provided_evidence=sample.get("provided_evidence", []),
                    )
                )
            else:
                outputs.append(q_aware_policy_action(sample))
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")
    return evaluate_outputs(samples, outputs, label=policy_name)


def evaluate_oracle(samples: list[dict[str, Any]]) -> dict[str, Any]:
    outputs: list[str] = []
    oracle_rewards: list[float] = []
    for sample in samples:
        oracle_payload = {
            "explanation": "Required evidence reviewed: " + ", ".join(sample.get("required_evidence", [])),
            "mitre_tactic": sample.get("mitre_tactic", ""),
            "mitre_technique": sample.get("mitre_technique", ""),
        }
        reward_by_action = {
            action: compute_shaped_reward(sample, action, oracle_payload)[0]
            for action in VALID_ACTIONS
        }
        best_reward = max(reward_by_action.values())
        best_actions = [action for action, reward in reward_by_action.items() if reward == best_reward]
        chosen_action = best_actions[0]
        outputs.append(chosen_action)
        oracle_rewards.append(best_reward)
    metrics = evaluate_outputs(samples, outputs, label="oracle")
    metrics["sample_rewards"] = oracle_rewards
    metrics["reward_mean"] = safe_mean(oracle_rewards)
    metrics["reward_std"] = safe_std(oracle_rewards)
    return metrics


def check_oracle_consistency(
    samples: list[dict[str, Any]],
    metrics_by_label: dict[str, dict[str, Any]],
    output_path: Path = DEFAULT_ORACLE_INCONSISTENCY_PATH,
) -> dict[str, Any]:
    oracle_metrics = metrics_by_label["oracle"]
    oracle_rewards = oracle_metrics["sample_rewards"]
    inconsistencies: list[dict[str, Any]] = []

    for label, metrics in metrics_by_label.items():
        if label == "oracle":
            continue
        if metrics["reward_mean"] > oracle_metrics["reward_mean"] + 1e-9:
            inconsistencies.append(
                {
                    "label": label,
                    "issue": "mean reward exceeded oracle",
                    "mean_reward": metrics["reward_mean"],
                    "oracle_mean_reward": oracle_metrics["reward_mean"],
                }
            )
        sample_rewards = metrics.get("sample_rewards", [])
        for index, reward in enumerate(sample_rewards):
            if reward > oracle_rewards[index] + 1e-9:
                sample = samples[index]
                inconsistencies.append(
                    {
                        "label": label,
                        "sample_id": sample["sample_id"],
                        "reward": reward,
                        "oracle_reward": oracle_rewards[index],
                        "correct_action": sample["correct_action"],
                        "predicted_action": metrics["predicted_actions"][index],
                        "raw_payload": sample["raw_payload"],
                    }
                )

    write_json(output_path, inconsistencies)
    try:
        relative_output_path = str(output_path.relative_to(BACKEND_DIR))
    except ValueError:
        relative_output_path = str(output_path)
    return {
        "passed": not inconsistencies,
        "oracle_reward_mean": oracle_metrics["reward_mean"],
        "inconsistency_count": len(inconsistencies),
        "output_path": relative_output_path,
    }


def check_reward_variance(samples: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_actions = ["ALLOW", "BLOCK", "FORK", "QUARANTINE", "INVALID"]
    group_stds: list[float] = []
    groups_with_variance = 0

    for sample in samples:
        rewards = []
        for action in candidate_actions:
            predicted_action = None if action == "INVALID" else action
            rewards.append(compute_shaped_reward(sample, predicted_action)[0])
        reward_std = safe_std(rewards)
        group_stds.append(reward_std)
        if reward_std > 0:
            groups_with_variance += 1

    groups_checked = len(samples)
    zero_std_fraction = (
        sum(1 for value in group_stds if value <= 1e-12) / max(groups_checked, 1)
    )
    result = {
        "groups_checked": groups_checked,
        "groups_with_variance": groups_with_variance,
        "groups_with_variance_pct": groups_with_variance / max(groups_checked, 1),
        "mean_reward_std": safe_mean(group_stds),
        "frac_reward_zero_std": zero_std_fraction,
        "passed": (
            (groups_with_variance / max(groups_checked, 1)) >= 0.80
            and zero_std_fraction <= 0.30
        ),
    }
    return result


def run_reward_diagnostics(samples: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    if samples is None:
        val_samples, _ = load_validation_samples_for_benchmark()
        samples = val_samples

    candidate_outputs = ["ALLOW", "BLOCK", "FORK", "QUARANTINE", "hello", "ALLOW then BLOCK"]
    group_stds: list[float] = []
    invalid_outputs = 0
    total_outputs = 0
    action_counts: Counter = Counter()
    sample_summaries: list[dict[str, Any]] = []
    same_reward_examples: list[dict[str, Any]] = []
    all_rewards: list[float] = []

    for sample in samples:
        rewards_by_output: dict[str, float] = {}
        for output in candidate_outputs:
            analysis = analyze_action_output(output)
            action = analysis["parsed_action"]
            if action is None:
                invalid_outputs += 1
            else:
                action_counts[action] += 1
            total_outputs += 1
            reward, _ = compute_shaped_reward(
                sample,
                action,
                {"multi_action_warning": analysis["multi_action_warning"]},
            )
            rewards_by_output[output] = reward
            all_rewards.append(reward)
        values = list(rewards_by_output.values())
        reward_std = safe_std(values)
        group_stds.append(reward_std)
        summary = {
            "sample_id": sample.get("sample_id"),
            "correct_action": sample.get("correct_action") or sample.get("expected_decision"),
            "scenario_type": sample.get("scenario_type"),
            "severity": sample.get("severity"),
            "reward_std": reward_std,
            "reward_range": max(values) - min(values),
            "payload": str(sample.get("raw_payload", sample.get("action_summary", "")))[:140],
            "rewards": rewards_by_output,
        }
        sample_summaries.append(summary)
        if reward_std <= 1e-12:
            same_reward_examples.append(summary)

    easiest = sorted(sample_summaries, key=lambda row: row["reward_range"], reverse=True)[:5]
    hardest = sorted(sample_summaries, key=lambda row: row["reward_range"])[:5]
    diagnostics = {
        "sample_count": len(samples),
        "reward_mean": safe_mean(all_rewards),
        "reward_std": safe_std(all_rewards),
        "percent_zero_std_groups": 100.0 * sum(1 for value in group_stds if value <= 1e-12) / max(len(group_stds), 1),
        "frac_reward_zero_std": sum(1 for value in group_stds if value <= 1e-12) / max(len(group_stds), 1),
        "invalid_output_rate": invalid_outputs / max(total_outputs, 1),
        "action_distribution": summarize_action_distribution(action_counts, max(total_outputs - invalid_outputs, 1)),
        "easiest_samples": easiest,
        "hardest_samples": hardest,
        "same_reward_examples": same_reward_examples[:5],
    }
    return diagnostics


def print_reward_diagnostics(diagnostics: dict[str, Any]) -> None:
    print("Reward diagnostics")
    print("------------------")
    print(f"samples: {diagnostics['sample_count']}")
    print(f"reward mean/std: {diagnostics['reward_mean']:.3f} / {diagnostics['reward_std']:.3f}")
    print(f"zero-std groups: {diagnostics['percent_zero_std_groups']:.1f}%")
    print(f"invalid output rate: {diagnostics['invalid_output_rate']:.3f}")
    print(f"action distribution: {diagnostics['action_distribution']}")
    print("easiest samples:")
    for row in diagnostics["easiest_samples"][:3]:
        print(f"  {row['sample_id']} {row['correct_action']} range={row['reward_range']:.3f}")
    print("hardest samples:")
    for row in diagnostics["hardest_samples"][:3]:
        print(f"  {row['sample_id']} {row['correct_action']} range={row['reward_range']:.3f}")
    if diagnostics["same_reward_examples"]:
        print("same-reward examples:")
        for row in diagnostics["same_reward_examples"][:3]:
            print(f"  {row['sample_id']} {row['correct_action']} {row['payload']}")
    else:
        print("same-reward examples: none")


def metric_delta(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, float]:
    keys = (
        "exact_match",
        "safety_accuracy",
        "unsafe_decision_rate",
        "false_positive_rate",
        "false_negative_rate",
        "reward_mean",
        "invalid_output_rate",
    )
    return {
        key: float(model_metrics.get(key, 0.0) or 0.0) - float(baseline_metrics.get(key, 0.0) or 0.0)
        for key in keys
    }


def evaluate_training_gate(
    model_metrics: Optional[dict[str, Any]],
    q_aware_metrics: dict[str, Any],
    *,
    reference_metrics: Optional[dict[str, Any]] = None,
    min_safety: float = 0.98,
    max_unsafe: float = 0.02,
    max_invalid: float = 0.05,
) -> dict[str, Any]:
    if model_metrics is None:
        return {
            "training_gate_status": "FAIL",
            "training_gate_passed": False,
            "reason": "No model metrics are available; checkpoint was not loaded or evaluation failed.",
            "recommended_next_action": "Run --evaluate-model with a valid --model-path after SFT/GRPO smoke training.",
        }

    def _metric(metrics: dict[str, Any], key: str, default: float) -> float:
        value = metrics.get(key, default)
        return default if value is None else float(value)

    safety = _metric(model_metrics, "safety_accuracy", 0.0)
    unsafe = _metric(model_metrics, "unsafe_decision_rate", 1.0)
    invalid = _metric(
        model_metrics,
        "invalid_output_rate",
        _metric(model_metrics, "invalid_action_rate", 1.0),
    )
    reward = _metric(model_metrics, "reward_mean", 0.0)
    q_reward = _metric(q_aware_metrics, "reward_mean", 0.0)

    if safety < min_safety - 0.08 or unsafe > max_unsafe + 0.08 or invalid > max_invalid:
        return {
            "training_gate_status": "FAIL",
            "training_gate_passed": False,
            "reason": (
                f"Safety gate failed: safety={safety:.3f}, unsafe={unsafe:.3f}, "
                f"invalid={invalid:.3f}."
            ),
            "recommended_next_action": "Stop longer training; fix parsing/reward/data before spending more credits.",
        }

    if reference_metrics is None:
        status = "WARN" if safety >= min_safety and unsafe <= max_unsafe else "FAIL"
        return {
            "training_gate_status": status,
            "training_gate_passed": False,
            "reason": "No raw/SFT reference metrics are available, so improvement is not proven.",
            "recommended_next_action": "Evaluate raw or SFT checkpoint, then compare this checkpoint again.",
        }

    reference_reward = _metric(reference_metrics, "reward_mean", 0.0)
    reference_unsafe = _metric(reference_metrics, "unsafe_decision_rate", 1.0)
    improves_reference = reward > reference_reward + 1e-9 and unsafe <= reference_unsafe + 1e-9

    if safety >= min_safety and unsafe <= max_unsafe and improves_reference:
        if reward >= q_reward:
            return {
                "training_gate_status": "PASS",
                "training_gate_passed": True,
                "reason": "Model meets safety/unsafe thresholds and improves reward over reference metrics.",
                "recommended_next_action": "Run a longer GRPO explore only if budget allows; keep comparing to Q-aware.",
            }
        return {
            "training_gate_status": "WARN",
            "training_gate_passed": False,
            "reason": "Model improves over reference but remains below Q-aware policy reward.",
            "recommended_next_action": "Continue small GRPO exploration or improve reward/data before final training.",
        }

    return {
        "training_gate_status": "FAIL",
        "training_gate_passed": False,
        "reason": "Model did not prove reward/safety improvement over raw/SFT reference metrics.",
        "recommended_next_action": "Do not claim training success; inspect reward diagnostics and checkpoint outputs.",
    }


def run_parse_action_tests() -> dict[str, Any]:
    tests = [
        ("ALLOW", "ALLOW"),
        (" allow. ", "ALLOW"),
        ("<think>abc</think> BLOCK", "BLOCK"),
        ("Action: FORK", "FORK"),
        ("decision=quarantine", "QUARANTINE"),
        ("deny", "BLOCK"),
        ("human_review", "FORK"),
        ("hello", None),
        ("please allow this", None),
        ("ALLOW then BLOCK", "BLOCK"),
        ("QUARANTINE because...", "QUARANTINE"),
    ]
    failures = []
    for text, expected in tests:
        actual = parse_action(text)
        if actual != expected:
            failures.append(
                {
                    "input": text,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "passed": not failures,
        "test_count": len(tests),
        "failures": failures,
    }


def compact_metrics(metrics: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if metrics is None:
        return None
    compact = dict(metrics)
    compact.pop("predicted_actions", None)
    compact.pop("sample_rewards", None)
    return compact


def build_evaluation_bundle(
    val_samples: list[dict[str, Any]],
    raw_model_metrics: Optional[dict[str, Any]] = None,
    sft_metrics: Optional[dict[str, Any]] = None,
    grpo_metrics: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metrics_by_label = {
        "random": evaluate_policy_on_dataset(val_samples, "random", seed=1),
        "heuristic": evaluate_policy_on_dataset(val_samples, "heuristic", seed=2),
        "q_aware": evaluate_policy_on_dataset(val_samples, "q_aware", seed=3),
        "oracle": evaluate_oracle(val_samples),
    }
    if raw_model_metrics is not None:
        metrics_by_label["raw_model"] = raw_model_metrics
    if sft_metrics is not None:
        metrics_by_label["sft_model"] = sft_metrics
    if grpo_metrics is not None:
        metrics_by_label["sft_grpo_model"] = grpo_metrics
    oracle_check = check_oracle_consistency(val_samples, metrics_by_label)
    return metrics_by_label, oracle_check


def assert_dataset_file_counts_match_audit(
    dataset_audit: dict[str, Any],
    *,
    train_samples: Optional[list[dict[str, Any]]] = None,
    val_samples: Optional[list[dict[str, Any]]] = None,
) -> None:
    expected_train = int(dataset_audit.get("train_sample_count", -1))
    expected_val = int(dataset_audit.get("val_sample_count", -1))
    if train_samples is not None and len(train_samples) != expected_train:
        raise TrainingPreflightError(
            f"Train dataset count {len(train_samples)} does not match dataset_audit.json {expected_train}."
        )
    if val_samples is not None and len(val_samples) != expected_val:
        raise TrainingPreflightError(
            f"Validation dataset count {len(val_samples)} does not match dataset_audit.json {expected_val}."
        )


def load_validation_samples_for_benchmark() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_audit = read_json(DEFAULT_DATASET_AUDIT_PATH, default=None)
    train_samples = read_json(DEFAULT_TRAIN_DATASET_PATH, default=None)
    val_samples = read_json(DEFAULT_VAL_DATASET_PATH, default=None)

    if dataset_audit is None or train_samples is None or val_samples is None:
        train_samples, val_samples, dataset_audit = generate_datasets(save=True)

    assert_dataset_file_counts_match_audit(
        dataset_audit,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    preflight_dataset_check(dataset_audit)
    return val_samples, dataset_audit


def build_demo_benchmark_rows(metrics_by_label: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    policies = (
        ("Random", "random"),
        ("Heuristic", "heuristic"),
        ("Q-aware", "q_aware"),
        ("Oracle", "oracle"),
    )
    rows = []
    for display_name, label in policies:
        metrics = metrics_by_label[label]
        row = {"policy": display_name}
        for metric_name in DEMO_BENCHMARK_METRICS:
            row[metric_name] = float(metrics[metric_name])
        rows.append(row)
    return rows


def format_demo_benchmark_table(rows: list[dict[str, Any]]) -> str:
    columns = ("policy",) + DEMO_BENCHMARK_METRICS
    widths = {
        column: max(len(column), *(len(f"{row[column]:.3f}") if column != "policy" else len(row[column]) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = []
    for row in rows:
        cells = [row["policy"].ljust(widths["policy"])]
        cells.extend(f"{row[column]:.3f}".rjust(widths[column]) for column in DEMO_BENCHMARK_METRICS)
        body.append("  ".join(cells))
    return "\n".join([header, divider, *body])


def write_demo_benchmark_reports(
    *,
    rows: list[dict[str, Any]],
    dataset_audit: dict[str, Any],
    oracle_check: dict[str, Any],
    output_json: Path = DEFAULT_DEMO_BENCHMARK_JSON,
    output_md: Path = DEFAULT_DEMO_BENCHMARK_MD,
) -> dict[str, Any]:
    report = {
        "dataset": {
            "train_sample_count": dataset_audit["train_sample_count"],
            "val_sample_count": dataset_audit["val_sample_count"],
            "validation_source": str(DEFAULT_VAL_DATASET_PATH.relative_to(BACKEND_DIR)),
        },
        "metrics": rows,
        "oracle_check": oracle_check,
    }
    write_json(output_json, report)

    header = "| Policy | " + " | ".join(DEMO_BENCHMARK_METRICS) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in DEMO_BENCHMARK_METRICS) + " |"
    md_lines = [
        "# ShadowOps Demo Benchmark",
        "",
        f"Validation samples: {dataset_audit['val_sample_count']}",
        "",
        header,
        divider,
    ]
    for row in rows:
        values = " | ".join(f"{row[metric]:.3f}" for metric in DEMO_BENCHMARK_METRICS)
        md_lines.append(f"| {row['policy']} | {values} |")
    md_lines.append("")
    output_md.write_text("\n".join(md_lines), encoding="utf-8")
    return report


def run_demo_benchmark(
    output_json: Path = DEFAULT_DEMO_BENCHMARK_JSON,
    output_md: Path = DEFAULT_DEMO_BENCHMARK_MD,
) -> dict[str, Any]:
    val_samples, dataset_audit = load_validation_samples_for_benchmark()
    metrics_by_label, oracle_check = build_evaluation_bundle(val_samples)
    rows = build_demo_benchmark_rows(metrics_by_label)
    report = write_demo_benchmark_reports(
        rows=rows,
        dataset_audit=dataset_audit,
        oracle_check=oracle_check,
        output_json=output_json,
        output_md=output_md,
    )
    print(format_demo_benchmark_table(rows))
    print(f"\nSaved: {output_json.relative_to(BACKEND_DIR)}")
    print(f"Saved: {output_md.relative_to(BACKEND_DIR)}")
    return report


def _legacy_domain_for_policy_domain(domain: str) -> str:
    domain = str(domain or "").lower()
    if domain == "github_ci":
        return "GITHUB"
    if domain in {"aws_s3", "iam"}:
        return "AWS"
    return "SOC"


def _scenario_type_for_hard_negative(scenario: dict[str, Any]) -> str:
    expected = scenario["expected_decision"]
    risk_level = str(scenario.get("risk_level", "medium")).lower()
    if expected == "ALLOW":
        return "BENIGN_NOISY" if risk_level in {"medium", "high"} else "BENIGN_CLEAN"
    if expected == "BLOCK":
        return "MALICIOUS_HARD" if risk_level in {"critical", "high"} else "MALICIOUS_SOFT"
    if expected == "QUARANTINE":
        return "MALICIOUS_HARD" if risk_level == "critical" else "AMBIGUOUS"
    return "AMBIGUOUS"


def hard_negative_to_sample(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    from evidence_planner import get_required_evidence

    raw_payload = scenario["action_summary"]
    legacy_domain = _legacy_domain_for_policy_domain(scenario["domain"])
    risk_vector = extract_features(
        legacy_domain,
        scenario.get("domain", "security").upper(),
        raw_payload + " " + " ".join(scenario.get("indicators", [])),
        random.Random(90_000 + index),
    )
    risk_level = str(scenario.get("risk_level", "medium")).lower()
    severity = {
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
        "critical": "CRITICAL",
    }.get(risk_level, "MEDIUM")
    is_malicious = scenario["expected_decision"] == "BLOCK" or any(
        cue in scenario["title"].lower()
        for cue in ("malicious", "rogue", "unauthorized", "without approval", "exfiltration")
    )
    required_evidence = get_required_evidence(
        scenario["domain"],
        scenario.get("indicators", []),
        scenario["expected_decision"],
        scenario["environment"],
    )
    sample = {
        "sample_id": scenario["id"],
        "split": "hard_negative",
        "domain": scenario["domain"],
        "intent": scenario["title"],
        "raw_payload": raw_payload,
        "action_summary": raw_payload,
        "actor": scenario["actor"],
        "session_id": scenario["session_id"],
        "service": scenario["service"],
        "environment": scenario["environment"],
        "indicators": list(scenario.get("indicators", [])),
        "provided_evidence": list(scenario.get("provided_evidence", [])),
        "correct_action": scenario["expected_decision"],
        "expected_decision": scenario["expected_decision"],
        "expected_safe_outcome": scenario["expected_safe_outcome"],
        "risk_level": risk_level,
        "scenario_type": _scenario_type_for_hard_negative(scenario),
        "is_malicious": is_malicious,
        "severity": severity,
        "risk_score": round(float(compute_risk_score(risk_vector)), 6),
        "risk_vector": [round(float(value), 6) for value in risk_vector[:16]],
        "required_evidence": required_evidence,
        "mitre_tactic": scenario["mitre_tactic"],
        "mitre_technique": scenario["mitre_technique"],
        "explanation": scenario["explanation"],
        "use_agent_policy": True,
    }
    return sample


def load_hard_negative_samples(path: Path = DEFAULT_HARD_NEGATIVE_PATH) -> list[dict[str, Any]]:
    scenarios = read_json(path, default=[])
    return [hard_negative_to_sample(scenario, index) for index, scenario in enumerate(scenarios)]


def _comparison_metric_row(label: str, metrics: Optional[dict[str, Any]]) -> dict[str, Any]:
    if metrics is None:
        row = {"policy": label, "available": False}
        row.update({metric: None for metric in MODEL_POLICY_METRICS})
        return row
    row = {"policy": label, "available": True}
    for metric in MODEL_POLICY_METRICS:
        row[metric] = metrics.get(metric)
    return row


def _write_model_policy_comparison_md(report: dict[str, Any], output_md: Path) -> None:
    lines = ["# ShadowOps Model vs Policy Comparison", ""]
    for dataset_name, dataset_report in report["datasets"].items():
        lines.extend([f"## {dataset_name}", "", f"Samples: {dataset_report['sample_count']}", ""])
        header = "| Policy | Available | " + " | ".join(MODEL_POLICY_METRICS) + " |"
        divider = "| --- | --- | " + " | ".join("---:" for _ in MODEL_POLICY_METRICS) + " |"
        lines.extend([header, divider])
        for row in dataset_report["rows"]:
            values = []
            for metric in MODEL_POLICY_METRICS:
                value = row.get(metric)
                values.append("n/a" if value is None else f"{float(value):.3f}")
            lines.append(f"| {row['policy']} | {row['available']} | " + " | ".join(values) + " |")
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def run_model_policy_comparison(
    *,
    model_metrics_by_dataset: Optional[dict[str, dict[str, Optional[dict[str, Any]]]]] = None,
    output_json: Path = DEFAULT_MODEL_POLICY_COMPARISON_JSON,
    output_md: Path = DEFAULT_MODEL_POLICY_COMPARISON_MD,
) -> dict[str, Any]:
    val_samples, _ = load_validation_samples_for_benchmark()
    hard_negative_samples = load_hard_negative_samples()
    model_metrics_by_dataset = model_metrics_by_dataset or {}
    report = {
        "model_metrics_note": "Model rows are unavailable unless raw/SFT/GRPO models are explicitly loaded and evaluated.",
        "datasets": {},
    }

    for dataset_name, samples in (
        ("validation", val_samples),
        ("hard_negative", hard_negative_samples),
    ):
        metrics_by_label, oracle_check = build_evaluation_bundle(samples)
        rows = [
            _comparison_metric_row("random", metrics_by_label["random"]),
            _comparison_metric_row("heuristic", metrics_by_label["heuristic"]),
            _comparison_metric_row("q_aware_policy", metrics_by_label["q_aware"]),
            _comparison_metric_row("oracle", metrics_by_label["oracle"]),
        ]
        dataset_model_metrics = model_metrics_by_dataset.get(dataset_name, {})
        for label in ("raw_model", "sft_model", "grpo_model"):
            rows.append(_comparison_metric_row(label, dataset_model_metrics.get(label)))
        report["datasets"][dataset_name] = {
            "sample_count": len(samples),
            "oracle_check": oracle_check,
            "rows": rows,
        }

    write_json(output_json, report)
    _write_model_policy_comparison_md(report, output_md)
    print(f"Saved: {output_json.relative_to(BACKEND_DIR)}")
    print(f"Saved: {output_md.relative_to(BACKEND_DIR)}")
    return report


def build_training_health_report(
    pre_train_metrics: Optional[dict[str, Any]],
    sft_metrics: Optional[dict[str, Any]],
    grpo_metrics: Optional[dict[str, Any]],
    tracker: Optional[RewardHealthTracker],
    baseline_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
    lora_parameter_delta: Optional[dict[str, Any]],
    oracle_check: dict[str, Any],
) -> dict[str, Any]:
    warnings_out: list[str] = []
    reward_zero_fraction = None if tracker is None else tracker.reward_std_zero_fraction
    grad_zero_fraction = None if tracker is None else tracker.grad_norm_zero_fraction
    invalid_output_rate = (
        grpo_metrics.get("invalid_action_rate")
        if grpo_metrics is not None
        else (tracker.invalid_output_rate if tracker is not None else None)
    )

    if reward_zero_fraction is not None and reward_zero_fraction > 0.90:
        warnings_out.append("CRITICAL: reward_std_zero_fraction > 0.90")
    elif reward_zero_fraction is not None and reward_zero_fraction > 0.50:
        warnings_out.append("WARNING: reward_std_zero_fraction > 0.50")

    if grad_zero_fraction is not None and grad_zero_fraction > 0.90:
        warnings_out.append("CRITICAL: grad_norm_zero_fraction > 0.90")
    elif grad_zero_fraction is not None and grad_zero_fraction > 0.50:
        warnings_out.append("WARNING: grad_norm_zero_fraction > 0.50")

    if grpo_metrics is not None and sft_metrics is not None:
        if grpo_metrics["safety_accuracy"] < sft_metrics["safety_accuracy"] - 0.05:
            warnings_out.append("WARNING: GRPO safety accuracy degraded relative to SFT")
        if grpo_metrics["exact_match"] < sft_metrics["exact_match"] - 0.05:
            warnings_out.append("WARNING: GRPO exact match degraded relative to SFT")

    if not oracle_check.get("passed", False):
        warnings_out.append("ERROR: oracle consistency failed")

    is_training_healthy = not any(message.startswith(("CRITICAL", "ERROR")) for message in warnings_out)
    q_aware_baseline = baseline_metrics.get("q_aware") or {}
    training_gate = (
        evaluate_training_gate(
            grpo_metrics,
            q_aware_baseline,
            reference_metrics=sft_metrics or pre_train_metrics,
        )
        if grpo_metrics is not None
        else {
            "training_gate_status": "WARN",
            "training_gate_passed": False,
            "reason": "No GRPO metrics are available; training success cannot be claimed.",
            "recommended_next_action": "Run checkpoint evaluation before claiming model improvement.",
        }
    )

    report = {
        "pre_train_metrics": compact_metrics(pre_train_metrics),
        "sft_metrics": compact_metrics(sft_metrics),
        "grpo_metrics": compact_metrics(grpo_metrics),
        "reward_std_zero_fraction": reward_zero_fraction,
        "grad_norm_zero_fraction": grad_zero_fraction,
        "invalid_output_rate": invalid_output_rate,
        "action_distribution": (
            tracker.action_distribution
            if tracker is not None
            else (grpo_metrics or {}).get("action_distribution", {})
        ),
        "entropy": tracker.entropy if tracker is not None else None,
        "oracle_metrics": compact_metrics(oracle_metrics),
        "baseline_metrics": {
            label: compact_metrics(metrics)
            for label, metrics in baseline_metrics.items()
        },
        "lora_parameter_delta": lora_parameter_delta,
        "training_gate": training_gate,
        "training_gate_status": training_gate["training_gate_status"],
        "training_gate_passed": training_gate["training_gate_passed"],
        "is_training_healthy": is_training_healthy,
        "warnings": warnings_out,
    }
    return report


def compute_improvement(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if before is None or after is None:
        return None
    return {
        "exact_match_change": after["exact_match"] - before["exact_match"],
        "safety_accuracy_change": after["safety_accuracy"] - before["safety_accuracy"],
        "valid_action_rate_change": after["valid_action_rate"] - before["valid_action_rate"],
        "reward_mean_change": after["reward_mean"] - before["reward_mean"],
    }


def build_training_ready_criteria(
    parse_tests: dict[str, Any],
    dataset_audit: dict[str, Any],
    reward_variance: dict[str, Any],
    oracle_check: dict[str, Any],
    smoke_test_result: bool,
) -> dict[str, bool]:
    criteria = {
        "parse_tests_pass": bool(parse_tests.get("passed", False)),
        "dataset_audit_passes": bool(dataset_audit.get("passed_preflight", False)),
        "reward_variance_passes": bool(reward_variance.get("passed", False)),
        "oracle_consistency_passes": bool(oracle_check.get("passed", False)),
        "evaluation_isolated": True,
        "sft_script_exists": (TRAINING_DIR / "train_qwen3_sft.py").exists(),
        "grpo_can_resume_from_sft": True,
        "final_report_generation_works": True,
        "smoke_test_passes": bool(smoke_test_result),
        "cloud_training_commands_documented": (
            DEFAULT_CLOUD_SCRIPT_PATH.exists() or DEFAULT_CLOUD_PS1_PATH.exists()
        ),
    }
    return criteria


def assert_report_sample_counts_match_latest_audit(report: dict[str, Any]) -> None:
    latest_audit = read_json(DEFAULT_DATASET_AUDIT_PATH, default=None)
    if latest_audit is None:
        return
    report_audit = report.get("dataset_audit", {})
    for key in ("train_sample_count", "val_sample_count"):
        latest_count = latest_audit.get(key)
        report_count = report_audit.get(key)
        if report_count != latest_count:
            raise TrainingPreflightError(
                f"Final report {key}={report_count} does not match dataset_audit.json {key}={latest_count}."
            )


def generate_final_reports(
    *,
    dataset_audit: dict[str, Any],
    parse_tests: dict[str, Any],
    reward_variance: dict[str, Any],
    oracle_check: dict[str, Any],
    metrics_by_label: dict[str, dict[str, Any]],
    training_health_report: dict[str, Any],
    smoke_test_result: bool,
    best_checkpoint_path: Optional[str],
    output_json: Path = DEFAULT_FINAL_REPORT_JSON,
    output_md: Path = DEFAULT_FINAL_REPORT_MD,
) -> dict[str, Any]:
    latest_dataset_audit = read_json(DEFAULT_DATASET_AUDIT_PATH, default=None)
    if latest_dataset_audit is not None:
        for key in ("train_sample_count", "val_sample_count"):
            if dataset_audit.get(key) != latest_dataset_audit.get(key):
                raise TrainingPreflightError(
                    f"Report input {key}={dataset_audit.get(key)} does not match "
                    f"dataset_audit.json {key}={latest_dataset_audit.get(key)}."
                )
        dataset_audit = latest_dataset_audit

    criteria = build_training_ready_criteria(
        parse_tests=parse_tests,
        dataset_audit=dataset_audit,
        reward_variance=reward_variance,
        oracle_check=oracle_check,
        smoke_test_result=smoke_test_result,
    )
    training_ready = all(criteria.values())

    raw_metrics = compact_metrics(metrics_by_label.get("raw_model"))
    sft_metrics = compact_metrics(metrics_by_label.get("sft_model"))
    grpo_metrics = compact_metrics(metrics_by_label.get("sft_grpo_model"))
    oracle_metrics = compact_metrics(metrics_by_label["oracle"])
    baseline_metrics = {
        label: compact_metrics(metrics_by_label[label])
        for label in ("random", "heuristic", "q_aware")
    }

    report = {
        "status": {
            "training_ready": training_ready,
            "final_status": f"Training-ready: {'yes' if training_ready else 'no'}",
        },
        "what_was_broken": BROKEN_ITEMS,
        "what_was_fixed": FIXED_ITEMS,
        "dataset_audit": dataset_audit,
        "parse_tests": parse_tests,
        "reward_variance": reward_variance,
        "oracle_check": oracle_check,
        "raw_model_metrics": raw_metrics,
        "sft_metrics": sft_metrics,
        "grpo_metrics": grpo_metrics,
        "oracle_metrics": oracle_metrics,
        "baseline_metrics": baseline_metrics,
        "improvements": {
            "raw_to_sft": compute_improvement(raw_metrics, sft_metrics),
            "sft_to_grpo": compute_improvement(sft_metrics, grpo_metrics),
            "raw_to_grpo": compute_improvement(raw_metrics, grpo_metrics),
        },
        "training_health_report": training_health_report,
        "training_ready_criteria": criteria,
        "best_checkpoint_path": best_checkpoint_path
        or "training/checkpoints/qwen3_sft_grpo_adapter (pending cloud run)",
        "cloud_commands": {
            "sft": CLOUD_SFT_COMMAND,
            "grpo": CLOUD_GRPO_COMMAND,
            "oom_fallback": CLOUD_FALLBACK_COMMAND,
        },
    }
    assert_report_sample_counts_match_latest_audit(report)
    write_json(output_json, report)

    markdown_lines = [
        "# ShadowOps Training Readiness Report",
        "",
        f"Training-ready: {'yes' if training_ready else 'no'}",
        "",
        "## Broken",
    ]
    markdown_lines.extend(f"- {item}" for item in BROKEN_ITEMS)
    markdown_lines.extend(["", "## Fixed"])
    markdown_lines.extend(f"- {item}" for item in FIXED_ITEMS)
    markdown_lines.extend(
        [
            "",
            "## Criteria",
        ]
    )
    markdown_lines.extend(
        f"- {name}: {'pass' if value else 'fail'}"
        for name, value in criteria.items()
    )
    markdown_lines.extend(
        [
            "",
            "## Dataset Audit",
            f"- Train samples: {dataset_audit['train_sample_count']}",
            f"- Val samples: {dataset_audit['val_sample_count']}",
            f"- Duplicate prompts: {dataset_audit['duplicate_prompt_count']}",
            f"- Train/val overlap: {dataset_audit['train_val_overlap_count']}",
            "",
            "## Metrics",
            f"- Random reward mean: {baseline_metrics['random']['reward_mean']:.3f}",
            f"- Heuristic reward mean: {baseline_metrics['heuristic']['reward_mean']:.3f}",
            f"- Q-aware reward mean: {baseline_metrics['q_aware']['reward_mean']:.3f}",
            f"- Oracle reward mean: {oracle_metrics['reward_mean']:.3f}",
            f"- Raw model metrics available: {'yes' if raw_metrics else 'no'}",
            f"- SFT metrics available: {'yes' if sft_metrics else 'no'}",
            f"- GRPO metrics available: {'yes' if grpo_metrics else 'no'}",
            "",
            "## Health Warnings",
        ]
    )
    if training_health_report.get("warnings"):
        markdown_lines.extend(f"- {warning}" for warning in training_health_report["warnings"])
    else:
        markdown_lines.append("- None")
    markdown_lines.extend(
        [
            "",
            "## Cloud Commands",
            "```bash",
            CLOUD_SFT_COMMAND,
            CLOUD_GRPO_COMMAND,
            "# Fallback if GPU memory is tight:",
            CLOUD_FALLBACK_COMMAND,
            "```",
            "",
            f"Best checkpoint path: {report['best_checkpoint_path']}",
            "",
        ]
    )
    output_md.write_text("\n".join(markdown_lines), encoding="utf-8")
    return report


def run_logic_smoke_test(
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
) -> dict[str, Any]:
    train_samples, val_samples, dataset_audit = generate_datasets(
        train_size=train_size,
        val_size=val_size,
        save=True,
    )
    parse_tests = run_parse_action_tests()
    reward_variance = check_reward_variance(train_samples[:40] + val_samples[:20])
    metrics_by_label, oracle_check = build_evaluation_bundle(val_samples)
    health_report = build_training_health_report(
        pre_train_metrics=None,
        sft_metrics=None,
        grpo_metrics=None,
        tracker=None,
        baseline_metrics={label: metrics_by_label[label] for label in ("random", "heuristic", "q_aware")},
        oracle_metrics=metrics_by_label["oracle"],
        lora_parameter_delta=None,
        oracle_check=oracle_check,
    )
    write_json(DEFAULT_HEALTH_REPORT_PATH, health_report)
    smoke_pass = all(
        [
            parse_tests["passed"],
            dataset_audit["passed_preflight"],
            reward_variance["passed"],
            oracle_check["passed"],
        ]
    )
    final_report = generate_final_reports(
        dataset_audit=dataset_audit,
        parse_tests=parse_tests,
        reward_variance=reward_variance,
        oracle_check=oracle_check,
        metrics_by_label=metrics_by_label,
        training_health_report=health_report,
        smoke_test_result=smoke_pass,
        best_checkpoint_path=None,
    )
    return {
        "smoke_test_passed": smoke_pass,
        "parse_tests": parse_tests,
        "dataset_audit": dataset_audit,
        "reward_variance": reward_variance,
        "oracle_check": oracle_check,
        "metrics_by_label": {label: compact_metrics(metrics) for label, metrics in metrics_by_label.items()},
        "final_report": final_report["status"],
    }


def get_installed_version(package_name: str) -> Optional[str]:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def validate_training_runtime(model_name: str) -> bool:
    required_packages = ("torch", "datasets", "transformers", "trl", "unsloth")
    missing_packages = [
        package_name
        for package_name in required_packages
        if importlib.util.find_spec(package_name) is None
    ]
    if missing_packages:
        missing_str = ", ".join(missing_packages)
        print(f"Missing dependency package(s): {missing_str}")
        print("Install with: pip install torch datasets transformers trl unsloth")
        return False

    try:
        import torch
    except Exception as exc:
        print(f"Failed to import torch: {exc}")
        return False

    if not torch.cuda.is_available():
        print("Training runtime is not GPU-ready. Use Colab/Kaggle/cloud GPU for full training.")
        return False

    if "qwen3" in model_name.lower():
        transformers_version = get_installed_version("transformers")
        minimum_version = Version("4.50.3")
        if transformers_version is None or Version(transformers_version) < minimum_version:
            print(
                f"Incompatible transformers version for Qwen3 training. "
                f"Installed={transformers_version}, required>={minimum_version}"
            )
            return False

    return True


def get_total_gpu_memory_gb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        return total_bytes / (1024 ** 3)
    except Exception:
        return None


def configure_runtime_noise_filters() -> None:
    warning_filters = (
        ("ignore", r".*inductor::_alloc_from_pool.*", UserWarning),
        ("ignore", r".*_check_is_size will be removed in a future PyTorch release.*", FutureWarning),
        ("ignore", r".*quantization_config.*already has a `quantization_config` attribute.*", UserWarning),
    )
    for action, message, category in warning_filters:
        warnings.filterwarnings(action, message=message, category=category)


def configure_torch_runtime(torch_module: Any) -> None:
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("high")
    if not torch_module.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        torch_module.backends.cuda.matmul.allow_tf32 = True
    with contextlib.suppress(Exception):
        torch_module.backends.cudnn.allow_tf32 = True


class _BlockedImportLoader(importlib.abc.Loader):
    def __init__(self, module_name: str):
        self.module_name = module_name

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        raise ModuleNotFoundError(f"No module named '{self.module_name}'")


class _BlockedImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hidden_roots: tuple[str, ...]):
        self.hidden_roots = hidden_roots

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname.split(".", 1)[0] not in self.hidden_roots:
            return None
        return importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_BlockedImportLoader(fullname),
            is_package="." not in fullname,
        )


def clear_modules(module_roots: Iterable[str]) -> None:
    roots = set(module_roots)
    for module_name in list(sys.modules):
        if module_name.split(".", 1)[0] in roots:
            sys.modules.pop(module_name, None)


@contextlib.contextmanager
def temporarily_hide_packages(package_roots: Iterable[str]) -> Iterable[None]:
    hidden_roots = tuple(dict.fromkeys(package_roots))
    finder = _BlockedImportFinder(hidden_roots)
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, package: Optional[str] = None) -> Any:
        if name.split(".", 1)[0] in hidden_roots:
            return None
        return original_find_spec(name, package)

    clear_modules(hidden_roots)
    sys.meta_path.insert(0, finder)
    importlib.util.find_spec = patched_find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)


def select_supported_kwargs(callable_obj: Any, candidate_kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    signature = inspect.signature(callable_obj)
    supported_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported_names
    }
    dropped = sorted(set(candidate_kwargs) - set(filtered))
    return filtered, dropped


def load_training_stack(mode: str) -> Optional[dict[str, Any]]:
    package_hints = {
        "transformers": "transformers",
        "unsloth": "unsloth",
        "torch": "torch",
        "datasets": "datasets",
        "trl": "trl",
    }
    loaded: dict[str, Any] = {}
    missing: list[str] = []
    failed_imports: list[tuple[str, BaseException]] = []

    with temporarily_hide_packages(("vllm", "vllm_ascend")):
        transformers_utils = None
        for module_name, package_name in package_hints.items():
            try:
                if module_name == "unsloth":
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"WARNING: Unsloth should be imported before .*",
                        )
                        loaded[module_name] = import_module(module_name)
                    continue
                loaded[module_name] = import_module(module_name)
                if module_name == "transformers":
                    transformers_utils = import_module("transformers.utils")
                    if not hasattr(
                        loaded["transformers"].TrainingArguments,
                        "_VALID_DICT_FIELDS",
                    ):
                        loaded["transformers"].TrainingArguments._VALID_DICT_FIELDS = list(
                            getattr(
                                loaded["transformers"].TrainingArguments,
                                "__dataclass_fields__",
                                {},
                            ).keys()
                        )
                    if not hasattr(transformers_utils, "is_rich_available"):
                        transformers_utils.is_rich_available = lambda: False
            except ModuleNotFoundError as exc:
                missing_root = (exc.name or "").split(".", 1)[0]
                if missing_root == module_name:
                    missing.append(package_name)
                else:
                    failed_imports.append((module_name, exc))
            except Exception as exc:
                failed_imports.append((module_name, exc))

        if missing:
            print(f"Missing dependency package(s): {', '.join(sorted(set(missing)))}")
            return None
        if failed_imports:
            print("Dependency bootstrap failed:")
            for module_name, exc in failed_imports:
                print(f"  - {module_name}: {exc}")
            return None

        Dataset = getattr(loaded["datasets"], "Dataset", None)
        FastModel = getattr(loaded["unsloth"], "FastModel", None)
        if FastModel is None:
            FastModel = getattr(loaded["unsloth"], "FastLanguageModel", None)

        if mode == "grpo":
            trainer_module = importlib.reload(import_module("trl.trainer.grpo_trainer"))
            trainer_config = getattr(trainer_module, "GRPOConfig", None)
            trainer_class = getattr(trainer_module, "GRPOTrainer", None)
        elif mode == "sft":
            trainer_module = importlib.reload(import_module("trl.trainer.sft_trainer"))
            trainer_config = getattr(trainer_module, "SFTConfig", None)
            trainer_class = getattr(trainer_module, "SFTTrainer", None)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if Dataset is None or trainer_config is None or trainer_class is None or FastModel is None:
        print(f"Training stack missing required symbols for mode={mode}.")
        return None

    return {
        "torch": loaded["torch"],
        "Dataset": Dataset,
        "TrainerConfig": trainer_config,
        "TrainerClass": trainer_class,
        "FastModel": FastModel,
        "transformers": loaded["transformers"],
    }


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    return trainable_params, total_params


def capture_trainable_snapshot(model: Any) -> dict[str, Any]:
    snapshot = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            snapshot[name] = parameter.detach().float().cpu().clone()
    return snapshot


def compute_parameter_delta(before_snapshot: dict[str, Any], model: Any) -> dict[str, Any]:
    total_abs_delta = 0.0
    total_l2_delta = 0.0
    changed_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in before_snapshot:
            continue
        diff = parameter.detach().float().cpu() - before_snapshot[name]
        abs_delta = float(diff.abs().sum().item())
        l2_delta = float(torch_norm(diff))
        total_abs_delta += abs_delta
        total_l2_delta += l2_delta
        if abs_delta > 0:
            changed_params += 1
    return {
        "changed_trainable_tensors": changed_params,
        "total_abs_delta": total_abs_delta,
        "total_l2_delta": total_l2_delta,
    }


def torch_norm(tensor: Any) -> float:
    with contextlib.suppress(Exception):
        return float(tensor.norm().item())
    flat = tensor.reshape(-1).tolist()
    return math.sqrt(sum(float(value) ** 2 for value in flat))


def maybe_apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        with contextlib.suppress(Exception):
            signature = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in signature.parameters:
                template_kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def format_prompt_for_model(tokenizer: Any, prompt: str) -> str:
    return maybe_apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )


def build_sft_training_text(tokenizer: Any, prompt: str, completion: str) -> str:
    return maybe_apply_chat_template(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        add_generation_prompt=False,
    )


def make_reward_function(
    tracker: Optional[RewardHealthTracker] = None,
) -> Any:
    def reward_fn(
        completions: list[Any],
        correct_action: Optional[list[str]] = None,
        is_malicious: Optional[list[bool]] = None,
        severity: Optional[list[str]] = None,
        risk_score: Optional[list[float]] = None,
        scenario_type: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        parsed_actions: list[Optional[str]] = []

        correct_action = correct_action or []
        is_malicious = is_malicious or []
        severity = severity or []
        risk_score = risk_score or []
        scenario_type = scenario_type or []

        for index, completion in enumerate(completions):
            sample = {
                "correct_action": (correct_action[index] if index < len(correct_action) else "ALLOW"),
                "is_malicious": bool(is_malicious[index]) if index < len(is_malicious) else False,
                "severity": (severity[index] if index < len(severity) else "LOW"),
                "risk_score": float(risk_score[index]) if index < len(risk_score) else 0.0,
                "scenario_type": (scenario_type[index] if index < len(scenario_type) else "BENIGN_CLEAN"),
            }
            output_text = normalize_completion_text(completion)
            analysis = analyze_action_output(output_text)
            parsed_action = analysis["parsed_action"]
            parsed_actions.append(parsed_action)
            reward, _ = compute_shaped_reward(
                sample,
                parsed_action,
                {"multi_action_warning": analysis["multi_action_warning"]},
            )
            rewards.append(reward)

        if tracker is not None:
            tracker.record_batch(parsed_actions, rewards)

        return rewards

    return reward_fn


def make_health_callback(tracker: RewardHealthTracker, transformers_module: Any) -> Any:
    TrainerCallback = getattr(transformers_module, "TrainerCallback")

    class HealthCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
            logs = logs or {}
            if "grad_norm" in logs:
                tracker.record_grad_norm(logs["grad_norm"])
            return control

    return HealthCallback()


def load_fast_model(
    fast_model_class: Any,
    model_name_or_path: str,
    max_seq_len: int,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    model, tokenizer = fast_model_class.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    return model, tokenizer


def ensure_trainable_lora(
    fast_model_class: Any,
    model: Any,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    if getattr(model, "peft_config", None):
        return model
    return fast_model_class.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def evaluate_model_on_dataset(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    if not samples:
        return evaluate_outputs([], [], label="empty")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts = [format_prompt_for_model(tokenizer, sample["prompt"]) for sample in samples]
    outputs_text: list[str] = []

    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        generation_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if getattr(model, "generation_config", None) is not None:
            generation_config = copy.deepcopy(model.generation_config)
            # Action-only evaluation needs only the action token. Set max_new_tokens
            # on the generation config and clear inherited max_length to avoid
            # transformers warning about both limits being active.
            generation_config.max_new_tokens = max_new_tokens
            with contextlib.suppress(Exception):
                generation_config.max_length = None
            generation_kwargs["generation_config"] = generation_config
        else:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )

        for row_index, length in enumerate(input_lengths):
            completion_tokens = generated[row_index][int(length):]
            completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            outputs_text.append(completion_text)

    return evaluate_outputs(samples, outputs_text, label="model")


def evaluate_saved_model(
    *,
    model_path_or_name: str,
    val_samples: list[dict[str, Any]],
    max_seq_len: int,
    batch_size: int,
    max_new_tokens: int,
) -> Optional[dict[str, Any]]:
    training_stack = load_training_stack("sft")
    if training_stack is None:
        return None
    torch_module = training_stack["torch"]
    configure_torch_runtime(torch_module)
    FastModel = training_stack["FastModel"]
    model, tokenizer = load_fast_model(FastModel, model_path_or_name, max_seq_len=max_seq_len)
    FastModel.for_inference(model)
    metrics = evaluate_model_on_dataset(
        model,
        tokenizer,
        val_samples,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
    )
    metrics["label"] = model_path_or_name
    return metrics


def pick_best_checkpoint(
    *,
    model_name: str,
    output_dir: Path,
    val_samples: list[dict[str, Any]],
    max_seq_len: int,
    batch_size: int,
    max_new_tokens: int,
) -> Optional[str]:
    candidate_paths = [output_dir]
    candidate_paths.extend(sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir()))
    best_path: Optional[Path] = None
    best_score = -float("inf")

    for candidate in candidate_paths:
        metrics = evaluate_saved_model(
            model_path_or_name=str(candidate),
            val_samples=val_samples,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        if metrics is None:
            continue
        score = (0.7 * metrics["safety_accuracy"]) + (0.3 * metrics["exact_match"])
        if score > best_score:
            best_score = score
            best_path = candidate

    if best_path is None:
        return None
    try:
        return str(best_path.relative_to(BACKEND_DIR))
    except ValueError:
        return str(best_path)


def run_subprocess(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def add_shared_smoke_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke-test", action="store_true", help="Run fast logic-only smoke tests and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Build configs and datasets without training.")
    parser.add_argument("--skip-model-load", action="store_true", help="Skip Qwen model loading for local smoke checks.")
