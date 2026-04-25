"""Load ShadowOps submission values only from generated report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(BACKEND_DIR))
    except ValueError:
        return str(path)


def _load(path: Path) -> tuple[Any | None, str]:
    if not path.exists():
        return None, _relative(path)
    return json.loads(path.read_text(encoding="utf-8")), _relative(path)


def _metric(value: Any, source_file: str, generated_at: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"value": "pending", "source_file": source_file, "generated_at": generated_at}
    return {"value": value, "source_file": source_file, "generated_at": generated_at}


def load_benchmark_metrics() -> dict[str, Any]:
    data, source = _load(TRAINING_DIR / "demo_benchmark_report.json")
    if not data:
        return {"status": "pending", "source_file": source}
    generated_at = data.get("generated_at")
    rows = {}
    for row in data.get("metrics", []):
        policy = str(row.get("policy", "")).lower().replace("-", "_")
        rows[policy] = {
            key: _metric(row.get(key), source, generated_at)
            for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")
        }
    return rows


def load_hidden_eval_metrics() -> dict[str, Any]:
    data, source = _load(REPORTS_DIR / "hidden_eval_report.json")
    if not data:
        return {"status": "pending", "source_file": source}
    hidden = ((data.get("datasets") or {}).get("hidden_eval") or {}).get("policies", {})
    rows = {}
    for policy, row in hidden.items():
        metrics = row.get("metrics", {})
        rows[policy] = {
            key: _metric(metrics.get(key), source, data.get("generated_at"))
            for key in ("exact_match", "safety_accuracy", "unsafe_decision_rate", "false_positive_rate", "reward_mean")
        }
    return rows


def load_multistep_metrics() -> dict[str, Any]:
    data, source = _load(REPORTS_DIR / "multistep_episode_report.json")
    if not data:
        return {"status": "pending", "source_file": source}
    metrics = data.get("metrics", data)
    return {
        key: _metric(metrics.get(key), source, data.get("generated_at"))
        for key in (
            "chain_detection_accuracy",
            "safe_chain_allow_rate",
            "malicious_chain_block_or_quarantine_rate",
            "unnecessary_escalation_rate",
            "memory_risk_improvement",
        )
    }


def load_reward_diagnostics() -> dict[str, Any]:
    data, source = _load(REPORTS_DIR / "reward_diagnostics_report.json")
    if not data:
        data, source = _load(REPORTS_DIR / "reward_diagnostics.json")
    if not data:
        return {"status": "pending", "source_file": source}
    return {
        key: _metric(data.get(key), source, data.get("generated_at"))
        for key in (
            "reward_mean",
            "reward_std",
            "zero_std_group_rate",
            "invalid_output_rate",
            "action_distribution_entropy",
        )
    }


def load_checkpoint_comparison() -> dict[str, Any]:
    data, source = _load(REPORTS_DIR / "checkpoint_comparison_report.json")
    if not data:
        return {"status": "pending", "source_file": source}
    champion = data.get("champion_selection", {})
    return {
        "champion": _metric(champion.get("champion"), source, data.get("generated_at")),
        "deployment_recommendation": _metric(champion.get("deployment_recommendation"), source, data.get("generated_at")),
        "checkpoint_count": _metric(len(data.get("checkpoints", [])), source, data.get("generated_at")),
    }


def load_model_policy_gate() -> dict[str, Any]:
    data, source = _load(REPORTS_DIR / "model_policy_gate_report.json")
    if not data:
        return {"status": "pending", "source_file": source}
    return {
        "gate_status": _metric(data.get("gate_status"), source, data.get("generated_at")),
        "deployment_recommendation": _metric(data.get("deployment_recommendation"), source, data.get("generated_at")),
    }


def build_submission_summary() -> dict[str, Any]:
    return {
        "benchmark_metrics": load_benchmark_metrics(),
        "hidden_eval_metrics": load_hidden_eval_metrics(),
        "multistep_metrics": load_multistep_metrics(),
        "reward_diagnostics": load_reward_diagnostics(),
        "checkpoint_comparison": load_checkpoint_comparison(),
        "model_policy_gate": load_model_policy_gate(),
        "note": "Missing values are reported as pending; no metric is invented.",
    }


def main() -> int:
    summary = build_submission_summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
