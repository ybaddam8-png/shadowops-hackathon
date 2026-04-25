"""Laptop-safe dataset audit for ShadowOps training/evaluation data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.shadowops_training_common import (  # noqa: E402
    DEFAULT_DATASET_AUDIT_PATH,
    DEFAULT_HARD_NEGATIVE_PATH,
    DEFAULT_TRAIN_DATASET_PATH,
    DEFAULT_VAL_DATASET_PATH,
    VALID_ACTION_SET,
    write_json,
)


DEFAULT_DATASET_AUDIT_MD = DEFAULT_DATASET_AUDIT_PATH.with_suffix(".md")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _label(sample: dict[str, Any]) -> str | None:
    value = sample.get("correct_action") or sample.get("expected_decision")
    return str(value).upper() if value else None


def _prompt(sample: dict[str, Any]) -> str:
    return str(sample.get("prompt") or sample.get("raw_payload") or sample.get("action_summary") or "")


def _distribution(samples: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(sample.get(key) or "unknown") for sample in samples))


def _risk_type_distribution(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(str(sample.get("severity") or sample.get("risk_level") or "unknown") for sample in samples)
    )


def _action_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(_label(sample) or "MISSING" for sample in samples))


def _action_pct(counts: dict[str, int], total: int) -> dict[str, float]:
    return {
        action: counts.get(action, 0) / max(total, 1)
        for action in sorted(VALID_ACTION_SET)
    }


def _false_positive_challenge(sample: dict[str, Any]) -> bool:
    if sample.get("is_malicious", False):
        return False
    if sample.get("provided_evidence"):
        return True
    if sample.get("scenario_type") == "BENIGN_NOISY":
        return True
    return str(sample.get("severity", "")).upper() in {"MEDIUM", "HIGH"}


def build_dataset_audit(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    hard_negative_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hard_negative_samples = hard_negative_samples or []
    all_train_val = train_samples + val_samples
    all_samples = all_train_val + hard_negative_samples

    train_prompts = [_prompt(sample) for sample in train_samples]
    val_prompts = [_prompt(sample) for sample in val_samples]
    combined_prompts = train_prompts + val_prompts
    prompt_counts = Counter(prompt for prompt in combined_prompts if prompt)

    missing_label_count = sum(1 for sample in all_samples if _label(sample) is None)
    invalid_action_label_count = sum(
        1 for sample in all_samples if _label(sample) is not None and _label(sample) not in VALID_ACTION_SET
    )
    train_action_counts = _action_counts(train_samples)
    val_action_counts = _action_counts(val_samples)
    hard_negative_action_counts = _action_counts(hard_negative_samples)
    combined_action_counts = Counter(train_action_counts)
    combined_action_counts.update(val_action_counts)
    action_distribution = {
        "train": train_action_counts,
        "val": val_action_counts,
        "hard_negative": hard_negative_action_counts,
        "train_pct": _action_pct(train_action_counts, len(train_samples)),
        "val_pct": _action_pct(val_action_counts, len(val_samples)),
        "combined_pct": _action_pct(dict(combined_action_counts), len(train_samples) + len(val_samples)),
    }
    failures: list[str] = []
    if any(count > 1 for count in prompt_counts.values()):
        failures.append("duplicate prompts detected")
    train_val_overlap = len(set(train_prompts) & set(val_prompts))
    if train_val_overlap:
        failures.append("train/validation prompt overlap detected")
    if missing_label_count:
        failures.append("missing labels detected")
    if invalid_action_label_count:
        failures.append("invalid action labels detected")

    return {
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "hard_negative_count": len(hard_negative_samples),
        "duplicate_prompt_count": sum(count - 1 for count in prompt_counts.values() if count > 1),
        "train_val_overlap_count": train_val_overlap,
        "action_distribution": action_distribution,
        "domain_distribution": _distribution(all_samples, "domain"),
        "risk_type_distribution": _risk_type_distribution(all_samples),
        "false_positive_challenge_count": sum(1 for sample in all_samples if _false_positive_challenge(sample)),
        "missing_label_count": missing_label_count,
        "invalid_action_label_count": invalid_action_label_count,
        "passed_preflight": not failures,
        "failures": failures,
    }


def write_dataset_audit_markdown(audit: dict[str, Any], output_path: Path = DEFAULT_DATASET_AUDIT_MD) -> None:
    lines = [
        "# ShadowOps Dataset Audit",
        "",
        f"- Train samples: {audit['train_sample_count']}",
        f"- Validation samples: {audit['val_sample_count']}",
        f"- Hard-negative samples: {audit['hard_negative_count']}",
        f"- Duplicate prompts: {audit['duplicate_prompt_count']}",
        f"- Train/validation overlap: {audit['train_val_overlap_count']}",
        f"- Missing labels: {audit['missing_label_count']}",
        f"- Invalid action labels: {audit['invalid_action_label_count']}",
        f"- False-positive challenges: {audit['false_positive_challenge_count']}",
        f"- Preflight: {'PASS' if audit.get('passed_preflight') else 'FAIL'}",
        "",
        "## Action Distribution",
        "",
    ]
    for split in ("train", "val", "hard_negative"):
        counts = audit["action_distribution"].get(split, {})
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| Action | Count |")
        lines.append("| --- | ---: |")
        for action, count in sorted(counts.items()):
            lines.append(f"| {action} | {count} |")
        lines.append("")
    lines.extend(["## Domain Distribution", "", "| Domain | Count |", "| --- | ---: |"])
    for domain, count in sorted(audit["domain_distribution"].items()):
        lines.append(f"| {domain} | {count} |")
    if audit.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in audit["failures"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_dataset_audit(
    *,
    output_json: Path = DEFAULT_DATASET_AUDIT_PATH,
    output_md: Path = DEFAULT_DATASET_AUDIT_MD,
) -> dict[str, Any]:
    train_samples = _read_json(DEFAULT_TRAIN_DATASET_PATH, [])
    val_samples = _read_json(DEFAULT_VAL_DATASET_PATH, [])
    hard_negative_samples = _read_json(DEFAULT_HARD_NEGATIVE_PATH, [])
    audit = build_dataset_audit(train_samples, val_samples, hard_negative_samples)
    write_json(output_json, audit)
    write_dataset_audit_markdown(audit, output_md)
    return audit


def main() -> int:
    audit = run_dataset_audit()
    print("Dataset audit:", "PASS" if audit.get("passed_preflight") else "FAIL")
    print(f"Saved: {DEFAULT_DATASET_AUDIT_PATH.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_DATASET_AUDIT_MD.relative_to(BACKEND_DIR)}")
    return 0 if audit.get("passed_preflight") else 1


if __name__ == "__main__":
    raise SystemExit(main())
