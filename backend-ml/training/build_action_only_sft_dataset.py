"""Build and audit the strict action-only ShadowOps SFT dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
DATASET_PATH = TRAINING_DIR / "action_only_sft_dataset.json"
AUDIT_JSON = REPORTS_DIR / "action_only_sft_dataset_audit.json"
AUDIT_MD = REPORTS_DIR / "action_only_sft_dataset_audit.md"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_action_parser import VALID_ACTIONS, parse_action_output  # noqa: E402
from training.shadowops_training_common import (  # noqa: E402
    DEFAULT_TRAIN_DATASET_PATH,
    DEFAULT_VAL_DATASET_PATH,
    generate_datasets,
    read_json,
    write_json,
)


def _load_source_samples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = read_json(DEFAULT_TRAIN_DATASET_PATH, default=None)
    val = read_json(DEFAULT_VAL_DATASET_PATH, default=None)
    if train is None or val is None:
        train, val, _ = generate_datasets(save=True)
    return list(train), list(val)


def _label(sample: dict[str, Any]) -> str:
    return str(sample.get("correct_action") or sample.get("expected_action") or sample.get("expected_decision") or "")


def _incident_key(sample: dict[str, Any]) -> str:
    text = str(sample.get("observation") or sample.get("raw_payload") or sample.get("action_summary") or sample.get("prompt") or "")
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _record(sample: dict[str, Any], split: str) -> dict[str, Any]:
    prompt = str(sample.get("prompt") or "")
    if not prompt:
        prompt = str(sample.get("observation") or sample.get("raw_payload") or sample.get("action_summary") or "")
    return {
        "sample_id": sample.get("sample_id"),
        "split": split,
        "domain": sample.get("policy_domain") or sample.get("domain", "unknown"),
        "scenario_type": sample.get("scenario_type") or sample.get("difficulty") or "unknown",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _label(sample)},
        ],
        "prompt": prompt,
        "completion": _label(sample),
    }


def build_action_only_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train, val = _load_source_samples()
    rows: list[dict[str, Any]] = []
    invalid_labels: list[dict[str, Any]] = []
    action_distribution: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    domain_distribution: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    scenario_distribution: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    examples_per_action: dict[str, list[dict[str, Any]]] = defaultdict(list)

    split_rows = (("train", train), ("validation", val))
    for split, samples in split_rows:
        for sample in samples:
            label = _label(sample)
            parsed = parse_action_output(label, mode="strict_train")
            if not parsed.valid:
                invalid_labels.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "split": split,
                        "label": label,
                        "reason": parsed.reason,
                    }
                )
            row = _record(sample, split)
            rows.append(row)
            action_distribution[split][label] += 1
            domain_distribution[split][str(row["domain"])] += 1
            scenario_distribution[split][str(row["scenario_type"])] += 1
            if label in VALID_ACTIONS and len(examples_per_action[label]) < 3:
                examples_per_action[label].append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "split": split,
                        "domain": row["domain"],
                        "prompt_preview": row["prompt"][:160],
                    }
                )

    keys_by_split = {
        "train": [_incident_key(sample) for sample in train],
        "validation": [_incident_key(sample) for sample in val],
    }
    duplicate_count = len(rows) - len({row["prompt"].strip().lower() + "\n" + row["completion"] for row in rows})
    leakage_hashes = sorted(set(keys_by_split["train"]) & set(keys_by_split["validation"]))
    leakage_warnings = [f"{len(leakage_hashes)} train/validation incident hash overlap(s) detected."] if leakage_hashes else []
    audit = {
        "total_samples": len(rows),
        "train_samples": len(train),
        "validation_samples": len(val),
        "action_distribution": {split: dict(counter) for split, counter in action_distribution.items()},
        "domain_distribution": {split: dict(counter) for split, counter in domain_distribution.items()},
        "scenario_distribution": {split: dict(counter) for split, counter in scenario_distribution.items()},
        "invalid_label_count": len(invalid_labels),
        "invalid_labels": invalid_labels[:20],
        "duplicate_count": duplicate_count,
        "leakage_warning_count": len(leakage_warnings),
        "leakage_warnings": leakage_warnings,
        "examples_per_action": dict(examples_per_action),
        "passed": len(invalid_labels) == 0 and duplicate_count == 0 and not leakage_warnings,
    }
    return rows, audit


def write_action_only_dataset(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    write_json(DATASET_PATH, rows)
    write_json(AUDIT_JSON, audit)
    lines = [
        "# ShadowOps Action-Only SFT Dataset Audit",
        "",
        f"Total samples: {audit['total_samples']}",
        f"Train samples: {audit['train_samples']}",
        f"Validation samples: {audit['validation_samples']}",
        f"Invalid labels: {audit['invalid_label_count']}",
        f"Duplicate prompt/label rows: {audit['duplicate_count']}",
        f"Leakage warnings: {audit['leakage_warning_count']}",
        f"Passed: {audit['passed']}",
        "",
        "## Action Distribution",
        "",
    ]
    for split, counts in audit["action_distribution"].items():
        lines.append(f"### {split}")
        for action in VALID_ACTIONS:
            lines.append(f"- {action}: {counts.get(action, 0)}")
        lines.append("")
    lines.extend(["## Examples Per Action", ""])
    for action, examples in audit["examples_per_action"].items():
        lines.append(f"### {action}")
        for example in examples:
            lines.append(f"- `{example['sample_id']}` ({example['split']}, {example['domain']}): {example['prompt_preview']}")
        lines.append("")
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows, audit = build_action_only_dataset()
    write_action_only_dataset(rows, audit)
    print(f"Action-only SFT samples: {audit['total_samples']}")
    print(f"Invalid labels: {audit['invalid_label_count']}")
    print(f"Saved: {DATASET_PATH.relative_to(BACKEND_DIR)}")
    print(f"Saved: {AUDIT_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {AUDIT_MD.relative_to(BACKEND_DIR)}")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
