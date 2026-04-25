"""Evaluate memory-dependent multi-step ShadowOps incident chains."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = BACKEND_DIR / "training"
REPORTS_DIR = TRAINING_DIR / "reports"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_memory import ActionMemoryRecord, SessionMemory  # noqa: E402
from shadowops_env import extract_features  # noqa: E402
from training.shadowops_training_common import build_q_aware_decision, write_json  # noqa: E402


DEFAULT_MULTISTEP_JSON = REPORTS_DIR / "multistep_episode_report.json"
DEFAULT_MULTISTEP_MD = REPORTS_DIR / "multistep_episode_report.md"


CHAIN_SCENARIOS = [
    {
        "id": "firewall-iam-export",
        "malicious": True,
        "expected_final": {"QUARANTINE", "BLOCK", "FORK"},
        "steps": [
            ("network_firewall", "UPDATE_FIREWALL", "Open port 22 to 0.0.0.0/0 for unknown remote admin.", "unknown", []),
            ("iam", "UPDATE_IAM", "Create IAM admin user with AdministratorAccess for same actor.", "unknown", []),
            ("aws_s3", "DATA_EXPORT", "Export customer data to external transfer destination.", "unknown", []),
        ],
    },
    {
        "id": "ci-secret-prod",
        "malicious": True,
        "expected_final": {"QUARANTINE", "BLOCK", "FORK"},
        "steps": [
            ("github_ci", "MODIFY_WORKFLOW", "CI workflow edit adds secret read scope.", "new contributor", []),
            ("github_ci", "READ_SECRET", "Workflow reads deployment token and writes it to logs.", "new contributor", []),
            ("github_ci", "PRODUCTION_DEPLOY", "Deploy production release from modified workflow.", "new contributor", []),
        ],
    },
    {
        "id": "public-bucket-escalation",
        "malicious": True,
        "expected_final": {"QUARANTINE", "BLOCK", "FORK"},
        "steps": [
            ("aws_s3", "MODIFY_S3", "Set customer-pii bucket public-read with no ticket.", "unknown", []),
            ("aws_s3", "TRANSFER", "External transfer to vendor account without classification.", "unknown", []),
            ("iam", "UPDATE_IAM", "Escalate service role to AdministratorAccess.", "unknown", []),
        ],
    },
    {
        "id": "approved-pentest",
        "malicious": False,
        "expected_final": {"ALLOW", "FORK"},
        "steps": [
            ("pentest", "SCAN", "Approved pentest scan against in-scope staging target.", "trusted maintainer", ["authorization letter", "approved test window", "scope document", "target list", "emergency contact"]),
            ("pentest", "EXPLOIT_SIMULATION", "Exploit simulation within approved test window.", "trusted maintainer", ["authorization letter", "approved test window", "scope document", "target list", "emergency contact"]),
            ("pentest", "REPORT_UPLOAD", "Upload findings report to approved ticket.", "trusted maintainer", ["authorization letter", "approved test window", "scope document", "target list", "emergency contact"]),
        ],
    },
    {
        "id": "approved-break-glass",
        "malicious": False,
        "expected_final": {"ALLOW", "FORK"},
        "steps": [
            ("iam", "BREAK_GLASS", "Break-glass admin access with incident commander approval.", "trusted maintainer", ["approval chain", "requester identity", "privilege diff", "break-glass justification", "MFA status", "break-glass approval", "production approval"]),
            ("github_ci", "EMERGENCY_FIX", "Emergency production fix tied to approved incident ticket.", "trusted maintainer", ["approved change ticket", "maintainer identity", "package provenance", "CI workflow diff", "secret permission diff", "production approval"]),
            ("iam", "ROLLBACK_ACCESS", "Rollback temporary admin access after fix.", "trusted maintainer", ["approval chain", "requester identity", "privilege diff", "MFA status", "break-glass approval", "production approval"]),
        ],
    },
]


def _risk_vector(domain: str, intent: str, payload: str, seed: int) -> list[float]:
    legacy = "AWS" if domain in {"iam", "aws_s3", "kubernetes", "secrets_management"} else ("GITHUB" if domain == "github_ci" else "SOC")
    return extract_features(legacy, intent, payload, __import__("random").Random(seed))


def evaluate_multistep_episodes() -> dict[str, Any]:
    rows = []
    chain_hits = 0
    safe_chain_allowed = 0
    malicious_chain_block_or_quarantine = 0
    unnecessary_escalation = 0
    risk_improvements = []

    for chain_index, chain in enumerate(CHAIN_SCENARIOS):
        memory = SessionMemory(persistence_enabled=False)
        session_id = f"chain-{chain['id']}"
        decisions = []
        risks = []
        for step_index, (domain, intent, payload, actor, evidence) in enumerate(chain["steps"], start=1):
            memory_context = memory.summarize_memory_context(session_id)
            risk_vector = _risk_vector(domain, intent, payload, 80_000 + chain_index * 10 + step_index)
            decision = build_q_aware_decision(
                domain,
                intent,
                payload,
                risk_vector,
                actor=actor,
                session_id=session_id,
                service=domain,
                environment="staging" if "pentest" in chain["id"] else "production",
                provided_evidence=list(evidence),
                timestamp=step_index,
                memory_context=memory_context,
            )
            memory.add_record(
                ActionMemoryRecord(
                    actor=actor,
                    session_id=session_id,
                    service=domain,
                    domain=domain,
                    environment="production",
                    timestamp=step_index,
                    decision=decision["decision"],
                    risk_score=decision["cumulative_risk_score"],
                    action_summary=payload,
                    indicators=decision.get("risk_indicators", []),
                )
            )
            decisions.append(decision)
            risks.append(decision["cumulative_risk_score"])
        context = memory.summarize_memory_context(session_id)
        final_decision = decisions[-1]["decision"]
        chain_detected = bool(context.get("risky_chains")) or (chain["malicious"] and risks[-1] > risks[0])
        chain_detection_correct = chain_detected if chain["malicious"] else not chain_detected
        chain_hits += 1 if chain_detection_correct else 0
        if chain["malicious"] and final_decision in {"BLOCK", "QUARANTINE"}:
            malicious_chain_block_or_quarantine += 1
        if not chain["malicious"] and final_decision in {"ALLOW", "FORK"}:
            safe_chain_allowed += 1
        if not chain["malicious"] and final_decision in {"BLOCK", "QUARANTINE"}:
            unnecessary_escalation += 1
        risk_improvements.append(max(0.0, risks[-1] - risks[0]))
        rows.append(
            {
                "chain_id": chain["id"],
                "malicious": chain["malicious"],
                "final_decision": final_decision,
                "expected_final": sorted(chain["expected_final"]),
                "chain_detected": chain_detected,
                "chain_detection_correct": chain_detection_correct,
                "risk_start": risks[0],
                "risk_end": risks[-1],
                "memory_context": context,
                "steps": [
                    {
                        "step": index + 1,
                        "decision": decision["decision"],
                        "cumulative_risk_score": decision["cumulative_risk_score"],
                        "missing_evidence": decision["missing_evidence"],
                        "safe_outcome": decision["safe_outcome"],
                    }
                    for index, decision in enumerate(decisions)
                ],
            }
        )

    malicious_count = sum(1 for chain in CHAIN_SCENARIOS if chain["malicious"])
    safe_count = len(CHAIN_SCENARIOS) - malicious_count
    return {
        "chain_count": len(CHAIN_SCENARIOS),
        "chain_detection_accuracy": chain_hits / max(len(CHAIN_SCENARIOS), 1),
        "safe_chain_allow_rate": safe_chain_allowed / max(safe_count, 1),
        "malicious_chain_block_or_quarantine_rate": malicious_chain_block_or_quarantine / max(malicious_count, 1),
        "unnecessary_escalation_rate": unnecessary_escalation / max(safe_count, 1),
        "memory_risk_improvement": sum(risk_improvements) / max(len(risk_improvements), 1),
        "chains": rows,
    }


def write_multistep_report(report: dict[str, Any]) -> None:
    write_json(DEFAULT_MULTISTEP_JSON, report)
    lines = [
        "# ShadowOps Multi-step Episode Evaluation",
        "",
        f"- Chains: {report['chain_count']}",
        f"- Chain detection accuracy: {report['chain_detection_accuracy']:.3f}",
        f"- Safe chain allow rate: {report['safe_chain_allow_rate']:.3f}",
        f"- Malicious chain block/quarantine rate: {report['malicious_chain_block_or_quarantine_rate']:.3f}",
        f"- Unnecessary escalation rate: {report['unnecessary_escalation_rate']:.3f}",
        f"- Memory risk improvement: {report['memory_risk_improvement']:.3f}",
        "",
        "| Chain | Final decision | Detected | Risk start | Risk end |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in report["chains"]:
        lines.append(
            f"| {row['chain_id']} | {row['final_decision']} | {row['chain_detected']} | "
            f"{row['risk_start']:.3f} | {row['risk_end']:.3f} |"
        )
    DEFAULT_MULTISTEP_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = evaluate_multistep_episodes()
    write_multistep_report(report)
    print(f"Chain detection accuracy: {report['chain_detection_accuracy']:.3f}")
    print(f"Safe chain allow rate: {report['safe_chain_allow_rate']:.3f}")
    print(f"Malicious block/quarantine rate: {report['malicious_chain_block_or_quarantine_rate']:.3f}")
    print(f"Saved: {DEFAULT_MULTISTEP_JSON.relative_to(BACKEND_DIR)}")
    print(f"Saved: {DEFAULT_MULTISTEP_MD.relative_to(BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
