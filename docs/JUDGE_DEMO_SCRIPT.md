# ShadowOps 3-Minute Judge Demo Script

## 0:00 - Opening Problem

Modern incident response agents are powerful, but a single unsafe allow can expose secrets, create admin access, or open production infrastructure. ShadowOps is a safety supervisor that decides whether an action should be allowed, blocked, quarantined, or forked to a human reviewer.

## 0:30 - Live Scenario 1: Malicious IAM Escalation

Run or show the rogue IAM admin replay. Point out:

- The request asks for a new admin identity.
- The actor and approval chain are not trusted.
- ShadowOps returns `QUARANTINE` instead of allowing production impact.
- The safe outcome prevents credential creation until approval, identity, privilege diff, and MFA status are verified.

## 1:00 - Live Scenario 2: Safe Approved Pentest

Show a pentest request with authorization letter, approved window, scope document, target list, and emergency contact.

- ShadowOps reduces false positives when trusted evidence exists.
- It does not blindly block risky-looking but authorized security work.
- It can allow with restrictions or fork only when evidence is partial.

## 1:30 - Evidence Plan

Open `backend-ml/training/judge_readiness_report.md` and show the evidence planning example:

- Identity first.
- Approval/change ticket second.
- Privilege, secret, or public exposure checks third.
- Rollback and monitoring last.

## 2:00 - Memory And Risk Accumulation

Show the chain:

`firewall open -> IAM admin creation -> data export`

Explain that each action alone may look explainable, but together they form a risky chain. ShadowOps accumulates session risk and changes the supervisor decision accordingly.

## 2:20 - Safe Outcome

Show `structured_safe_outcome`:

- remediation steps
- rollback requirement
- human review requirement
- monitoring requirement
- allowed next actions

This turns a raw model/policy decision into an operational incident response action.

## 2:40 - Benchmark Report

Show:

```powershell
cd backend-ml
python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
```

Current laptop-safe guardrail benchmark:

- Q-aware exact match: 0.990
- Q-aware safety accuracy: 1.000
- Q-aware unsafe decision rate: 0.000
- Q-aware false positive rate: 0.000
- Q-aware reward mean: 1.899

## 3:00 - Model-vs-Policy Safety Gate

End with the honest claim:

> ShadowOps currently proves production-safe behavior through the Q-aware policy guardrail. Trained model improvement is only claimed after checkpoint evaluation passes the model-vs-policy safety gate.

Loss decrease alone is not proof. A trained checkpoint must beat raw/SFT baselines and remain safe compared with the Q-aware guardrail.
