# ShadowOps Two-Minute Pitch Script

## 0:00-0:20 Problem
Autonomous agents are entering security operations, but a single bad decision can expose customer data, create an IAM admin path, or open production infrastructure to the internet. The hard part is not only detecting risk. The agent must know when to allow, block, quarantine, or ask a human.

## 0:20-0:50 Environment
ShadowOps turns that problem into an OpenEnv-compatible RL environment. Each episode is a multi-step cyber incident trajectory across cloud, CI/CD, IAM, firewall, and pentest workflows. The observation includes payload risk, domain context, missing evidence, memory, and production impact.

## 0:50-1:20 Agent Behavior
The agent chooses from four actions: allow, block, quarantine, or fork to human review. Its decisions affect future state: risk accumulates, evidence plans change, quarantine holds continue, and attack chains become visible over time.

## 1:20-1:50 Reward And Training
The reward is decomposed so judges can inspect it: action correctness, safety penalty, false-positive penalty, evidence completeness, risk-chain handling, and uncertainty handling. The Q-aware policy is the verified pre-training guardrail. SFT and GRPO training are ready, but model improvement is only claimed after checkpoint evaluation passes the safety gate.

## 1:50-2:00 Impact
ShadowOps gives the hackathon a real RL environment for cyber defense: safe by default, measurable, auditable, and ready for trained-policy comparison after GPU runs.

## Optional Judge Follow-up: Why This Is Hard To Game
The reward punishes lazy strategies. Always block fails approved work. Always fork fails complete-evidence cases. Always allow fails safety. The hidden evaluation set includes false-positive traps and adversarial chains so the policy must reason over evidence and memory.

## Optional Judge Follow-up: Hidden Evaluation And Memory
ShadowOps has a hidden-style benchmark and a multi-step episode benchmark. The memory benchmark shows that a firewall opening, IAM admin creation, and data export become more dangerous together than as isolated events.

## Optional Judge Follow-up: Evidence Honesty
ShadowOps currently reports policy-baseline performance from reproducible benchmark artifacts. Trained checkpoint metrics are reported only after `model_eval_report.json` or `checkpoint_comparison_report.json` is generated from a real checkpoint. Q-aware supervisor is the safety guardrail; a trained checkpoint must pass the model-vs-policy gate before it is considered a deployment candidate.

ShadowOps prevents common RL action-format failures by requiring exact action outputs: `ALLOW`, `BLOCK`, `FORK`, or `QUARANTINE`.
