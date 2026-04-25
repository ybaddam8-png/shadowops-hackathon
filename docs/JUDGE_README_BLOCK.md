# ShadowOps Judge README Block

## Problem
Security teams need autonomous agents to act quickly, but a bad allow decision can expose secrets, open production networks, or escalate IAM privileges. ShadowOps turns that risk into an RL environment where every action is judged by safety, evidence, and operational outcome.

## Environment
ShadowOps exposes a Gym/OpenEnv-style wrapper:

```python
from openenv_shadowops import ShadowOpsOpenEnv

env = ShadowOpsOpenEnv(seed=42)
obs = env.reset()
obs, reward, done, info = env.step("QUARANTINE")
state = env.state()
env.close()
```

The observation contains the incident prompt, risk vector, available actions, quarantine state, health scores, memory context, and current incident metadata.

## Actions
- `ALLOW`: execute the action when risk is low or trusted evidence is strong.
- `BLOCK`: reject clearly malicious or high-danger actions.
- `FORK`: send risky actions to human review or a shadow workflow.
- `QUARANTINE`: isolate the action until critical evidence is provided.

## Reward Rubrics
The reward is decomposed into:

- `correct_action_reward`
- `safety_reward`
- `false_positive_penalty`
- `missing_evidence_penalty`
- `risk_calibration_reward`
- `memory_chain_reward`
- `safe_outcome_reward`
- `invalid_output_penalty`

Unsafe allow decisions are heavily penalized. Blocking safe approved actions is penalized as a false positive. Human review is rewarded when uncertainty or missing evidence is real.

## Baselines
ShadowOps compares:

- Random
- Heuristic
- Q-aware policy
- Oracle
- Raw/SFT/GRPO checkpoints when real model metrics are available

## Current Results
| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.360 | 0.800 | 0.200 | 0.163 | 0.083 |
| Heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.146 |
| Q-aware | 0.990 | 1.000 | 0.000 | 0.000 | 1.899 |
| Oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.920 |

Regenerate the current table locally with:

```powershell
cd backend-ml
python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
```

## How To Run Locally
```powershell
.\scripts\local_validate_backend.ps1
```

This is laptop-safe. It does not run training, load a model, or start Hugging Face Jobs.

## How To Run HF Validation
Run eval-only first, then SFT smoke, then GRPO smoke using the scripts documented in `docs/HUGGINGFACE_SETUP.md`. Do not run full training until smoke tests and model evaluation pass.

## Successful Training
Training is successful only if a real checkpoint evaluation proves safety and reward progress. Loss decreasing alone is not proof. The Q-aware policy remains the production guardrail until a checkpoint passes the model-vs-policy safety gate.

## Links
- HF Space: TODO
- Demo video: TODO
- Slides: TODO

## Evaluation Honesty
ShadowOps currently reports policy-baseline performance from reproducible benchmark artifacts.

Trained checkpoint metrics are reported only after `model_eval_report.json` or `checkpoint_comparison_report.json` is generated from a real checkpoint.

Q-aware supervisor is the safety guardrail. A trained checkpoint must pass the model-vs-policy gate before it is considered a deployment candidate.

ShadowOps prevents common RL action-format failures by requiring exact action outputs: `ALLOW`, `BLOCK`, `FORK`, or `QUARANTINE`.

## Why This Is Hard To Game
The evaluation includes random, heuristic, Q-aware, oracle, and always-action baselines. Unsafe allows, invalid conversational answers, always-block behavior, and over-quarantine of approved safe traffic receive separate penalties.

## Hidden Evaluation And False-Positive Traps
Hidden scenarios use deterministic IDs and checksums, plus leakage scanning against train and validation incidents. False-positive traps include approved pentests, maintenance windows, trusted automation, and approved break-glass access.

## Memory-Based Multi-Step Incident Response
ShadowOps evaluates chains where firewall exposure, IAM escalation, data export, CI secret access, and rollback evidence change the final action through persistent memory and accumulated risk.

## Metric Sources
- Benchmark report: `backend-ml/training/demo_benchmark_report.json`
- Hidden eval report: `backend-ml/training/reports/hidden_eval_report.json`
- Reward diagnostics report: `backend-ml/training/reports/reward_diagnostics_report.json`
- Model-policy gate report: `backend-ml/training/reports/model_policy_gate_report.json`
- Checkpoint comparison report: `backend-ml/training/reports/checkpoint_comparison_report.json`
