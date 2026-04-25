# ShadowOps Judge Readiness Report

Status: `judge_ready_laptop_safe`

No model improvement is claimed unless model_eval_report proves it.

Guardrail: q_aware_demo_policy remains the production guardrail.

## Baseline Benchmark

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.360 | 0.800 | 0.200 | 0.163 | 0.083 |
| Heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.146 |
| Q-aware | 0.850 | 1.000 | 0.000 | 0.000 | 1.802 |
| Oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.920 |

## Reward Diagnostics

- Samples: 100
- Reward mean/std: -0.401 / 1.958
- Zero-std groups: 0.0%
- Invalid output rate: 0.167

## Dataset Audit

- Train samples: 500
- Validation samples: 100
- Hard negatives: 40
- Duplicate prompts: 0
- Train/validation overlap: 0
- Missing labels: 0
- Invalid action labels: 0

## Model-vs-Policy Gate

- Status: FAIL
- Reason: No model metrics are available; checkpoint was not loaded or evaluation failed.
- Recommended next action: Run --evaluate-model with a valid --model-path after SFT/GRPO smoke training.

## Evidence Planner Example

- Step 1 [critical]: Verify requester identity before reviewing approval evidence.
- Step 2 [high]: Provide approval chain for iam production action.
- Step 3 [high]: Provide privilege diff for iam production action.
- Step 4 [low]: Provide rollback plan and monitoring owner before exposure is allowed.

## Safe Outcome Example

- Outcome: Fork to human reviewer with IAM approval chain.
- Human review required: True
- Remediation steps: 2

## Memory/Risk Chain Example

- Session risk: 1.000
- Detected chains: firewall open -> IAM admin creation -> data export