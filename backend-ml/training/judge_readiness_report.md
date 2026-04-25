# ShadowOps Judge Readiness Report

## 1. Executive Summary

ShadowOps is a laptop-safe autonomous incident response supervisor that evaluates high-risk cloud, CI, IAM, network, and pentest actions before they reach production.

## 2. What ShadowOps Does

- Scores action risk from payload signals and domain policy.
- Accumulates session memory to catch multi-step attack chains.
- Requests missing operational evidence before allowing risky actions.
- Returns safe outcomes such as quarantine, fork-to-human, rollback, or restricted allow.

## 3. Why It Is Different

- Uses a deterministic Q-aware guardrail by default, so the demo is safe without a GPU.
- Compares future model checkpoints against policy baselines before trusting them.
- Explains decisions through audit traces, evidence plans, and safe outcomes.
- Optimizes for false-positive reduction when trusted evidence is present.

## 4. Benchmark Results

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.360 | 0.800 | 0.200 | 0.163 | 0.083 |
| Heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.146 |
| Q-aware | 0.850 | 1.000 | 0.000 | 0.000 | 1.802 |
| Oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.920 |

## 5. Dataset Audit

- Train samples: 500
- Validation samples: 100
- Hard negatives: 40
- Duplicate prompts: 0
- Train/validation overlap: 0
- Missing labels: 0
- Invalid action labels: 0
- False-positive challenges: 192

## 6. Reward Diagnostics

- Samples: 100
- Reward mean/std: -0.401 / 1.958
- Zero-std groups: 0.0%
- Invalid output rate: 0.167

## 7. Safety Guardrail Explanation

- Policy: `q_aware_demo_policy`
- The Q-aware policy is the production guardrail until a trained checkpoint beats reference metrics and passes the safety gate.

## 8. Evidence Planning Example

- Step 1 [critical]: Verify requester identity before reviewing approval evidence.
- Step 2 [high]: Provide approval chain for iam production action.
- Step 3 [high]: Provide privilege diff for iam production action.
- Step 4 [low]: Provide rollback plan and monitoring owner before exposure is allowed.

## 9. Memory + Risk Chain Example

- Action chain: open firewall port 22 to 0.0.0.0/0 -> create IAM admin user with AdministratorAccess -> export customer data to external transfer destination
- Session risk: 1.000
- Detected chains: firewall open -> IAM admin creation -> data export

## 10. Model-vs-Policy Gate

- Status: FAIL
- Reason: No model metrics are available; checkpoint was not loaded or evaluation failed.
- Recommended next action: Run --evaluate-model with a valid --model-path after SFT/GRPO smoke training.

## 11. GPU Training Handoff

- Pull shadowops-backend-agent-upgrade.
- Run laptop-safe baseline/eval first.
- Run SFT smoke, then evaluate SFT checkpoint.
- Run GRPO smoke, then evaluate GRPO checkpoint.
- Only run longer GRPO after model_eval_report proves safety and reward progress.

## 12. Honest Limitations

- No model improvement is claimed without a real checkpoint evaluation report.
- Laptop validation verifies policy, parser, evidence planning, memory, reports, and tests only.
- HF GPU training must be launched manually and monitored for credit safety.

## 13. Final Judge Claim

> ShadowOps currently proves production-safe behavior through the Q-aware policy guardrail. Trained model improvement is only claimed after checkpoint evaluation passes the model-vs-policy safety gate.

## Replay Summary

- Replay scenarios passed: 5/5