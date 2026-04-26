# ShadowOps Model Evaluation Report

Model: `shadowops_qwen3_1p7b_model`
Checkpoint: `shadowops_qwen3_1p7b_model`
Eval split: `validation`
Samples: 100

## Metrics

| Metric | Model | Q-aware | Delta |
| --- | ---: | ---: | ---: |
| exact_match | n/a | 0.990 | n/a |
| safety_accuracy | n/a | 1.000 | n/a |
| unsafe_decision_rate | n/a | 0.000 | n/a |
| false_positive_rate | n/a | 0.000 | n/a |
| reward_mean | n/a | 1.937 | n/a |
| invalid_output_rate | n/a | 0.000 | n/a |

## Training Gate

Status: **FAIL**

Reason: No model metrics are available; checkpoint was not loaded or evaluation failed.

Recommended next action: Run --evaluate-model with a valid --model-path after SFT/GRPO smoke training.
