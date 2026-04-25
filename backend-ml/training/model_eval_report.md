# ShadowOps Model Evaluation Report

Model: `training\checkpoints\qwen3_sft_grpo_adapter`
Checkpoint: `training\checkpoints\qwen3_sft_grpo_adapter`
Eval split: `validation`
Samples: 100

## Metrics

| Metric | Model | Q-aware | Delta |
| --- | ---: | ---: | ---: |
| exact_match | n/a | 0.930 | n/a |
| safety_accuracy | n/a | 1.000 | n/a |
| unsafe_decision_rate | n/a | 0.000 | n/a |
| false_positive_rate | n/a | 0.000 | n/a |
| reward_mean | n/a | 1.866 | n/a |
| invalid_output_rate | n/a | 0.000 | n/a |

## Training Gate

Status: **FAIL**

Reason: No model metrics are available; checkpoint was not loaded or evaluation failed.

Recommended next action: Run --evaluate-model with a valid --model-path after SFT/GRPO smoke training.
