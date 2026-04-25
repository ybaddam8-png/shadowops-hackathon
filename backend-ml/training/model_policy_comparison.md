# ShadowOps Model vs Policy Comparison

## validation

Samples: 100

| Policy | Available | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | false_negative_rate | reward_mean | quarantine_precision | fork_precision | allow_precision | block_precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | True | 0.360 | 0.800 | 0.200 | 0.163 | 0.228 | 0.069 | 0.294 | 0.450 | 0.375 | 0.364 |
| heuristic | True | 0.520 | 0.920 | 0.080 | 0.000 | 0.140 | 1.040 | 0.000 | 0.552 | 0.814 | 0.036 |
| q_aware_policy | True | 0.850 | 1.000 | 0.000 | 0.000 | 0.000 | 1.645 | 1.000 | 0.800 | 1.000 | 0.652 |
| oracle | True | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| sft_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| grpo_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## hard_negative

Samples: 40

| Policy | Available | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | false_negative_rate | reward_mean | quarantine_precision | fork_precision | allow_precision | block_precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | True | 0.150 | 0.550 | 0.450 | 0.296 | 0.538 | -0.862 | 0.286 | 0.000 | 0.154 | 0.000 |
| heuristic | True | 0.725 | 0.975 | 0.025 | 0.000 | 0.077 | 1.375 | 0.000 | 0.231 | 0.933 | 1.000 |
| q_aware_policy | True | 0.350 | 0.975 | 0.025 | 0.037 | 0.000 | 0.830 | 0.350 | 0.000 | 0.667 | 0.833 |
| oracle | True | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2.137 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| sft_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| grpo_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
