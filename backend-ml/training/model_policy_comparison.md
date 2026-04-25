# ShadowOps Model vs Policy Comparison

## validation

Samples: 100

| Policy | Available | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | false_negative_rate | reward_mean | quarantine_precision | fork_precision | allow_precision | block_precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | True | 0.350 | 0.800 | 0.200 | 0.163 | 0.228 | 0.065 | 0.294 | 0.350 | 0.417 | 0.364 |
| heuristic | True | 0.510 | 0.920 | 0.080 | 0.000 | 0.140 | 1.146 | 0.000 | 0.364 | 0.840 | 0.036 |
| q_aware_policy | True | 0.990 | 1.000 | 0.000 | 0.000 | 0.000 | 1.920 | 1.000 | 0.952 | 1.000 | 1.000 |
| oracle | True | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.942 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| sft_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| grpo_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## hard_negative

Samples: 40

| Policy | Available | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | false_negative_rate | reward_mean | quarantine_precision | fork_precision | allow_precision | block_precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | True | 0.150 | 0.550 | 0.450 | 0.296 | 0.538 | -1.006 | 0.286 | 0.000 | 0.154 | 0.000 |
| heuristic | True | 0.725 | 0.975 | 0.025 | 0.000 | 0.077 | 1.283 | 0.000 | 0.231 | 0.933 | 1.000 |
| q_aware_policy | True | 0.475 | 0.950 | 0.050 | 0.074 | 0.000 | 1.201 | 0.467 | 0.000 | 0.833 | 0.778 |
| oracle | True | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2.152 | 1.000 | 1.000 | 1.000 | 1.000 |
| raw_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| sft_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| grpo_model | False | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
