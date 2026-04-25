# ShadowOps Hidden Evaluation Report

## validation

Samples: 100

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.360 | 0.800 | 0.200 | 0.163 | 0.083 |
| heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.146 |
| q_aware | 0.990 | 1.000 | 0.000 | 0.000 | 1.899 |
| oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.920 |

Q-aware exact 95% CI: [0.9704982462327102, 1.0]

## hard_negative

Samples: 40

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.150 | 0.550 | 0.450 | 0.296 | -1.006 |
| heuristic | 0.725 | 0.975 | 0.025 | 0.000 | 1.283 |
| q_aware | 0.475 | 0.950 | 0.050 | 0.074 | 1.201 |
| oracle | 1.000 | 1.000 | 0.000 | 0.000 | 2.152 |

Q-aware exact 95% CI: [0.3202422053659332, 0.6297577946340668]

## hidden_eval

Samples: 150

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.233 | 0.733 | 0.267 | 0.256 | 0.033 |
| heuristic | 0.800 | 1.000 | 0.000 | 0.000 | 1.672 |
| q_aware | 1.000 | 1.000 | 0.000 | 0.000 | 1.848 |
| oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.928 |

Q-aware exact 95% CI: [1.0, 1.0]

## multi_step_chain

Samples: 3

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.000 | 0.667 | 0.333 | 0.000 | -1.300 |
| heuristic | 0.667 | 1.000 | 0.000 | 0.000 | 1.601 |
| q_aware | 0.667 | 1.000 | 0.000 | 0.000 | 1.614 |
| oracle | 1.000 | 1.000 | 0.000 | 0.000 | 2.094 |

Q-aware exact 95% CI: [0.13322223379388565, 1.0]
