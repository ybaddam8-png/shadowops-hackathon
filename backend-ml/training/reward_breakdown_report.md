# ShadowOps Reward Breakdown

Label: Policy baseline / pre-training validation
Validation samples: 100

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | reward_mean | action_correctness | safety_penalty | false_positive_penalty | evidence_completeness | risk_chain | uncertainty | policy_compliance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.350 | 0.800 | 0.200 | 0.065 | 0.703 | -0.590 | -0.132 | 0.250 | 0.000 | 0.004 | -0.052 |
| heuristic | 0.510 | 0.920 | 0.080 | 1.146 | 1.244 | -0.240 | 0.000 | 0.250 | 0.000 | 0.000 | 0.094 |
| q_aware | 0.990 | 1.000 | 0.000 | 1.920 | 1.770 | 0.000 | 0.000 | 0.250 | 0.000 | 0.006 | 0.144 |
| oracle | 1.000 | 1.000 | 0.000 | 1.942 | 1.785 | 0.000 | 0.000 | 0.250 | 0.000 | 0.006 | 0.144 |

## Notes

- No trained model improvement is claimed in this report.
- The breakdown shows how baseline policies receive reward credit or penalties.
- Real model metrics should be merged only after checkpoint evaluation.