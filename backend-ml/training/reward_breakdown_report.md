# ShadowOps Reward Breakdown

Label: Policy baseline / pre-training validation
Validation samples: 100

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | reward_mean | action_correctness | safety_penalty | false_positive_penalty | evidence_completeness | risk_chain | uncertainty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.360 | 0.800 | 0.200 | 0.083 | 0.713 | -0.590 | -0.108 | 0.250 | 0.000 | 0.004 |
| heuristic | 0.520 | 0.920 | 0.080 | 1.146 | 1.224 | -0.240 | 0.000 | 0.250 | 0.000 | 0.000 |
| q_aware | 0.930 | 1.000 | 0.000 | 1.866 | 1.645 | 0.000 | 0.000 | 0.250 | 0.000 | 0.006 |
| oracle | 1.000 | 1.000 | 0.000 | 1.920 | 1.750 | 0.000 | 0.000 | 0.250 | 0.000 | 0.006 |

## Notes

- No trained model improvement is claimed in this report.
- The breakdown shows how baseline policies receive reward credit or penalties.
- Real model metrics should be merged only after checkpoint evaluation.