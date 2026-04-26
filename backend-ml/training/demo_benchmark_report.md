# ShadowOps Demo Benchmark

Validation samples: 100

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.360 | 0.800 | 0.200 | 0.163 | 0.087 |
| Heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.181 |
| Q-aware | 0.990 | 1.000 | 0.000 | 0.000 | 1.937 |
| Oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.958 |
