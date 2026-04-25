# ShadowOps Benchmark Table

Validation source: `training\qwen3_val_dataset.json`

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean | invalid_output_rate | zero_std_reward_group_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.360 | 0.800 | 0.200 | 0.163 | 0.083 | 0.000 | 0.000 |
| Heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.146 | 0.000 | 0.000 |
| Q-aware | 0.930 | 1.000 | 0.000 | 0.000 | 1.866 | 0.000 | 0.000 |
| Oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.920 | 0.000 | 0.000 |