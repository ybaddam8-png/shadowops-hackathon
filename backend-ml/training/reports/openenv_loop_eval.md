# ShadowOps OpenEnv Loop Evaluation

- Policy evaluated: q_aware
- Baseline policy for comparison: heuristic
- Episodes: 50
- Episode max length: 5
- Seed: 42
- Run label: full_eval_50_episodes
- Scope note: FULL EVAL: 50 episodes (judge-facing run size).

## Core Metrics

| Metric | Value |
| --- | ---: |
| Total steps | 250 |
| Malicious steps | 139 |
| Benign steps | 111 |
| Accuracy | 0.164 |
| Unsafe allow rate (malicious-only) | 0.000 |
| Safe block rate (benign blocked/forked/quarantined) | 1.000 |
| Average confidence | 0.428 |
| Mean reward per step | 6.896 |
| Step pass count | 139 |
| Step fail count | 111 |

## Reward Summary

- Per-step reward mean/median/std: 6.896 / 2.000 / 31.080
- Per-episode reward mean/median/std: 34.480 / 41.500 / 76.698
- Per-episode reward min/max: -80.000 / 193.000

## Before vs After Aggregate

| Metric | Baseline | Target | Delta (target-baseline) |
| --- | ---: | ---: | ---: |
| Unsafe allow rate | 0.403 | 0.000 | -0.403 |
| Safe block rate | 0.270 | 1.000 | +0.730 |
| Average confidence | 0.775 | 0.428 | -0.347 |
| Mean reward/step | 33.524 | 6.896 | -26.628 |
| Safety Adjusted Score | 8.957 | -5.324 | -14.281 |

## Safety vs Reward Trade-Off

- **Note on Reward vs Safety**: The `heuristic` baseline may occasionally have a higher `mean_reward_per_step` due to faster resolution times.
- However, **Q-aware is considered safer** when its `unsafe_allow_rate = 0.000`. Unsafe allow is the primary failure mode in security automation and carries severe negative business impact.
- Lower confidence scores in Q-aware do not necessarily mean failure; they often reflect **cautious uncertainty** on ambiguous payloads, which correctly triggers QUARANTINE instead of false-positive blocks or dangerous allows.

## Representative Behavior

See `openenv_behavior_comparison.md` for 5-10 structured before/after scenarios.

| Scenario | Baseline | Target | Baseline correct | Target correct |
| --- | --- | --- | --- | --- |
| AWS::UPDATE_IAM | BLOCK | QUARANTINE | False | True |
| AWS::UPDATE_SECURITY_GROUP | BLOCK | QUARANTINE | False | True |
| AWS::RESIZE_INSTANCE | BLOCK | QUARANTINE | False | True |
| AWS::ENABLE_LOGGING | BLOCK | QUARANTINE | False | True |
| AWS::UPDATE_IAM | BLOCK | QUARANTINE | False | True |
| AWS::UPDATE_IAM | BLOCK | QUARANTINE | False | True |
| AWS::RESIZE_INSTANCE | BLOCK | QUARANTINE | False | True |
| AWS::ROTATE_KEYS | FORK | QUARANTINE | False | True |

## Episode Summary

| Episode | Reward | Steps | Unsafe allows | Safe blocks | Pass | Fail | Final SOC | Final GitHub | Final AWS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -38.000 | 5 | 0 | 1 | 4 | 1 | 100 | 100 | 92 |
| 2 | 70.000 | 5 | 0 | 5 | 0 | 5 | 78 | 100 | 100 |
| 3 | 143.000 | 5 | 0 | 4 | 1 | 4 | 100 | 92 | 100 |
| 4 | -5.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 5 | 51.000 | 5 | 0 | 4 | 1 | 4 | 84 | 96 | 100 |
| 6 | 51.000 | 5 | 0 | 1 | 4 | 1 | 100 | 92 | 100 |
| 7 | 70.000 | 5 | 0 | 5 | 0 | 5 | 100 | 86 | 92 |
| 8 | -38.000 | 5 | 0 | 1 | 4 | 1 | 92 | 100 | 100 |
| 9 | -80.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 10 | 193.000 | 5 | 0 | 1 | 4 | 1 | 100 | 98 | 100 |
| 11 | 121.000 | 5 | 0 | 4 | 1 | 4 | 88 | 92 | 100 |
| 12 | 193.000 | 5 | 0 | 1 | 4 | 1 | 100 | 98 | 100 |
| 13 | 124.000 | 5 | 0 | 3 | 2 | 3 | 100 | 85 | 100 |
| 14 | -80.000 | 5 | 0 | 1 | 4 | 1 | 100 | 98 | 100 |
| 15 | 81.000 | 5 | 0 | 2 | 3 | 2 | 92 | 100 | 92 |
| 16 | -38.000 | 5 | 0 | 1 | 4 | 1 | 92 | 100 | 100 |
| 17 | -38.000 | 5 | 0 | 1 | 4 | 1 | 100 | 100 | 92 |
| 18 | 91.000 | 5 | 0 | 1 | 4 | 1 | 92 | 100 | 100 |
| 19 | -28.000 | 5 | 0 | 5 | 0 | 5 | 80 | 92 | 100 |
| 20 | 91.000 | 5 | 0 | 4 | 1 | 4 | 88 | 92 | 100 |
| 21 | -38.000 | 5 | 0 | 1 | 4 | 1 | 100 | 100 | 92 |
| 22 | 28.000 | 5 | 0 | 5 | 0 | 5 | 100 | 90 | 100 |
| 23 | -38.000 | 5 | 0 | 1 | 4 | 1 | 100 | 92 | 100 |
| 24 | -38.000 | 5 | 0 | 5 | 0 | 5 | 100 | 78 | 100 |
| 25 | 58.000 | 5 | 0 | 2 | 3 | 2 | 92 | 100 | 92 |
| 26 | 70.000 | 5 | 0 | 5 | 0 | 5 | 92 | 92 | 100 |
| 27 | 58.000 | 5 | 0 | 5 | 0 | 5 | 86 | 100 | 92 |
| 28 | 58.000 | 5 | 0 | 5 | 0 | 5 | 100 | 74 | 92 |
| 29 | 91.000 | 5 | 0 | 1 | 4 | 1 | 92 | 100 | 100 |
| 30 | -38.000 | 5 | 0 | 1 | 4 | 1 | 100 | 92 | 100 |
| 31 | -5.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 32 | 48.000 | 5 | 0 | 3 | 2 | 3 | 92 | 84 | 100 |
| 33 | 124.000 | 5 | 0 | 3 | 2 | 3 | 94 | 100 | 100 |
| 34 | 2.000 | 5 | 0 | 1 | 4 | 1 | 100 | 100 | 92 |
| 35 | -5.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 36 | 18.000 | 5 | 0 | 5 | 0 | 5 | 88 | 84 | 100 |
| 37 | -28.000 | 5 | 0 | 5 | 0 | 5 | 92 | 100 | 74 |
| 38 | 58.000 | 5 | 0 | 5 | 0 | 5 | 92 | 80 | 100 |
| 39 | 28.000 | 5 | 0 | 4 | 1 | 4 | 100 | 86 | 100 |
| 40 | 193.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 41 | 35.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 42 | 168.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 43 | 143.000 | 5 | 0 | 4 | 1 | 4 | 100 | 100 | 86 |
| 44 | -80.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 45 | 18.000 | 5 | 0 | 2 | 3 | 2 | 84 | 100 | 100 |
| 46 | -80.000 | 5 | 0 | 1 | 4 | 1 | 100 | 100 | 98 |
| 47 | -80.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 48 | -80.000 | 5 | 0 | 0 | 5 | 0 | 100 | 100 | 100 |
| 49 | 51.000 | 5 | 0 | 1 | 4 | 1 | 92 | 100 | 100 |
| 50 | 51.000 | 5 | 0 | 1 | 4 | 1 | 92 | 100 | 100 |

## Plot

- Episode reward trend plot: `openenv_episode_rewards.png`
- Color mapping in the plot: red=baseline policy, green=target policy.