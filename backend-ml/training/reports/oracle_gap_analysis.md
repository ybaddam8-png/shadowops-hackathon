# ShadowOps Oracle Gap Analysis

Samples: 100

## Before vs After Calibration

| Version | exact_match | reward_mean | misses |
| --- | ---: | ---: | ---: |
| Before | 0.930 | 1.866 | 7 |
| After | 0.990 | 1.899 | 1 |
| Oracle | 1.000 | 1.920 | 0 |

## Top Reasons Q-aware Differs From Oracle

- human review is sufficient; quarantine is too conservative: 1

## Domain Distribution Of Misses

- AWS: 1

## Missed Samples

### val-00020
- Domain: AWS
- Q-aware: QUARANTINE
- Oracle: FORK
- Reward loss: 2.141
- Reason: human review is sufficient; quarantine is too conservative
- Payload: New security group rule: allow 23.95.248.104/32 on port 22. Vendor claims it's monitoring.
