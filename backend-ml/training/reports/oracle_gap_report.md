# ShadowOps Oracle Gap Analysis

Samples: 100

## Before vs After Calibration

| Version | exact_match | reward_mean | misses |
| --- | ---: | ---: | ---: |
| Before | 0.920 | 1.866 | 8 |
| After | 0.990 | 1.920 | 1 |
| Oracle | 1.000 | 1.942 | 0 |

## Top Reasons Q-aware Differs From Oracle

- ambiguous network/vendor exposure needs quarantine until evidence is provided: 1

## Domain Distribution Of Misses

- SOC: 1

## Missed Samples

### val-00042
- Domain: SOC
- Q-aware: FORK
- Oracle: QUARANTINE
- Reward loss: 2.150
- Reason: ambiguous network/vendor exposure needs quarantine until evidence is provided
- Payload: SIEM flagged 192.168.1.195 for port scanning. User claims it was authorized pen-test.


## Fix Safety Assessment

Do not overfit the final miss. Only apply a policy improvement when the rule is generic, hidden evaluation does not degrade, and false positives do not increase.

- `val-00042`: safe_to_fix_generically=True | hidden_eval_risk=low if limited to exposure terms and missing evidence | recommendation=generic quarantine rule for medium/high ambiguous exposure with missing evidence