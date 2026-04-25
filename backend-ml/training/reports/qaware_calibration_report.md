# Q-aware Policy Calibration Report

Samples: 100
Calibration split: 50
Held-out split: 50

Selected improved: True
Selected reason: narrow network/vendor quarantine terms

| Candidate | safe | held_out_exact | held_out_safety | held_out_unsafe | held_out_fpr | held_out_reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | True | 0.920 | 1.000 | 0.000 | 0.000 | 1.860 |
| 2 | True | 1.000 | 1.000 | 0.000 | 0.000 | 1.921 |
| 3 | True | 1.000 | 1.000 | 0.000 | 0.000 | 1.921 |

## Safety Gate

- Safety floor: 1.0
- Max unsafe decision rate: 0.0
- Max false-positive rate: 0.01