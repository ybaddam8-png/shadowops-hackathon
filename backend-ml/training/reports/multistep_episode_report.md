# ShadowOps Multi-step Episode Evaluation

- Chains: 5
- Chain detection accuracy: 1.000
- Safe chain allow rate: 1.000
- Malicious chain block/quarantine rate: 1.000
- Unnecessary escalation rate: 0.000
- Memory risk improvement: 0.047

| Chain | Final decision | Detected | Risk start | Risk end |
| --- | --- | --- | ---: | ---: |
| firewall-iam-export | QUARANTINE | True | 1.000 | 1.000 |
| ci-secret-prod | QUARANTINE | True | 0.790 | 0.842 |
| public-bucket-escalation | QUARANTINE | True | 0.948 | 1.000 |
| approved-pentest | ALLOW | False | 0.000 | 0.000 |
| approved-break-glass | ALLOW | False | 0.401 | 0.533 |