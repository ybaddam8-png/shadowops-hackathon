# ShadowOps Dataset Audit

- Train samples: 500
- Validation samples: 100
- Hard-negative samples: 40
- Duplicate prompts: 0
- Train/validation overlap: 0
- Missing labels: 0
- Invalid action labels: 0
- False-positive challenges: 192
- Preflight: PASS

## Action Distribution

### train

| Action | Count |
| --- | ---: |
| ALLOW | 242 |
| BLOCK | 88 |
| FORK | 87 |
| QUARANTINE | 83 |

### val

| Action | Count |
| --- | ---: |
| ALLOW | 42 |
| BLOCK | 15 |
| FORK | 20 |
| QUARANTINE | 23 |

### hard_negative

| Action | Count |
| --- | ---: |
| ALLOW | 14 |
| BLOCK | 13 |
| FORK | 3 |
| QUARANTINE | 10 |

## Domain Distribution

| Domain | Count |
| --- | ---: |
| AWS | 203 |
| GITHUB | 208 |
| SOC | 189 |
| aws_s3 | 9 |
| github_ci | 14 |
| iam | 8 |
| network_firewall | 5 |
| pentest | 4 |