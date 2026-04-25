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
| ALLOW | 189 |
| BLOCK | 88 |
| FORK | 156 |
| QUARANTINE | 67 |

### val

| Action | Count |
| --- | ---: |
| ALLOW | 35 |
| BLOCK | 15 |
| FORK | 28 |
| QUARANTINE | 22 |

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