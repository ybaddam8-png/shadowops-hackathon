# ShadowOps Action-Only SFT Dataset Audit

Total samples: 600
Train samples: 500
Validation samples: 100
Invalid labels: 0
Duplicate prompt/label rows: 0
Leakage warnings: 0
Passed: True

## Action Distribution

### train
- ALLOW: 242
- BLOCK: 88
- FORK: 87
- QUARANTINE: 83

### validation
- ALLOW: 42
- BLOCK: 15
- FORK: 20
- QUARANTINE: 23

## Examples Per Action

### ALLOW
- `train-00000` (train, AWS): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00001` (train, AWS): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00002` (train, AWS): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl

### FORK
- `train-00003` (train, AWS): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00004` (train, GITHUB): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00010` (train, SOC): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl

### QUARANTINE
- `train-00007` (train, SOC): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00008` (train, AWS): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00026` (train, AWS): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl

### BLOCK
- `train-00013` (train, GITHUB): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00025` (train, GITHUB): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
- `train-00034` (train, GITHUB): You are ShadowOps, a cybersecurity decision policy.

Choose exactly one action:
ALLOW, BLOCK, FORK, QUARANTINE.

Rules:
ALLOW = safe benign activity.
BLOCK = cl
