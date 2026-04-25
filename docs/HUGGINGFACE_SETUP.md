# Hugging Face Cloud Validation Setup

This document prepares the manual Hugging Face Jobs setup for ShadowOps validation. Do not paste tokens into repository files, scripts, logs, or pull requests.

## Login

```bash
pip install -U huggingface_hub
hf auth login
hf auth whoami
```

## Token Permissions

Required:

- Jobs: Start and manage Jobs on your behalf

Recommended:

- Repository read access for public/gated repos

Optional:

- Repository write access only if uploading checkpoints

## Manual CPU Validation

Run this manually only when you are ready to spend Hugging Face Jobs credits:

```bash
hf jobs run python:3.12-bookworm -- bash -lc "apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install -r requirements.txt && python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load && python demo_judge.py"
```

## Manual GPU Environment Check

Run this manually only after CPU validation passes:

```bash
hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc "apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install -r requirements.txt && python -c \"import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))\" && python training/train_qwen3_grpo.py --help"
```

## Credit Safety

- Hugging Face Jobs can consume credits while Starting or Running.
- Do not run GPU jobs repeatedly.
- CPU validation should be run manually first.
- GPU check should be run manually only after CPU validation passes.
- Full training should not be started until smoke tests are successful.
- Start with only a 50-step SFT smoke run later.
