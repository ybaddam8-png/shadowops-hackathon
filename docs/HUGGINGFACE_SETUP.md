# Hugging Face Cloud Validation Setup

This document prepares the manual Hugging Face Jobs setup for ShadowOps validation. Do not paste tokens into repository files, scripts, logs, or pull requests.

## Local Laptop Dependencies

Use the lightweight dependency file on a laptop:

```bash
cd backend-ml
pip install -r requirements-lite.txt
```

`requirements-lite.txt` is only for local validation, policy evaluation, reports, and tests. It intentionally excludes `torch`, `transformers`, `trl`, `unsloth`, `vllm`, and `xformers`.

Use the full `requirements.txt` only inside the Linux GPU training environment. Do not run `pip install -r requirements.txt` on the Windows laptop unless you are intentionally setting up a full training machine.

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
hf jobs run python:3.12-bookworm --secrets HF_TOKEN -- bash -lc "apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install packaging rich pydantic numpy && python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load && python demo_judge.py"
```

## Manual GPU Environment Check

Run this manually only after CPU validation passes:

```bash
hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc "apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install packaging rich pydantic numpy && python - <<'PY'
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
python training/train_qwen3_grpo.py --help"
```

## Manual Training Launchers

Confirm login:

```bash
hf auth whoami
```

Confirm `HF_TOKEN` exists locally in PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

Do not save the token in the repo, scripts, docs, logs, or pull requests. The launchers pass the token securely to Hugging Face Jobs with `--secrets HF_TOKEN`.

Run eval-only first:

```powershell
.\scripts\hf_train_eval_only.ps1
```

Run SFT smoke only after eval passes:

```powershell
.\scripts\hf_train_sft_smoke.ps1
```

Run GRPO smoke only after SFT smoke passes:

```powershell
.\scripts\hf_train_grpo_smoke.ps1
```

Monitor jobs:

```powershell
.\scripts\hf_jobs_list.ps1
```

For stop/cancel guidance:

```powershell
.\scripts\hf_jobs_stop_help.ps1
```

Credit safety:

- GPU jobs consume credits while starting/running.
- Do not run multiple GPU jobs at once.
- Do not run full training until smoke tests pass.
- Stop/cancel stuck jobs immediately.
- Keep $30 credits safe by doing eval -> SFT smoke -> GRPO smoke.

Training gate:

- Run eval-only and reward diagnostics before any GPU job.
- Evaluate every SFT/GRPO checkpoint with `--evaluate-model --model-path <checkpoint> --compare-against-policy`.
- Do not claim model improvement unless the model evaluation report shows safety/reward improvement over raw or SFT and compares cleanly against Q-aware.
- Long training should wait until `training/model_eval_report.json` exists and the gate is PASS or an intentional WARN with a clear next action.
- Stop spending credits if unsafe decisions increase, invalid output rate increases, or reward diagnostics show no useful reward variation.

## Credit Safety

- Hugging Face Jobs can consume credits while Starting or Running.
- Do not run GPU jobs repeatedly.
- CPU validation should be run manually first.
- GPU check should be run manually only after CPU validation passes.
- Full training should not be started until smoke tests are successful.
- Start with only a 50-step SFT smoke run later.
