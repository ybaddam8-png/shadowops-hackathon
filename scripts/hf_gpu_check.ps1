# WARNING: This starts a Hugging Face GPU Job and may consume credits while Starting or Running.
# Run manually only after CPU validation passes and after confirming your Hugging Face credit budget.
# Requires a local HF_TOKEN environment variable; the token is passed with --secrets HF_TOKEN.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN is not set. Set `$env:HF_TOKEN before launching this paid Hugging Face Job."
}

$Command = @'
apt-get update && apt-get install -y git
git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo
cd repo/backend-ml
pip install packaging rich pydantic numpy
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
python training/train_qwen3_grpo.py --help
'@

hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc "$Command"
