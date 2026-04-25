# WARNING: This starts a Hugging Face GPU Job and may consume credits while Starting or Running.
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
bash training/hf_train_entrypoint.sh
'@

hf jobs run --flavor a10g-small --secrets HF_TOKEN --env TRAIN_MODE=grpo_smoke pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc "$Command"
