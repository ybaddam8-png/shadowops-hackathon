# WARNING: This starts a Hugging Face Job and may consume credits while Starting or Running.
# Run manually only after confirming your Hugging Face token permissions and credit budget.
# Requires a local HF_TOKEN environment variable; the token is passed with --secrets HF_TOKEN.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN is not set. Set `$env:HF_TOKEN before launching this paid Hugging Face Job."
}

hf jobs run python:3.12-bookworm --secrets HF_TOKEN -- bash -lc 'apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install packaging rich pydantic numpy && python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load && python demo_judge.py'
