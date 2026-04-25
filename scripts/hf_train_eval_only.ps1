# WARNING: This starts a Hugging Face CPU Job and may consume credits while Starting or Running.
# Requires a local HF_TOKEN environment variable; the token is passed with --secrets HF_TOKEN.

$Command = @'
apt-get update && apt-get install -y git
git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo
cd repo/backend-ml
bash training/hf_train_entrypoint.sh
'@

hf jobs run python:3.12-bookworm --secrets HF_TOKEN --env TRAIN_MODE=eval_only -- bash -lc $Command
