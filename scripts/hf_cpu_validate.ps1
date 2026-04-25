# WARNING: This starts a Hugging Face Job and may consume credits while Starting or Running.
# Run manually only after confirming your Hugging Face token permissions and credit budget.

hf jobs run python:3.12-bookworm -- bash -lc 'apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install -r requirements.txt && python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load && python demo_judge.py'
