# WARNING: This starts a Hugging Face GPU Job and may consume credits while Starting or Running.
# Run manually only after CPU validation passes and after confirming your Hugging Face credit budget.

hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -lc 'apt-get update && apt-get install -y git && git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo && cd repo/backend-ml && pip install -r requirements.txt && python -c "import torch; print(''CUDA:'', torch.cuda.is_available()); print(''GPU:'', torch.cuda.get_device_name(0))" && python training/train_qwen3_grpo.py --help'
