# Hugging Face Space Deployment Plan

This document is a packaging plan only. Do not run these commands unless you intentionally deploy the Space.

## Files To Include
- `backend-ml/main.py`
- `backend-ml/shadowops_env.py`
- `backend-ml/openenv_shadowops.py`
- `backend-ml/openenv_shadowops_env.py`
- `backend-ml/openenv.yaml`
- `backend-ml/schema_contract.json`
- `backend-ml/agent_memory.py`
- `backend-ml/risk_accumulator.py`
- `backend-ml/evidence_planner.py`
- `backend-ml/domain_policies.py`
- `backend-ml/safe_outcome.py`
- `backend-ml/demo_judge.py`
- `backend-ml/demo_replays.py`
- `backend-ml/training/*.py`
- `backend-ml/training/*.json`
- `backend-ml/training/*.md`
- `backend-ml/training/plots/*.png`
- `docs/SUBMISSION_README_DRAFT.md`
- `docs/DEMO_PITCH_SCRIPT.md`
- `docs/JUDGE_DEMO_CHECKLIST.md`

Do not include:
- Hugging Face tokens
- `.env`
- model checkpoints
- `*.pt`
- `*.pth`
- `*.safetensors`
- cache folders

## Suggested Space Runtime
Use a CPU Space for the judge demo. The submitted baseline path does not require a GPU.

Recommended local preparation:

```powershell
cd backend-ml
python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
python training/generate_submission_plots.py
python demo_judge.py
```

## Safe Deployment Commands
These are documentation-only commands. Run them manually only when ready.

```powershell
pip install -U huggingface_hub
hf auth login
hf auth whoami
```

Create or clone the Space manually from the Hugging Face UI, then copy the backend demo files into the Space repository.

## Credit Safety
- CPU Space deployment should not start GPU training.
- Do not run `hf jobs run` from deployment steps.
- Do not upload checkpoints unless real model evidence is ready.
- The production guardrail remains `q_aware_demo_policy` until checkpoint evaluation passes.
