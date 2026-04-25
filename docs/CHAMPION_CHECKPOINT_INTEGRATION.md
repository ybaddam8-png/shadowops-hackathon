# Champion Checkpoint Integration

ShadowOps uses the Q-aware supervisor as the production safety guardrail until a real checkpoint passes the model-vs-policy gate.

Do not commit model weights, adapters, checkpoints, `.pt`, `.pth`, or `.safetensors` files to GitHub.

## Local Checkpoint Path

Place an external adapter locally under:

```powershell
backend-ml\training\checkpoints\yashwanth_adapter
```

This folder is intentionally ignored by Git.

## Hugging Face Model Path

You can also evaluate a model repo path directly:

```powershell
cd backend-ml

python training/evaluate_checkpoints.py `
  --checkpoint yashwanth=ybaddam8-png/some-model-path `
  --compare-against-policy `
  --constrain-actions
```

Only run that command when you intentionally want to load/download the model.

## Local Adapter Evaluation

```powershell
cd backend-ml

python training/evaluate_checkpoints.py `
  --checkpoint yashwanth=training/checkpoints/yashwanth_adapter `
  --compare-against-policy `
  --constrain-actions

python training/model_policy_gate.py
```

Report only real measured metrics from:

- `backend-ml/training/reports/checkpoint_comparison_report.json`
- `backend-ml/training/reports/model_policy_gate_report.json`

If no checkpoint beats Q-aware safely, deployment remains Q-aware and the checkpoint is reported as a training candidate only.

## Kaggle Flow

```python
!git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git repo
%cd repo/backend-ml
!python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
!python training/build_action_only_sft_dataset.py
```

After a real Kaggle training run writes an adapter, evaluate it before making any model-improvement claim:

```python
!python training/evaluate_checkpoints.py --checkpoint kaggle=training/checkpoints/kaggle_adapter --compare-against-policy --constrain-actions
!python training/model_policy_gate.py
```
