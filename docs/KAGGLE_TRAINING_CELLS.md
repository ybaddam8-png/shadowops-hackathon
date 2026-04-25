# Kaggle Training Cells

These cells prepare Kaggle for ShadowOps validation and optional smoke training. Training cells consume runtime/GPU quota and are optional.

## 1. GPU Check

```python
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 2. Clone Repo Branch

```python
!git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git shadowops
%cd shadowops/backend-ml
```

## 3. Laptop-Safe Validation First

```python
!python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
!python training/build_action_only_sft_dataset.py
!python training/model_policy_gate.py --skip-model-load
!python demo_judge.py
```

## 4. Install Training Requirements

Run this only inside the Kaggle/Linux GPU runtime:

```python
!pip install -r requirements.txt
```

Do not run the full training requirements on a normal Windows laptop.

## 5. Prompt Preflight

```python
!python training/train_qwen3_grpo.py --print-prompt-preflight
```

The prompt should request exactly one action token: `ALLOW`, `BLOCK`, `FORK`, or `QUARANTINE`.

## 6. Build Action-Only Dataset

```python
!python training/build_action_only_sft_dataset.py
```

## 7. Optional Smoke Training

This consumes Kaggle runtime/GPU quota. Run only after validation passes.

```python
# Optional SFT smoke. Keep steps small.
!python training/train_qwen3_grpo.py --run-sft --skip-grpo --max-steps 50 --train-size 100 --val-size 100
```

## 8. Checkpoint Evaluation

```python
!python training/evaluate_checkpoints.py --checkpoint kaggle=training/checkpoints/qwen3_sft_adapter --compare-against-policy --constrain-actions
!python training/model_policy_gate.py
```

Do not claim model improvement unless the generated checkpoint reports prove it.

## 9. Zip Outputs

```python
!zip -r shadowops_training_reports.zip training/reports training/*.json training/*.md
```
