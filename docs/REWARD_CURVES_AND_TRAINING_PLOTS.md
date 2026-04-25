# Reward Curves And Training Plots

ShadowOps generates judge-friendly PNG plots from real training and evaluation artifacts only. Missing training logs are reported as pending; the plotter does not invent curve points.

## Generated Plots

- `backend-ml/training/plots/reward_curve.png`
- `backend-ml/training/plots/loss_curve.png`
- `backend-ml/training/plots/reward_std_curve.png`
- `backend-ml/training/plots/grad_norm_curve.png`
- `backend-ml/training/plots/completion_length_curve.png`
- `backend-ml/training/plots/invalid_output_rate_curve.png`
- `backend-ml/training/plots/model_policy_comparison.png`
- `backend-ml/training/plots/safety_reward_comparison.png`

Curve plots are created only when matching real values are present in artifacts such as `trainer_state.json`, `reward_curves*.json`, reward diagnostics, model eval reports, or checkpoint comparison reports.

## Run After Evaluation

```powershell
cd backend-ml

python training/generate_reward_curves.py
```

Reports are saved to:

- `backend-ml/training/reports/reward_curve_data.json`
- `backend-ml/training/reports/reward_curve_report.md`

## Compare SFT vs GRPO

```powershell
cd backend-ml

python training/compare_training_runs.py `
  --run sft=training/checkpoints/qwen3_sft_adapter `
  --run grpo=training/checkpoints/qwen3_sft_grpo_adapter_explore `
  --output-dir training/plots
```

If a run folder does not exist, the script warns and continues with available runs.

## Generate Automatically After Training

After a real training run, pass `--generate-plots`:

```powershell
cd backend-ml

python training/train_qwen3_grpo.py --config training/configs/grpo_explore.json --generate-plots
```

Use only flags supported by the local training script. Do not claim model improvement unless the generated model evaluation or checkpoint comparison report proves it.

## Presentation Use

Use `model_policy_comparison.png` and `safety_reward_comparison.png` for the baseline story before training. Use the curve plots only after real logs exist.

Clear judge line:

“Plots are generated only from real training/evaluation artifacts.”
