# ShadowOps Training Evaluation Playbook

This playbook keeps training claims judge-proof. A falling loss is not enough: every checkpoint must be compared against the validation split, hard negatives, and the Q-aware policy baseline.

## Laptop Setup

Use only the lightweight dependency file for laptop validation:

```powershell
cd backend-ml
pip install -r requirements-lite.txt
```

Use `requirements.txt` only inside the Linux GPU training environment.

## Safe Order

1. Run baseline-only validation:

   ```powershell
   cd backend-ml
   python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
   ```

2. Run reward diagnostics:

   ```powershell
   python training/train_qwen3_grpo.py --reward-diagnostics --skip-model-load
   ```

3. Run SFT smoke manually on Hugging Face only after local eval passes:

   ```powershell
   ..\scripts\hf_train_sft_smoke.ps1
   ```

4. Evaluate the SFT checkpoint:

   ```powershell
   python training/train_qwen3_grpo.py --evaluate-model --model-path training/cloud_runs/sft_smoke --compare-against-policy
   ```

5. Run GRPO smoke only after SFT smoke and SFT evaluation work:

   ```powershell
   ..\scripts\hf_train_grpo_smoke.ps1
   ```

6. Evaluate the GRPO checkpoint:

   ```powershell
   python training/train_qwen3_grpo.py --evaluate-model --model-path training/cloud_runs/grpo_smoke --compare-against-policy
   ```

7. Only then run a longer GRPO explore config.

8. Stop training if safety drops, unsafe decisions increase, invalid output rate rises, or the model-vs-policy gate fails.

## Gate Rules

- Do not claim training improved unless the model evaluation report proves it.
- Do not claim the model beats Q-aware unless `delta_vs_q_aware.reward_mean` and safety metrics prove it.
- `training_gate_status=PASS` means safety/unsafe thresholds passed and reward improved over a reference model.
- `training_gate_status=WARN` means some improvement may exist, but the proof is incomplete or still below policy.
- `training_gate_status=FAIL` means stop and inspect data, reward diagnostics, parsing, or checkpoint output.

## Hugging Face Budget Guard

- Eval-only first.
- Smoke training second.
- Long training only after `model_eval_report.json` works.
- Stop if logs show no reward improvement, high `frac_reward_zero_std`, high invalid output rate, or a failed gate.
- Keep credits safe by running eval -> SFT smoke -> GRPO smoke before any longer run.

## Post-Training Artifact Refresh

After real training finishes, run:

```powershell
cd backend-ml
python training/post_training_artifacts.py
python training/train_qwen3_grpo.py --evaluate-model --model-path <REAL_CHECKPOINT_PATH> --compare-against-policy --generate-plots
python training/generate_reward_curves.py
python training/check_submission_artifacts.py
```

- Reward curves are generated only from real trainer logs (`trainer_state.json`, `metrics.jsonl`, or real `reward_curve_data.json`).
- If logs/checkpoint metrics are missing, reports stay in `PENDING_REAL_TRAINING_LOGS` instead of inventing values.
- Q-aware policy remains the baseline safety guardrail and model gate reference.
- A trained checkpoint must pass action-format validation, hidden eval, and model-vs-policy gating.
- Do not claim model improvement until real checkpoint evaluation evidence exists.
