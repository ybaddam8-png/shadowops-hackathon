# ShadowOps Hackathon

ShadowOps is an OpenEnv-compatible autonomous cyber incident response environment. It evaluates whether an agent should `ALLOW`, `BLOCK`, `FORK`, or `QUARANTINE` risky operational actions across CI/CD, cloud IAM, public storage, firewall, pentest, and deployment workflows.

ShadowOps currently reports policy-baseline performance from reproducible benchmark artifacts.

Trained checkpoint metrics are reported only after `model_eval_report.json` or `checkpoint_comparison_report.json` is generated from a real checkpoint.

Q-aware supervisor is the safety guardrail. A trained checkpoint must pass the model-vs-policy gate before it is considered a deployment candidate.

ShadowOps prevents common RL action-format failures by requiring exact action outputs: `ALLOW`, `BLOCK`, `FORK`, or `QUARANTINE`.

## Local Validation

```powershell
cd backend-ml
python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
python training/build_action_only_sft_dataset.py
python training/run_hidden_eval.py
python training/model_policy_gate.py --skip-model-load
python training/generate_reward_curves.py
python training/check_submission_artifacts.py
python demo_judge.py
```

These commands are CPU-safe and do not run SFT, GRPO, GPU jobs, or Hugging Face Jobs.

## Key Artifacts

- OpenEnv manifest: `openenv.yaml`
- Backend OpenEnv wrapper: `backend-ml/openenv_shadowops_env.py`
- Benchmark report: `backend-ml/training/demo_benchmark_report.json`
- Hidden eval report: `backend-ml/training/reports/hidden_eval_report.json`
- Reward diagnostics: `backend-ml/training/reports/reward_diagnostics_report.json`
- Model-policy gate: `backend-ml/training/reports/model_policy_gate_report.json`
- Reward curve report: `backend-ml/training/reports/reward_curve_report.md`
- Training plot guide: `docs/REWARD_CURVES_AND_TRAINING_PLOTS.md`
- Submission draft: `docs/SUBMISSION_README_DRAFT.md`
- Champion checkpoint guide: `docs/CHAMPION_CHECKPOINT_INTEGRATION.md`
- Kaggle training cells: `docs/KAGGLE_TRAINING_CELLS.md`
