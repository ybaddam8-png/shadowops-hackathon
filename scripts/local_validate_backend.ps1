# Laptop-safe ShadowOps backend validation.
# This script does not run training, install full GPU dependencies, start HF Jobs, or consume credits.

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot
try {
    python -m py_compile backend-ml/evidence_planner.py
    python -m py_compile backend-ml/agent_memory.py
    python -m py_compile backend-ml/risk_accumulator.py
    python -m py_compile backend-ml/safe_outcome.py
    python -m py_compile backend-ml/openenv_shadowops.py
    python -m py_compile backend-ml/openenv_shadowops_env.py
    python -m py_compile backend-ml/main.py
    python -m py_compile backend-ml/model_action_parser.py
    python -m py_compile backend-ml/training/reward_rubric.py
    python -m py_compile backend-ml/training/build_action_only_sft_dataset.py
    python -m py_compile backend-ml/training/evaluate_checkpoints.py
    python -m py_compile backend-ml/training/model_policy_gate.py
    python -m py_compile backend-ml/training/submission_values.py
    python -m py_compile backend-ml/training/check_submission_artifacts.py
    python -m py_compile backend-ml/training/generate_reward_curves.py
    python -m py_compile backend-ml/training/compare_training_runs.py
    python -m py_compile backend-ml/training/run_hidden_eval.py
    python -m py_compile backend-ml/training/analyze_oracle_gap.py
    python -m py_compile backend-ml/training/generate_reward_breakdown_report.py
    python -m py_compile backend-ml/training/generate_submission_plots.py
    python -m py_compile backend-ml/training/merge_real_training_results.py
    python -m py_compile backend-ml/training/openenv_eval.py
    python -m py_compile backend-ml/training/generate_report_artifacts.py
    python -m py_compile backend-ml/training/train_qwen3_grpo.py

    Push-Location backend-ml
    try {
        python training/build_action_only_sft_dataset.py
        python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
        python training/train_qwen3_grpo.py --reward-diagnostics --skip-model-load
        python training/run_hidden_eval.py
        python training/analyze_oracle_gap.py
        python training/model_policy_gate.py --skip-model-load
        python training/generate_reward_curves.py
        python training/compare_training_runs.py --output-dir training/plots
        python training/check_submission_artifacts.py
        if (Test-Path training/generate_replay_report.py) {
            python training/generate_replay_report.py
        }
        if (Test-Path training/generate_judge_report.py) {
            python training/generate_judge_report.py
        }
        if (Test-Path training/latency_determinism_check.py) {
            python training/latency_determinism_check.py
        }
        if (Test-Path training/generate_report_artifacts.py) {
            python training/generate_report_artifacts.py
        }
        if (Test-Path training/generate_submission_plots.py) {
            python training/generate_submission_plots.py
        }
        python demo_judge.py
        python -m pytest tests -q
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
