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
    python -m py_compile backend-ml/main.py
    python -m py_compile backend-ml/training/train_qwen3_grpo.py

    Push-Location backend-ml
    try {
        python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
        python training/train_qwen3_grpo.py --reward-diagnostics --skip-model-load
        if (Test-Path training/generate_replay_report.py) {
            python training/generate_replay_report.py
        }
        if (Test-Path training/generate_judge_report.py) {
            python training/generate_judge_report.py
        }
        if (Test-Path training/latency_determinism_check.py) {
            python training/latency_determinism_check.py
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
