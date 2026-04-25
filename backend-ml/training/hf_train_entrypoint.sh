#!/usr/bin/env bash
set -euo pipefail

echo "Python version:"
python --version

echo "CUDA availability:"
python - <<'PY'
try:
    import torch
except Exception as exc:
    print("CUDA: unavailable; torch import failed:", exc)
else:
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Installing training requirements..."
pip install -r requirements.txt

echo "Training script help:"
python training/train_qwen3_grpo.py --help

require_flag() {
    local flag="$1"
    if ! python training/train_qwen3_grpo.py --help | grep -q -- "$flag"; then
        echo "Required flag is not supported by training/train_qwen3_grpo.py: $flag" >&2
        exit 2
    fi
}

TRAIN_MODE="${TRAIN_MODE:-eval_only}"
echo "TRAIN_MODE=${TRAIN_MODE}"

mkdir -p training/cloud_runs

case "${TRAIN_MODE}" in
    eval_only)
        python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
        python demo_judge.py
        ;;
    sft_smoke)
        require_flag "--run-sft"
        require_flag "--skip-grpo"
        require_flag "--train-size"
        require_flag "--val-size"
        require_flag "--sft-epochs"
        require_flag "--sft-output-dir"
        require_flag "--max-steps"
        # train_qwen3_grpo.py has no SFT --max-steps flag; 50 samples and one epoch cap this smoke run.
        python training/train_qwen3_grpo.py \
            --run-sft \
            --skip-grpo \
            --train-size 50 \
            --val-size 50 \
            --val-eval-eps 50 \
            --sft-epochs 1 \
            --sft-learning-rate 2e-4 \
            --sft-output-dir training/cloud_runs/sft_smoke \
            --max-steps 50 \
            --max-seq-len 256 \
            --logging-steps 5
        ;;
    grpo_smoke)
        require_flag "--run-sft"
        require_flag "--max-steps"
        require_flag "--num-generations"
        require_flag "--batch-size"
        require_flag "--grad-accum"
        require_flag "--output-dir"
        python training/train_qwen3_grpo.py \
            --run-sft \
            --train-size 50 \
            --val-size 50 \
            --val-eval-eps 50 \
            --sft-epochs 1 \
            --sft-output-dir training/cloud_runs/grpo_smoke/sft_warm_start \
            --resume-from-sft training/cloud_runs/grpo_smoke/sft_warm_start \
            --output-dir training/cloud_runs/grpo_smoke \
            --max-steps 50 \
            --batch-size 1 \
            --grad-accum 1 \
            --num-generations 2 \
            --eval-steps 25 \
            --eval-batch-size 2 \
            --logging-steps 5 \
            --max-seq-len 256
        ;;
    *)
        echo "Unsupported TRAIN_MODE: ${TRAIN_MODE}" >&2
        echo "Supported modes: eval_only, sft_smoke, grpo_smoke" >&2
        exit 2
        ;;
esac
