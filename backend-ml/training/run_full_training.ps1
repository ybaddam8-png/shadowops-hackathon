# Run this from backend-ml on a GPU machine.
#
# Default full-training commands:
#   python training\train_qwen3_sft.py --model-name unsloth/Qwen3-1.7B --sft-epochs 2 --batch-size 1 --grad-accum 8 --max-seq-len 256 --learning-rate 2e-4 --sft-output-dir training\checkpoints\qwen3_sft_adapter
#   python training\train_qwen3_grpo.py --model-name unsloth/Qwen3-1.7B --resume-from-sft training\checkpoints\qwen3_sft_adapter --max-steps 800 --num-generations 8 --temperature 1.0 --top-p 0.95 --top-k 50 --max-new-tokens 8 --batch-size 1 --grad-accum 4 --val-eval-eps 100 --eval-batch-size 4 --learning-rate 1e-5 --output-dir training\checkpoints\qwen3_sft_grpo_adapter
#
# Fallback if GPU memory is tight:
#   python training\train_qwen3_grpo.py --model-name unsloth/Qwen3-1.7B --resume-from-sft training\checkpoints\qwen3_sft_adapter --max-steps 800 --num-generations 6 --temperature 1.0 --top-p 0.95 --top-k 50 --max-new-tokens 8 --batch-size 1 --grad-accum 8 --val-eval-eps 50 --eval-batch-size 4 --learning-rate 1e-5 --output-dir training\checkpoints\qwen3_sft_grpo_adapter

$ErrorActionPreference = "Stop"

Write-Host "[1/7] Compile checks"
python -m py_compile training\train_qwen3_grpo.py
python -m py_compile training\train_qwen3_sft.py
python -m py_compile shadowops_env.py

Write-Host "[2/7] Smoke tests"
python training\train_qwen3_grpo.py --smoke-test --skip-model-load
python training\train_qwen3_sft.py --smoke-test --skip-model-load

Write-Host "[3/7] Dataset audit"
python training\train_qwen3_grpo.py --dry-run --skip-model-load

Write-Host "[4/7] SFT training"
python training\train_qwen3_sft.py --model-name unsloth/Qwen3-1.7B --sft-epochs 2 --batch-size 1 --grad-accum 8 --max-seq-len 256 --learning-rate 2e-4 --sft-output-dir training\checkpoints\qwen3_sft_adapter

Write-Host "[5/7] SFT validation"
# SFT validation is produced by the SFT script and written to training\sft_metrics.json.

Write-Host "[6/7] GRPO training from SFT adapter"
python training\train_qwen3_grpo.py --model-name unsloth/Qwen3-1.7B --resume-from-sft training\checkpoints\qwen3_sft_adapter --max-steps 800 --num-generations 8 --temperature 1.0 --top-p 0.95 --top-k 50 --max-new-tokens 8 --batch-size 1 --grad-accum 4 --val-eval-eps 100 --eval-batch-size 4 --learning-rate 1e-5 --output-dir training\checkpoints\qwen3_sft_grpo_adapter

Write-Host "[7/7] Final validation + report"
# The GRPO script writes:
#   training\qwen3_training_health_report.json
#   training\final_training_report.json
#   training\final_training_report.md
