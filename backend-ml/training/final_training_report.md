# ShadowOps Training Readiness Report

Training-ready: yes

## Broken
- Action parsing accepted noisy outputs inconsistently and let <think> tags leak into metrics.
- Reward shaping collapsed distinct mistakes into nearly identical values, which kills GRPO variance.
- Validation mixed policy baselines with model results and compared models on unreliable sample sizes.
- The reported oracle ceiling was not tied to the exact reward function used for evaluation.
- Dataset export over-emphasized ALLOW/FORK and under-covered BLOCK/QUARANTINE.
- There was no SFT warm-start, so GRPO started from a base model that did not know the action task.
- Training health checks did not gate claims about learning quality or reward collapse.

## Fixed
- Added a shared parser, reward model, oracle evaluator, and dataset audit used by SFT, GRPO, baselines, and reports.
- Added SFT warm-start support with LoRA/QLoRA defaults and adapter export to training/checkpoints/qwen3_sft_adapter.
- Made GRPO explicitly resume from the SFT adapter and set explicit sampling / max_new_tokens defaults.
- Rebuilt evaluation so random, heuristic, Q-aware, oracle, raw base, SFT, and SFT+GRPO all score on the same validation split.
- Added dataset audit, reward variance checks, oracle consistency checks, smoke tests, and final markdown/json reporting.
- Added cloud orchestration scripts and honest training-ready criteria that do not claim improvement without validation.

## Criteria
- parse_tests_pass: pass
- dataset_audit_passes: pass
- reward_variance_passes: pass
- oracle_consistency_passes: pass
- evaluation_isolated: pass
- sft_script_exists: pass
- grpo_can_resume_from_sft: pass
- final_report_generation_works: pass
- smoke_test_passes: pass
- cloud_training_commands_documented: pass

## Dataset Audit
- Train samples: 80
- Val samples: 32
- Duplicate prompts: 0
- Train/val overlap: 0

## Metrics
- Random reward mean: 0.531
- Heuristic reward mean: 0.891
- Q-aware reward mean: 1.672
- Oracle reward mean: 2.000
- Raw model metrics available: no
- SFT metrics available: no
- GRPO metrics available: no

## Health Warnings
- None

## Cloud Commands
```bash
python training/train_qwen3_sft.py --model-name unsloth/Qwen3-1.7B --sft-epochs 2 --batch-size 1 --grad-accum 8 --max-seq-len 256 --learning-rate 2e-4 --sft-output-dir training/checkpoints/qwen3_sft_adapter
python training/train_qwen3_grpo.py --model-name unsloth/Qwen3-1.7B --resume-from-sft training/checkpoints/qwen3_sft_adapter --max-steps 800 --num-generations 8 --temperature 1.0 --top-p 0.95 --top-k 50 --max-new-tokens 8 --batch-size 1 --grad-accum 4 --val-eval-eps 100 --eval-batch-size 4 --learning-rate 1e-5 --output-dir training/checkpoints/qwen3_sft_grpo_adapter
# Fallback if GPU memory is tight:
python training/train_qwen3_grpo.py --model-name unsloth/Qwen3-1.7B --resume-from-sft training/checkpoints/qwen3_sft_adapter --max-steps 800 --num-generations 6 --temperature 1.0 --top-p 0.95 --top-k 50 --max-new-tokens 8 --batch-size 1 --grad-accum 8 --val-eval-eps 50 --eval-batch-size 4 --learning-rate 1e-5 --output-dir training/checkpoints/qwen3_sft_grpo_adapter
```

Best checkpoint path: training/checkpoints/qwen3_sft_grpo_adapter (pending cloud run)
