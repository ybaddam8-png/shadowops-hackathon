# ShadowOps Judge Demo Checklist

## Before The Demo
- Confirm branch: `shadowops-backend-agent-upgrade`
- Confirm no secrets are present in docs or scripts.
- Confirm HF jobs are not running unless manually started for the final GPU run.

## Local Validation
```powershell
cd backend-ml
python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
python training/generate_submission_plots.py
python demo_judge.py
```

## Show During Demo
- Run `python demo_judge.py`
- Show benchmark table in `backend-ml/training/demo_benchmark_report.md`
- Show evidence plan from replay output
- Show memory/risk accumulation chain
- Show structured safe outcome
- Show reward plots from `backend-ml/training/plots/`
- Show hidden eval report from `backend-ml/training/reports/hidden_eval_report.md`
- Show multi-step memory report from `backend-ml/training/reports/multistep_episode_report.md`
- Show OpenEnv manifest: `openenv.yaml`
- Show schema contract: `backend-ml/schema_contract.json`
- Show HF Space link placeholder
- Show README links and training evidence placeholder

## Judge Talking Points
- This is an executable RL environment, not just a dashboard.
- Q-aware policy is a verified pre-training guardrail.
- Unsafe allow rate is zero for Q-aware baseline.
- GRPO pipeline is ready, but trained improvement is not claimed without checkpoint metrics.

## Why This Is Hard To Game
- Always BLOCK is penalized on approved pentest, break-glass, maintenance, and trusted automation cases.
- Always FORK is penalized when evidence is complete and a restricted allow is safe.
- Unsafe ALLOW is heavily penalized on exfiltration, public exposure, IAM admin creation, and secret access.
- Hidden false-positive traps verify the policy does not blindly block scary-looking safe actions.

## Hidden Evaluation And False-Positive Traps
- `python training/run_hidden_eval.py`
- Show Q-aware safety, unsafe decision rate, and per-domain results.
- Point out that benign traps include approved maintenance, public website buckets, known automation, and authorized pentests.

## Memory-Based Multi-Step Incident Response
- `python training/multistep_episode_eval.py`
- Show firewall -> IAM admin -> data export.
- Show CI workflow -> secret access -> production deploy.
- Show approved pentest chain avoids unnecessary blocking.
