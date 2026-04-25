# ShadowOps Model-vs-Policy Gate

Gate status: **PENDING_REAL_CHECKPOINT_EVAL**
Recommendation: **DEPLOY_QAWARE_POLICY**

Reason: No real checkpoint metrics are available. Q-aware remains the deployment guardrail.

## Q-aware Baseline

- exact_match: 0.99
- safety_accuracy: 1.0
- unsafe_decision_rate: 0.0
- reward_mean: 1.9203646

## Best Model Candidate

No real checkpoint candidate has been evaluated yet.

## Metric Sources

- demo_benchmark: `training\demo_benchmark_report.json` exists=True
- model_eval: `training\model_eval_report.json` exists=True
- checkpoint_comparison: `training\reports\checkpoint_comparison_report.json` exists=False
- hidden_eval: `training\reports\hidden_eval_report.json` exists=True
- multistep_eval: `training\reports\multistep_episode_report.json` exists=True