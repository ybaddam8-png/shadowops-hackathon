# ShadowOps OpenEnv 50-Episode Evaluation (README Snippet)

## Reproducible Evaluation Command

```bash
cd backend-ml
python training/openenv_eval.py --episodes 50 --baseline-policy heuristic --policy q_aware --episode-max-length 5 --seed 42
```

## What This Produces

- `training/reports/openenv_loop_eval.json`
- `training/reports/openenv_loop_eval.md`
- `training/reports/openenv_behavior_comparison.json`
- `training/reports/openenv_behavior_comparison.md`
- `training/reports/openenv_episode_rewards.png`

## Scope And Claims

- This is a **full 50-episode OpenEnv run** (not a 3-episode smoke check).
- The before/after file compares a baseline policy (`heuristic`) against the serving policy (`q_aware`) on validation scenarios.
- If trained checkpoint metrics are missing, the report labels that explicitly and does **not** claim model-training gains.
- Frontend fallback mode is intentionally preserved and visibly labeled when backend inference is unavailable.
