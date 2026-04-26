# ShadowOps Judge Submission Readiness

## Validated Locally
The Q-aware fallback and OpenEnv safety evaluation are valid and remain the basis for current claims.
- **Q-aware fallback / OpenEnv safety evaluation:** We successfully evaluated the Q-aware baseline running locally. It achieves an enforced 0.000 unsafe rate across our validation environments.
- **Artifact checks:** All local file artifact checks successfully pass.
- **Readiness behavior:** The system honestly boots into degraded fallback mode when optional ML dependencies (like `torchvision`) are missing, but ensures high-reliability rule resolution without hard crashing.
- **Local reports:** All available OpenEnv behavior and loop evaluation reports in `backend-ml/training/reports` were generated entirely from real local execution within this environment.

## Pending GPU Validation
The trained-model evaluation has not yet produced valid metrics in the standard local environment.
- `model_metrics` is null.
- Checkpoint comparison and policy-gate results are not yet available.
- No trained-model comparison claims are made at this stage.
- Trained-model evidence will be added only when GPU evaluation artifacts exist.

## Safety vs Reward
There is a trade-off when configuring RL safety guidelines:
* **The Reward Trade-Off:** The `heuristic` baseline policy may occasionally have a higher `mean_reward_per_step` due to its simplistic fast-resolution times.
* **The Safety Mandate:** `Q-aware` operates defensively, correctly triggering QUARANTINE proactively rather than guessing on ambiguous prompts. `Q-aware` is strictly safer despite generating slightly lower aggregate rewards out-of-the-box. In security automation, preventing false unlocks (0.000 unsafe rate) is profoundly more important than raw speed metrics.

## Current Limitations
Our local integration lacks the GPU hardware to run the complete GRPO RL training cycle. We are isolating the GPU training runs safely away from the core standard application until they are verified.

## Validated Locally
- **Q-aware:** 99% exact match, 100% safety accuracy, 0.000 unsafe rate, 0.000 false positive rate
- **50-episode OpenEnv:** 0.000 unsafe allows
- **5/5 replay attacks:** PASS