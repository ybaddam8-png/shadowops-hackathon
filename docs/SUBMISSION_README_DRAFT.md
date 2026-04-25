# ShadowOps OpenEnv Submission Draft

## What ShadowOps Is
ShadowOps is an OpenEnv-style reinforcement learning environment for autonomous cyber incident response. The agent receives a live security action, observes risk signals, missing evidence, memory from prior actions, and production impact, then chooses one supervisor action: `ALLOW`, `BLOCK`, `QUARANTINE`, or `FORK`.

The current demo is laptop-safe and model-free by default. The Q-aware supervisor establishes a strong pre-training baseline while the SFT/GRPO training pipeline remains ready for GPU execution.

## Why Autonomous Cyber Incident Response Is Hard
Security automation is not just classification. A wrong `ALLOW` can expose customer data, open SSH to the internet, or create an IAM admin path. A wrong `BLOCK` can stop an approved pentest, break-glass recovery, or critical deployment. The environment rewards agents that balance safety, evidence, risk memory, and false-positive reduction.

## What The Agent Learns
The agent learns to:

- Separate malicious actions from risky-looking but approved work.
- Ask for missing evidence before allowing production changes.
- Accumulate risk across multi-step chains.
- Prefer quarantine or human review when uncertainty is real.
- Avoid unsafe allow decisions on high-risk incidents.

## Environment Design
ShadowOps wraps the existing simulation in `backend-ml/openenv_shadowops_env.py`.

Interface:

```python
from openenv_shadowops_env import make_env

env = make_env(seed=42)
observation = env.reset()
observation, reward, done, info = env.step("QUARANTINE")
state = env.state()
env.close()
```

The environment includes deterministic seed support, multi-step incident trajectories, memory persistence support, reward breakdowns, evidence plans, and safe outcomes.

## Observation, Action, Reward
Observation includes:

- Cybersecurity incident prompt
- 16-dimensional risk vector
- Incident domain, intent, payload, and tier
- Health scores and quarantine status
- Memory context and accumulated risk
- Available actions

Action space:

- `ALLOW`: execute when risk is low or trusted evidence is strong
- `BLOCK`: reject clearly malicious or high-danger action
- `QUARANTINE`: isolate until missing evidence is provided
- `FORK`: route to human/shadow review when approval is required

Reward is decomposed into:

- Action correctness reward
- Safety penalty
- False-positive penalty
- Evidence-completeness reward
- Risk-chain handling reward
- Uncertainty handling reward

## Why This Is Not A Normal Dashboard
ShadowOps is an RL environment with an executable step loop, not a static UI. Agent actions mutate future state: production/shadow state, quarantine holds, memory, cumulative risk, evidence plans, and final outcomes.

## Baseline Results
Generated from local validation with no model loading:

| Policy | exact_match | safety_accuracy | unsafe_decision_rate | false_positive_rate | reward_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.360 | 0.800 | 0.200 | 0.163 | 0.083 |
| Heuristic | 0.520 | 0.920 | 0.080 | 0.000 | 1.146 |
| Q-aware | 0.990 | 1.000 | 0.000 | 0.000 | 1.899 |
| Oracle | 1.000 | 1.000 | 0.000 | 0.000 | 1.920 |

Q-aware supervisor establishes a strong pre-training baseline. GRPO training pipeline is ready. Real trained checkpoint metrics should be inserted after the HF run.

## Why This Is Hard To Game
The reward is not a single exact-match score. It includes safety, false-positive cost, evidence completeness, uncertainty handling, memory-chain handling, remediation quality, and policy compliance. Always blocking loses reward on approved workflows. Always forking loses reward when evidence is complete. Unsafe allow decisions receive the strongest penalty.

## Hidden Evaluation And False-Positive Traps
ShadowOps includes a hidden-style evaluation set with 150 scenarios: malicious/adversarial incidents, benign but scary-looking approved workflows, and ambiguous cases requiring human review. The set covers CI/CD, IAM, S3/public storage, Kubernetes, endpoint security, SaaS admin, data export, firewall policy, secrets management, and production deployment.

Run:

```powershell
cd backend-ml
python training/run_hidden_eval.py
```

The generated report is saved to `backend-ml/training/reports/hidden_eval_report.md`.

## Memory-Based Multi-Step Incident Response
ShadowOps evaluates chains where the final decision depends on prior actions, not only the last payload. Examples include firewall exposure followed by IAM admin creation and data export, CI workflow edits followed by secret access and production deploy, and public bucket exposure followed by external transfer and privilege escalation.

Run:

```powershell
cd backend-ml
python training/multistep_episode_eval.py
```

The generated report is saved to `backend-ml/training/reports/multistep_episode_report.md`.

## Training Commands
Laptop-safe validation:

```powershell
cd backend-ml
python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
python training/generate_submission_plots.py
python demo_judge.py
```

GPU training should be launched only from the documented Hugging Face scripts after eval-only validation passes. Do not claim improvement until `model_eval_report.json` proves it and the model-vs-policy gate passes.

## Real Training Evidence
Real training evidence will be added here after GPU run.

Use:

```powershell
cd backend-ml
python training/merge_real_training_results.py --input training/training_results_template.json
```

Replace template values with real checkpoint metrics first. Do not edit code to fake results.

## Hugging Face Space
HF Space link: TODO

## Demo Video And Slides
Demo video: TODO

Slides: TODO
