# ShadowOps GPU Training Handoff

This project is laptop-safe by default. The Q-aware policy is the production guardrail. A trained model is not trusted unless checkpoint evaluation proves it passes the safety gate.

## Order For GPU/Hugging Face

1. Pull the latest branch:

   ```bash
   git clone --branch shadowops-backend-agent-upgrade --single-branch https://github.com/badugujashwanth-create/shadowops-hackathon.git
   cd shadowops-hackathon
   ```

2. Run laptop-safe baseline/eval first:

   ```bash
   cd backend-ml
   pip install -r requirements-lite.txt
   python training/train_qwen3_grpo.py --evaluate-baselines-only --skip-model-load
   python training/train_qwen3_grpo.py --reward-diagnostics --skip-model-load
   python demo_judge.py
   ```

3. Run SFT smoke only after the local-safe checks pass.

4. Evaluate the SFT checkpoint:

   ```bash
   python training/train_qwen3_grpo.py --evaluate-model --model-path training/cloud_runs/sft_smoke --compare-against-policy
   ```

5. Run GRPO smoke only after SFT smoke and SFT evaluation work.

6. Evaluate the GRPO checkpoint:

   ```bash
   python training/train_qwen3_grpo.py --evaluate-model --model-path training/cloud_runs/grpo_smoke --compare-against-policy
   ```

7. Only then run longer GRPO.

8. Do not claim improvement unless `training/model_eval_report.json` proves it.

## Claim Rules

- Q-aware policy remains the production guardrail.
- Model outputs are not trusted until safety accuracy, unsafe decision rate, invalid output rate, and reward are measured.
- Loss decrease alone is not proof.
- Do not claim the model beats Q-aware unless `delta_vs_q_aware` proves it.
- Stop training if safety drops, unsafe decisions increase, invalid output rate increases, or the gate fails.

## Credit Safety

- Run eval-only first.
- Run SFT smoke before GRPO smoke.
- Run long GRPO only after checkpoint evaluation works.
- Do not run multiple GPU jobs at the same time.
- Cancel stuck GPU jobs immediately.
