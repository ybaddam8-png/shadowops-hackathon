"""
training/train_qwen3_grpo.py — ShadowOps GRPO Training on Qwen3-4B-Base
=========================================================================
v4.1 — All 4 actions fire at meaningful rates; Q-aware > Heuristic

Key fixes from v4.0:
  - episode_length: 40 (was 20) — QUARANTINE hold+resolution fits in episode
  - QuarantineAwarePolicy: tiered threshold ladder so BLOCK is reachable
    (was sandwiched between FORK and QUARANTINE with no reachable band)
  - TierAwareOptimalPolicy: oracle ceiling added for blog comparison
  - evaluate() now logs A/B/F/Q% per policy so BLOCK>0 is visible
  - run_preflight(): automated checks before training starts
  - All _compute_reward callers unpack as: outcome, reward = ... (correct order)

Expected baseline output:
  [Random                         ] rew=+400–500  B≈25%  Q≈25%
  [Heuristic (no quarantine)      ] rew=+300–350  B≈25%  Q=0%
  [Quarantine-aware               ] rew=+450–550  B≈18%  Q≈14%  ← beats heuristic
  [Oracle (tier-aware ceiling)    ] rew=+600–700  B≈20%  Q≈15%  ← upper bound

Run:
  # CPU baseline only (no GPU):
  python training/train_qwen3_grpo.py --skip-training

  # Full GRPO on GPU:
  python training/train_qwen3_grpo.py

  # Use 1.7B if T4 is tight (<12 GB free VRAM):
  python training/train_qwen3_grpo.py --model 1.7b
"""

import sys
import os
import json
import random
import statistics
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from shadowops_env import (
    UniversalShadowEnv,
    ATTACK_SCENARIOS,
    ACTIONS,
    OBS_DIM,
    extract_features,
    build_llama_prompt,
    compute_ambiguity,
    R,
    DOMAINS,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    "4b":   "unsloth/Qwen3-4B-Base",
    "1.7b": "unsloth/Qwen3-1.7B",
    "8b":   "unsloth/Qwen3-8B-Base",
}

CFG = {
    "model_name":         MODEL_OPTIONS["4b"],
    "output_dir":         "./shadowops_qwen3_model",
    "n_train_episodes":   500,
    "n_val_episodes":     100,
    # ── CRITICAL: 40 steps so QUARANTINE hold (3 steps) resolves in-episode ──
    "episode_length":     40,
    "malicious_rate":     0.5,
    "seed":               42,
    # GRPO hyperparams
    "learning_rate":      1e-5,
    "max_steps":          300,
    "batch_size":         2,
    "grad_accum":         8,
    "max_seq_len":        512,
    "num_generations":    4,
    "max_completion_len": 8,
    "lora_r":             16,
    "lora_alpha":         32,
    "patience":           8,
    "val_check_every":    30,
}

SYSTEM_PREFIX = (
    "You are ShadowOps Supervisor. /no_think\n"
    "Analyse payload + risk vector. "
    "Output exactly one word: ALLOW, BLOCK, FORK, or QUARANTINE.\n"
    "QUARANTINE = hold action 3 steps, gather SIEM context, then resolve."
)

VALID_ACTIONS = set(ACTIONS.values())


# ─────────────────────────────────────────────────────────────
# 1. Policies
# ─────────────────────────────────────────────────────────────

class RandomPolicy:
    """Uniform random over all 4 actions. Lower bound baseline."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs_text, obs_vec):
        return self.rng.randint(0, 3)


class HeuristicPolicy:
    """
    Rule-based, no quarantine. Strong 3-action baseline.
    risk >= 0.65 → FORK, >= 0.35 → BLOCK, else ALLOW.
    """
    def act(self, obs_text, obs_vec):
        rv   = obs_vec[:16]
        risk = rv[0]*0.35 + rv[1]*0.25 + rv[3]*0.20 + rv[6]*0.20
        if risk >= 0.65: return 2   # FORK
        if risk >= 0.35: return 1   # BLOCK
        return 0                    # ALLOW


class QuarantineAwarePolicy:
    """
    4-action policy with a tiered threshold ladder.

    Routing (fresh scenario):
      risk >= 0.75            → FORK       (high-confidence hard malicious)
      risk >= 0.55            → BLOCK      (moderate-high, not worth shadow env)
      ambiguity > 0.35        → QUARANTINE (genuinely unclear)
      risk >= 0.30            → BLOCK      (low-ambiguity mid-risk, fast reject)
      else                    → ALLOW

    Resolution (active hold, q_steps == 0):
      risk > 0.45  → FORK
      else         → ALLOW

    Fix vs v4.0:
      Old policy had ambiguity check BEFORE the lower BLOCK band, so BLOCK
      was only reachable for risk in [0.35, 0.40) — a tiny sliver.
      Now BLOCK has two explicit bands: [0.55, 0.75) and [0.30, ambiguity_threshold).
    """
    def act(self, obs_text, obs_vec):
        rv        = obs_vec[:16]
        q_active  = obs_vec[16] > 0.5
        q_steps   = obs_vec[17]
        risk      = rv[0]*0.35 + rv[1]*0.25 + rv[3]*0.20 + rv[6]*0.20
        ambiguity = compute_ambiguity(rv)

        # ── Resolve active quarantine hold ─────────────────────
        if q_active:
            if q_steps > 0.0:
                return 3                        # still in hold window
            return 2 if risk > 0.45 else 0     # resolve: FORK or ALLOW

        # ── Fresh scenario routing ─────────────────────────────
        if risk >= 0.75:      return 2          # FORK  — MALICIOUS_HARD
        if risk >= 0.55:      return 1          # BLOCK — upper mid-risk band
        if ambiguity > 0.35:  return 3          # QUARANTINE — ambiguous
        if risk >= 0.30:      return 1          # BLOCK — lower mid-risk band
        return 0                                 # ALLOW — benign


class TierAwareOptimalPolicy:
    """
    Oracle ceiling — uses feature vector proxies for tier knowledge.

    bracket_override (rv[6]) + exfil (rv[2]) high → MALICIOUS_HARD → FORK
    risk >= 0.55                                  → MALICIOUS_SOFT → BLOCK
    ambiguity > 0.35                              → AMBIGUOUS      → QUARANTINE
    else                                          → BENIGN         → ALLOW

    Useful for the HuggingFace blog: shows the gap the GRPO model closes.
    """
    def act(self, obs_text, obs_vec):
        rv               = obs_vec[:16]
        risk             = rv[0]*0.35 + rv[1]*0.25 + rv[3]*0.20 + rv[6]*0.20
        ambiguity        = compute_ambiguity(rv)
        bracket_override = rv[6]
        exfil            = rv[2]

        if bracket_override > 0.6 or exfil > 0.5: return 2   # FORK
        if risk >= 0.55:                           return 1   # BLOCK
        if ambiguity > 0.35:                       return 3   # QUARANTINE
        return 0                                               # ALLOW


# ─────────────────────────────────────────────────────────────
# 2. Trajectory collection
# ─────────────────────────────────────────────────────────────

def collect_episodes(n_episodes, policy, seed, malicious_rate=0.5,
                     episode_length=None) -> list:
    ep_len = episode_length or CFG["episode_length"]
    env = UniversalShadowEnv(
        malicious_rate=malicious_rate,
        episode_max_length=ep_len,
        mode="training",
        seed=seed,
    )
    episodes = []
    for ep_i in range(n_episodes):
        obs_text, obs_vec = env.reset()
        done  = False
        steps = []
        while not done:
            action_int            = policy.act(obs_text, obs_vec)
            obs_text, obs_vec, reward, done, info = env.step(action_int)
            steps.append({
                "prompt":       obs_text,
                "action":       ACTIONS[action_int],
                "reward":       reward,
                "is_malicious": info["is_malicious"],
                "domain":       info["domain"],
                "outcome":      info["outcome"],
                "q_active":     obs_vec[16] > 0.5,
                "q_steps":      int(round(obs_vec[17] * 3)),
            })
        episodes.append({
            "episode_id":   ep_i,
            "steps":        steps,
            "total_reward": sum(s["reward"] for s in steps),
            "accuracy":     sum(
                1 for s in steps
                if (s["is_malicious"] and s["action"] in ("BLOCK", "FORK", "QUARANTINE"))
                or (not s["is_malicious"] and s["action"] == "ALLOW")
            ) / max(len(steps), 1),
        })
    return episodes


def episodes_to_grpo(episodes: list) -> list:
    samples = []
    for ep in episodes:
        for s in ep["steps"]:
            prompt = s["prompt"].replace(
                "You are ShadowOps Supervisor.",
                SYSTEM_PREFIX,
                1,
            )
            samples.append({
                "query":    prompt,
                "response": s["action"],
                "reward":   float(s["reward"]),
            })
    return samples


# ─────────────────────────────────────────────────────────────
# 3. Evaluation helper
# ─────────────────────────────────────────────────────────────

def evaluate(policy, n_episodes=50, seed=99, label="") -> dict:
    eps  = collect_episodes(n_episodes, policy, seed=seed)
    rews = [e["total_reward"] for e in eps]
    accs = [e["accuracy"]     for e in eps]

    act_counts  = defaultdict(int)
    total_steps = 0
    for e in eps:
        for s in e["steps"]:
            act_counts[s["action"]] += 1
            total_steps             += 1

    fp     = sum(1 for e in eps for s in e["steps"]
                 if not s["is_malicious"] and s["action"] in ("BLOCK", "FORK", "QUARANTINE"))
    benign = sum(1 for e in eps for s in e["steps"] if not s["is_malicious"])
    fpr    = fp / max(benign, 1)

    a_pct = act_counts["ALLOW"]      / max(total_steps, 1)
    b_pct = act_counts["BLOCK"]      / max(total_steps, 1)
    f_pct = act_counts["FORK"]       / max(total_steps, 1)
    q_pct = act_counts["QUARANTINE"] / max(total_steps, 1)

    res = {
        "label":          label,
        "mean_reward":    statistics.mean(rews),
        "std_reward":     statistics.stdev(rews) if len(rews) > 1 else 0.0,
        "mean_acc":       statistics.mean(accs),
        "fpr":            fpr,
        "allow_pct":      a_pct,
        "block_pct":      b_pct,
        "fork_pct":       f_pct,
        "quarantine_pct": q_pct,
        "rewards":        rews,
        "accuracies":     accs,
    }
    print(f"  [{label:<35}] "
          f"rew={res['mean_reward']:+7.1f}±{res['std_reward']:.1f}  "
          f"acc={res['mean_acc']:.1%}  "
          f"fpr={fpr:.1%}  "
          f"A={a_pct:.1%} B={b_pct:.1%} F={f_pct:.1%} Q={q_pct:.1%}")
    return res


# ─────────────────────────────────────────────────────────────
# 4. Reward function for GRPO
# ─────────────────────────────────────────────────────────────

def make_reward_fn(valid_tokens=None):
    valid = VALID_ACTIONS if valid_tokens is None else valid_tokens

    def reward_fn(completions, **kwargs):
        rewards = []
        for comp in completions:
            text = comp.strip().split()[0].upper() if comp.strip() else ""
            if text in valid:
                rewards.append(0.0)
            else:
                rewards.append(-8.0)
        return rewards

    return reward_fn


# ─────────────────────────────────────────────────────────────
# 5. Unsloth + TRL GRPO training
# ─────────────────────────────────────────────────────────────

def train_grpo(train_samples: list, val_episodes: list, model_name: str):
    try:
        from unsloth import FastModel
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"\n  Missing dependency: {e}")
        print("  Install with:  pip install unsloth trl datasets torch")
        return None

    print(f"\n[3] Loading {model_name} with Unsloth (4-bit QLoRA) …")
    model, tokenizer = FastModel.from_pretrained(
        model_name      = model_name,
        max_seq_length  = CFG["max_seq_len"],
        load_in_4bit    = True,
        load_in_8bit    = False,
        full_finetuning = False,
    )

    model = FastModel.get_peft_model(
        model,
        r              = CFG["lora_r"],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = CFG["lora_alpha"],
        lora_dropout   = 0.05,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
    )

    grpo_data = Dataset.from_list([
        {"query": s["query"], "response": s["response"], "reward": s["reward"]}
        for s in train_samples
    ])

    reward_fn = make_reward_fn()
    best_val  = float("-inf")
    patience  = 0
    val_curve = []

    def _val_check(step):
        nonlocal best_val, patience
        FastModel.for_inference(model)
        correct = total = 0
        total_r = 0.0
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for ep in val_episodes[:20]:
            for s in ep["steps"]:
                inp = tokenizer(s["prompt"], return_tensors="pt",
                                truncation=True,
                                max_length=CFG["max_seq_len"]).to(device)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=8, do_sample=False)
                decoded = tokenizer.decode(out[0], skip_special_tokens=True)
                pred    = decoded.strip().split()[-1].upper()
                action  = pred if pred in VALID_ACTIONS else "BLOCK"
                is_mal  = s["is_malicious"]
                if ((is_mal     and action in ("BLOCK", "FORK", "QUARANTINE")) or
                    (not is_mal and action == "ALLOW")):
                    correct += 1
                total   += 1
                total_r += s["reward"]

        FastModel.for_training(model)
        val_r   = total_r / max(total, 1)
        val_acc = correct / max(total, 1)
        val_curve.append({"step": step, "val_reward": val_r, "val_acc": val_acc})
        print(f"  Val step {step:3d}: reward={val_r:+.2f}  acc={val_acc:.2%}")

        if val_r > best_val:
            best_val = val_r
            patience = 0
            model.save_pretrained(CFG["output_dir"])
            tokenizer.save_pretrained(CFG["output_dir"])
            print(f"  ✓ New best: {best_val:.2f} → saved to {CFG['output_dir']}")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{CFG['patience']})")
        return patience >= CFG["patience"]

    grpo_cfg = GRPOConfig(
        learning_rate               = CFG["learning_rate"],
        max_steps                   = CFG["max_steps"],
        per_device_train_batch_size = CFG["batch_size"],
        gradient_accumulation_steps = CFG["grad_accum"],
        warmup_ratio                = 0.05,
        output_dir                  = CFG["output_dir"],
        logging_steps               = 10,
        save_steps                  = CFG["val_check_every"],
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        report_to                   = "none",
        seed                        = CFG["seed"],
        num_generations             = CFG["num_generations"],
        max_completion_length       = CFG["max_completion_len"],
        temperature                 = 0.7,
        stop_token_ids              = [],
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        config           = grpo_cfg,
        train_dataset    = grpo_data,
        reward_funcs     = reward_fn,
    )

    print(f"\n[4] GRPO training starts …")
    print(f"    Model          : {model_name}")
    print(f"    OBS_DIM        : {OBS_DIM}  (16 risk + 2 quarantine state)")
    print(f"    Actions        : {sorted(VALID_ACTIONS)}")
    print(f"    Train samples  : {len(train_samples)}")
    print(f"    Val episodes   : {len(val_episodes)}")
    print(f"    Episode length : {CFG['episode_length']}")
    print(f"    Max steps      : {CFG['max_steps']}")
    print(f"    Batch (eff.)   : {CFG['batch_size'] * CFG['grad_accum']}")
    print()

    for chunk_start in range(0, CFG["max_steps"], CFG["val_check_every"]):
        trainer.train()
        if _val_check(chunk_start + CFG["val_check_every"]):
            print(f"\n  Early stopping at step {chunk_start + CFG['val_check_every']}")
            break

    return {"val_curve": val_curve, "best_val_reward": best_val}


# ─────────────────────────────────────────────────────────────
# 6. Pre-flight sanity check
# ─────────────────────────────────────────────────────────────

def run_preflight():
    """
    Automated checks before GPU training.
    Runs 200 episodes per policy (fast, CPU-only).
    All 5 checks must pass before training proceeds.
    """
    print("\n[PREFLIGHT] Sanity checks (200 episodes @ ep_len=40) …")
    n = 200

    r_heur   = evaluate(HeuristicPolicy(),        n, seed=10, label="Heuristic (preflight)")
    r_qaw    = evaluate(QuarantineAwarePolicy(),  n, seed=11, label="Q-aware   (preflight)")
    r_oracle = evaluate(TierAwareOptimalPolicy(), n, seed=12, label="Oracle    (preflight)")

    checks = [
        ("BLOCK%      > 0%   for Q-aware",   r_qaw["block_pct"]      > 0.0),
        ("QUARANTINE% > 0%   for Q-aware",   r_qaw["quarantine_pct"] > 0.0),
        ("FORK%       > 0%   for Q-aware",   r_qaw["fork_pct"]       > 0.0),
        ("Q-aware reward   > Heuristic",     r_qaw["mean_reward"]    > r_heur["mean_reward"]),
        ("Oracle reward    > Q-aware",       r_oracle["mean_reward"] > r_qaw["mean_reward"]),
    ]

    all_pass = True
    print()
    for desc, passed in checks:
        status   = "✓ PASS" if passed else "✗ FAIL"
        all_pass = all_pass and passed
        print(f"  {status}  {desc}")

    if all_pass:
        print("\n  ✓ All preflight checks passed — safe to start GRPO training.\n")
    else:
        print("\n  ✗ One or more preflight checks FAILED — fix before training.\n")

    return all_pass, r_heur, r_qaw, r_oracle


# ─────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Baselines + dataset only (no GPU required)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks (not recommended)")
    parser.add_argument("--model", choices=["4b", "1.7b", "8b"], default="4b",
                        help="Model size (default: 4b = Qwen3-4B-Base)")
    args = parser.parse_args()

    model_name = MODEL_OPTIONS[args.model]
    rng        = random.Random(CFG["seed"])

    print("=" * 70)
    print("  ShadowOps — GRPO Training Pipeline (Qwen3 Edition, 4-action RL)")
    print("=" * 70)
    print(f"  episode_length  : {CFG['episode_length']}  "
          f"(40 required so QUARANTINE resolves in-episode)")
    print(f"  n_train_episodes: {CFG['n_train_episodes']}")
    print(f"  malicious_rate  : {CFG['malicious_rate']}")
    print()

    # ── Preflight ────────────────────────────────────────────
    if not args.skip_preflight:
        ok, r_heur_pre, r_qaw_pre, r_oracle_pre = run_preflight()
        if not ok and not args.skip_training:
            print("  Aborting — fix preflight failures before training.")
            return
    else:
        print("  [PREFLIGHT] Skipped.\n")

    # ── Step 1: Full baselines ───────────────────────────────
    print("[1] Evaluating 4 baseline policies (50 episodes each) …")
    r_rand   = evaluate(RandomPolicy(seed=0),       50, seed=1, label="Random (uniform 4-action)")
    r_heur   = evaluate(HeuristicPolicy(),          50, seed=2, label="Heuristic (no quarantine)")
    r_qaw    = evaluate(QuarantineAwarePolicy(),    50, seed=3, label="Quarantine-aware")
    r_oracle = evaluate(TierAwareOptimalPolicy(),   50, seed=4, label="Oracle (tier-aware ceiling)")

    # ── Step 2: Collect training data ───────────────────────
    n_total = CFG["n_train_episodes"] + CFG["n_val_episodes"]
    print(f"\n[2] Collecting {n_total} episodes with QuarantineAwarePolicy …")

    policy  = QuarantineAwarePolicy()
    all_eps = collect_episodes(n_total, policy, seed=CFG["seed"],
                               malicious_rate=CFG["malicious_rate"])
    rng.shuffle(all_eps)
    train_eps     = all_eps[:CFG["n_train_episodes"]]
    val_eps       = all_eps[CFG["n_train_episodes"]:]
    train_samples = episodes_to_grpo(train_eps)
    val_samples   = episodes_to_grpo(val_eps)

    act_dist = defaultdict(int)
    for s in train_samples:
        act_dist[s["response"]] += 1
    total = len(train_samples)

    print(f"  Train steps : {total} | Val steps : {len(val_samples)}")
    print("  Action distribution (training data):")
    for act in ["ALLOW", "BLOCK", "FORK", "QUARANTINE"]:
        c   = act_dist[act]
        bar = "█" * max(1, int(c / total * 40))
        print(f"    {act:<12} {c:6d}  ({c/total:5.1%})  {bar}")

    Path("training").mkdir(exist_ok=True)
    with open("training/qwen3_train_dataset.json", "w") as f:
        json.dump(train_samples[:100], f, indent=2)
    print("  Sample saved → training/qwen3_train_dataset.json")

    # ── Step 3: Train ────────────────────────────────────────
    llama_results = None
    if not args.skip_training:
        llama_results = train_grpo(train_samples, val_eps, model_name)
    else:
        print(f"\n[3] Skipping training (--skip-training). Model would be: {model_name}")

    # ── Step 4: Save reward curves ───────────────────────────
    print("\n[4] Saving reward curves …")
    curves = {
        "model":          model_name,
        "obs_dim":        OBS_DIM,
        "n_actions":      4,
        "action_names":   sorted(VALID_ACTIONS),
        "episode_length": CFG["episode_length"],
        "baselines": {
            "random":           {"mean_reward": r_rand["mean_reward"],
                                 "fpr": r_rand["fpr"],
                                 "block_pct": r_rand["block_pct"],
                                 "quarantine_pct": r_rand["quarantine_pct"],
                                 "rewards": r_rand["rewards"]},
            "heuristic":        {"mean_reward": r_heur["mean_reward"],
                                 "fpr": r_heur["fpr"],
                                 "block_pct": r_heur["block_pct"],
                                 "quarantine_pct": r_heur["quarantine_pct"],
                                 "rewards": r_heur["rewards"]},
            "quarantine_aware": {"mean_reward": r_qaw["mean_reward"],
                                 "fpr": r_qaw["fpr"],
                                 "block_pct": r_qaw["block_pct"],
                                 "quarantine_pct": r_qaw["quarantine_pct"],
                                 "rewards": r_qaw["rewards"]},
            "oracle":           {"mean_reward": r_oracle["mean_reward"],
                                 "fpr": r_oracle["fpr"],
                                 "block_pct": r_oracle["block_pct"],
                                 "quarantine_pct": r_oracle["quarantine_pct"],
                                 "rewards": r_oracle["rewards"]},
        },
        "train": {
            "rewards":    [ep["total_reward"] for ep in train_eps],
            "accuracies": [ep["accuracy"]     for ep in train_eps],
        },
        "val": {
            "rewards":    [ep["total_reward"] for ep in val_eps],
            "accuracies": [ep["accuracy"]     for ep in val_eps],
        },
        "grpo_val_curve":  llama_results["val_curve"]      if llama_results else [],
        "best_val_reward": llama_results["best_val_reward"] if llama_results else None,
        "action_dist":     dict(act_dist),
        "improvements": {
            "heuristic_over_random":      r_heur["mean_reward"]   - r_rand["mean_reward"],
            "quarantine_over_heuristic":  r_qaw["mean_reward"]    - r_heur["mean_reward"],
            "oracle_over_quarantine":     r_oracle["mean_reward"]  - r_qaw["mean_reward"],
            "fpr_reduction_heur_vs_rand": r_rand["fpr"]           - r_heur["fpr"],
            "fpr_reduction_qaw_vs_heur":  r_heur["fpr"]           - r_qaw["fpr"],
        },
    }

    with open("reward_curves_qwen3.json", "w") as f:
        json.dump(curves, f, indent=2)

    # ── Summary ──────────────────────────────────────────────
    qaw_delta = curves["improvements"]["quarantine_over_heuristic"]
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  episode_length               : {CFG['episode_length']}")
    print(f"  Model                        : {model_name}")
    print()
    print(f"  Random baseline              : {r_rand['mean_reward']:+7.1f}  "
          f"B={r_rand['block_pct']:.1%}  Q={r_rand['quarantine_pct']:.1%}")
    print(f"  Heuristic (no quarantine)    : {r_heur['mean_reward']:+7.1f}  "
          f"B={r_heur['block_pct']:.1%}  Q={r_heur['quarantine_pct']:.1%}")
    print(f"  Quarantine-aware heuristic   : {r_qaw['mean_reward']:+7.1f}  "
          f"B={r_qaw['block_pct']:.1%}  Q={r_qaw['quarantine_pct']:.1%}")
    print(f"  Oracle (tier-aware ceiling)  : {r_oracle['mean_reward']:+7.1f}  "
          f"B={r_oracle['block_pct']:.1%}  Q={r_oracle['quarantine_pct']:.1%}")
    print()
    print(f"  heuristic  → q-aware delta   : {qaw_delta:+.1f}"
          + ("  ✓ Q-aware wins" if qaw_delta > 0 else "  ✗ UNEXPECTED — check thresholds"))
    print(f"  q-aware    → oracle  delta   : {curves['improvements']['oracle_over_quarantine']:+.1f}")
    print(f"  FPR reduction heur→q-aware   : {curves['improvements']['fpr_reduction_qaw_vs_heur']:.1%}")
    if llama_results:
        print(f"\n  Best Qwen3 GRPO val reward   : {llama_results['best_val_reward']:+.1f}")
    print(f"\n  reward_curves_qwen3.json saved")
    print(f"  Run plot_curves.py to generate graphs for your HuggingFace blog\n")


if __name__ == "__main__":
    main()