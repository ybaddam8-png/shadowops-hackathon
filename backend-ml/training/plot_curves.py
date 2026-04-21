"""
training/plot_curves.py — ShadowOps Demo Reward Curve Generator
================================================================
Run AFTER train_shadowops.py has produced reward_curves.json

Output: reward_curves.png  (used in HuggingFace blog + pitch deck)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statistics

with open("reward_curves.json") as f:
    data = json.load(f)

summary = data["summary"]

# ── Figure setup ──────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#080810")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

CYAN   = "#00e5ff"
RED    = "#ff4d4d"
GREEN  = "#39ff14"
ORANGE = "#ff9900"
PURPLE = "#bf5fff"
GREY   = "#444455"

def style_ax(ax, title):
    ax.set_facecolor("#0d0d1f")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor(GREY)
    ax.grid(alpha=0.12, color="white", linestyle="--")

# ── Panel 1: Baseline vs Trained reward ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Episode Reward — Baseline vs Trained")

rand_r = data["random_rewards"]
heur_r = data["heuristic_rewards"]

ax1.plot(rand_r, color=RED,   linewidth=1.2, alpha=0.7, label="Random (untrained)")
ax1.plot(heur_r, color=CYAN,  linewidth=2.0,            label="Supervisor (trained)")
ax1.axhline(statistics.mean(rand_r), color=RED,  linestyle=":", alpha=0.5)
ax1.axhline(statistics.mean(heur_r), color=CYAN, linestyle=":", alpha=0.5)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.legend(fontsize=7, facecolor="#0d0d1f", labelcolor="white", framealpha=0.8)

# ── Panel 2: Smoothed training curve ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Training Convergence (Smoothed)")

smoothed = data["smoothed_train"]
raw      = data["train_rewards"]

ax2.plot(raw,      color=GREY,   linewidth=0.8, alpha=0.5, label="Raw")
ax2.plot(smoothed, color=ORANGE, linewidth=2.2, label="Smoothed (window=10)")
ax2.set_xlabel("Training Episode")
ax2.set_ylabel("Reward")
ax2.legend(fontsize=7, facecolor="#0d0d1f", labelcolor="white", framealpha=0.8)

# ── Panel 3: Train vs Val accuracy ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Train vs Val Accuracy (Overfit Check)")

tr_acc  = data["train_accuracies"]
val_acc = data["val_accuracies"]

ax3.plot(tr_acc,  color=GREEN,  linewidth=1.5, alpha=0.8, label="Train accuracy")
ax3.plot(val_acc, color=PURPLE, linewidth=1.5, alpha=0.8, label="Val accuracy")
ax3.axhline(0.5, color=RED, linestyle="--", linewidth=1, label="Random baseline (50%)")
ax3.set_ylim(0, 1.05)
ax3.set_xlabel("Episode")
ax3.set_ylabel("Accuracy")
ax3.legend(fontsize=7, facecolor="#0d0d1f", labelcolor="white", framealpha=0.8)

# ── Panel 4: Summary stats bar chart ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Performance Summary")

labels = ["Random\nBaseline", "Trained\nSupervisor"]
values = [summary["random_mean"], summary["heuristic_mean"]]
colors = [RED, CYAN]
bars   = ax4.bar(labels, values, color=colors, width=0.4, edgecolor=GREY)

for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}", ha="center", color="white", fontsize=9, fontweight="bold")

ax4.axhline(0, color=GREY, linewidth=0.8)
ax4.set_ylabel("Mean Episode Reward")

improvement = summary["improvement"]
ax4.text(0.5, 0.92, f"Improvement: +{improvement:.1f}",
         transform=ax4.transAxes, ha="center", color=GREEN,
         fontsize=10, fontweight="bold")

# ── Title ─────────────────────────────────────────────────────
fig.suptitle(
    "ShadowOps: Universal Cyberinfrastructure Firewall — Training Metrics",
    color="white", fontsize=12, fontweight="bold", y=0.98
)

plt.savefig("reward_curves.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → reward_curves.png")