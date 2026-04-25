"""
shadowops_cli.py — ShadowOps Rich CLI Live Dashboard
=====================================================
Replaces the frontend entirely. Live terminal dashboard with:
  - Per-domain health bars + quarantine status
  - Supervisor decision feed (ALLOW/BLOCK/FORK/QUARANTINE with colour)
  - Active fork tracker (shadow state delta)
  - Quarantine hold countdown + SIEM context signals
  - Reward curve sparkline (rolling 20 episodes)
  - Incident report panel (latest CRITICAL)
  - Action distribution histogram

Run:
  python shadowops_cli.py                    # mock mode (no GPU)
  python shadowops_cli.py --speed 0.3        # slower (0.3s per step)
  python shadowops_cli.py --episodes 50      # run 50 episodes
  python shadowops_cli.py --malicious 0.7    # 70% malicious rate
"""

import argparse
import time
import random
import sys
import os
from collections import deque, defaultdict

# ── Rich imports ──────────────────────────────────────────────
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn
from rich.columns import Columns
from rich import box

# ── ShadowOps env ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────
# Minimal self-contained env shim so the CLI runs standalone
# even without the full backend installed.
# If shadowops_env.py is present it is imported instead.
# ─────────────────────────────────────────────────────────────
try:
    from shadowops_env import (
        UniversalShadowEnv,
        ACTIONS,
        R,
        DOMAINS,
        extract_features,
        compute_ambiguity,
        build_llama_prompt,
    )
    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False
    DOMAINS = ["SOC", "GITHUB", "AWS"]
    ACTIONS = {0: "ALLOW", 1: "BLOCK", 2: "FORK", 3: "QUARANTINE"}
    R = {}
    def extract_features(*a, **kw): return [0.1]*16
    def compute_ambiguity(v): return 0.5
    def build_llama_prompt(*a, **kw): return ""

try:
    from training.shadowops_training_common import build_q_aware_decision, q_aware_demo_policy_action
    _Q_AWARE_POLICY_AVAILABLE = True
except Exception:
    build_q_aware_decision = None
    q_aware_demo_policy_action = None
    _Q_AWARE_POLICY_AVAILABLE = False

DEMO_POLICY_NAME = "q_aware_demo_policy"

# ─────────────────────────────────────────────────────────────
# Colour map
# ─────────────────────────────────────────────────────────────
ACTION_STYLE = {
    "ALLOW":      "bold green",
    "BLOCK":      "bold red",
    "FORK":       "bold magenta",
    "QUARANTINE": "bold yellow",
}
DOMAIN_ICON = {"SOC": "🛡 ", "GITHUB": "🐙", "AWS": "☁️ "}
SEVERITY_STYLE = {"CRITICAL": "bold red", "HIGH": "yellow", "MEDIUM": "cyan"}

# ─────────────────────────────────────────────────────────────
# Demo supervisor
# ─────────────────────────────────────────────────────────────
def threshold_heuristic_supervisor(risk_vec: list, ambiguity: float) -> str:
    risk = (risk_vec[0] * 0.35 + risk_vec[1] * 0.25 +
            risk_vec[3] * 0.20 + risk_vec[6] * 0.20)
    if risk > 0.65:   return "FORK"
    if ambiguity > 0.40: return "QUARANTINE"
    if risk > 0.35:   return "BLOCK"
    return "ALLOW"


def mock_supervisor(
    risk_vec: list,
    ambiguity: float,
    domain: str = "SOC",
    intent: str = "",
    raw_payload: str = "",
) -> str:
    if _Q_AWARE_POLICY_AVAILABLE and q_aware_demo_policy_action is not None:
        try:
            return q_aware_demo_policy_action(domain, intent, raw_payload, risk_vec)
        except Exception:
            pass
    return threshold_heuristic_supervisor(risk_vec, ambiguity)


def mock_supervisor_details(
    risk_vec: list,
    ambiguity: float,
    domain: str = "SOC",
    intent: str = "",
    raw_payload: str = "",
) -> dict:
    if _Q_AWARE_POLICY_AVAILABLE and build_q_aware_decision is not None:
        try:
            return build_q_aware_decision(domain, intent, raw_payload, risk_vec)
        except Exception:
            pass
    decision = threshold_heuristic_supervisor(risk_vec, ambiguity)
    risk = risk_vec[0] * 0.35 + risk_vec[1] * 0.25 + risk_vec[3] * 0.20 + risk_vec[6] * 0.20
    return {
        "decision": decision,
        "confidence": 1.0 - min(1.0, ambiguity * 0.5),
        "uncertainty": ambiguity,
        "cumulative_risk_score": risk,
        "missing_evidence": [],
        "safe_outcome": "Threshold fallback decision.",
    }

# ─────────────────────────────────────────────────────────────
# Sparkline helper
# ─────────────────────────────────────────────────────────────
SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: list, width: int = 20) -> str:
    if not values:
        return "─" * width
    tail   = list(values)[-width:]
    lo, hi = min(tail), max(tail)
    span   = hi - lo or 1
    return "".join(
        SPARK_CHARS[int((v - lo) / span * (len(SPARK_CHARS) - 1))]
        for v in tail
    )

# ─────────────────────────────────────────────────────────────
# Dashboard state
# ─────────────────────────────────────────────────────────────
class DashboardState:
    def __init__(self, max_feed: int = 14):
        self.feed: deque      = deque(maxlen=max_feed)  # decision log rows
        self.ep_rewards: list = []
        self.step_rewards: deque = deque(maxlen=60)
        self.action_counts    = defaultdict(int)
        self.health           = {"SOC": 100, "GITHUB": 100, "AWS": 100}
        self.quarantine_hold  = {"SOC": 0, "GITHUB": 0, "AWS": 0}
        self.fork_active      = {"SOC": False, "GITHUB": False, "AWS": False}
        self.latest_incident  = None
        self.latest_signals: list = []
        self.episode_num      = 0
        self.step_num         = 0
        self.total_reward     = 0.0
        self.ep_reward        = 0.0
        self.correct          = 0
        self.total_decisions  = 0
        self.breach_count     = 0
        self.fork_count       = 0
        self.quarantine_count = 0

# ─────────────────────────────────────────────────────────────
# Layout builders
# ─────────────────────────────────────────────────────────────

def build_header(state: DashboardState) -> Panel:
    acc = (state.correct / state.total_decisions * 100) if state.total_decisions else 0
    t = Text()
    t.append("⚡ SHADOWOPS ", style="bold bright_white")
    t.append("SUPERVISOR LIVE DASHBOARD  ", style="bold cyan")
    t.append(f" Ep {state.episode_num:>4}  ", style="dim")
    t.append(f"Step {state.step_num:>6}  ", style="dim")
    t.append(f"Acc {acc:5.1f}%  ", style="bold green" if acc >= 70 else "yellow")
    t.append(f"Reward ∑ {state.total_reward:+8.1f}  ", style="bold cyan")
    t.append(f"Breaches {state.breach_count}  ", style="bold red")
    t.append(f"Forks {state.fork_count}  ", style="bold magenta")
    t.append(f"Quarantines {state.quarantine_count}", style="bold yellow")
    return Panel(t, border_style="bright_black", padding=(0, 1))


def build_health_panel(state: DashboardState) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Domain", width=10)
    table.add_column("Health", width=22)
    table.add_column("HP",     width=5)
    table.add_column("Status", width=18)

    for d in DOMAINS:
        hp      = state.health[d]
        hp_col  = "green" if hp >= 70 else "yellow" if hp >= 40 else "red"
        bar_len = int(hp / 5)  # out of 20 chars
        bar     = f"[{hp_col}]{'█' * bar_len}{'░' * (20 - bar_len)}[/{hp_col}]"
        q_hold  = state.quarantine_hold[d]
        fork_on = state.fork_active[d]

        if fork_on:
            status = "[bold magenta]⫸ FORK ACTIVE[/bold magenta]"
        elif q_hold > 0:
            status = f"[bold yellow]⏳ HOLD {q_hold} steps[/bold yellow]"
        else:
            status = "[dim green]● OPERATIONAL[/dim green]"

        table.add_row(
            f"{DOMAIN_ICON[d]} [bold]{d}[/bold]",
            bar,
            f"[{hp_col}]{hp:>3}[/{hp_col}]",
            status,
        )

    return Panel(table, title="[bold]DOMAIN HEALTH[/bold]", border_style="blue", padding=(0, 1))


def build_decision_feed(state: DashboardState) -> Panel:
    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1))
    table.add_column("#",       width=5,  style="dim")
    table.add_column("Domain",  width=8)
    table.add_column("Intent",  width=20)
    table.add_column("Action",  width=12)
    table.add_column("Conf",    width=6)
    table.add_column("Unc",     width=6)
    table.add_column("CumRisk", width=7)
    table.add_column("Missing", width=20)
    table.add_column("Safe Outcome", width=28)
    table.add_column("Mal?",    width=5)
    table.add_column("Reward",  width=8)
    table.add_column("Outcome", width=22)

    for row in reversed(list(state.feed)):
        action_str = row["action"]
        mal_str    = "[red]✗[/red]" if row["is_malicious"] else "[green]✓[/green]"
        reward_str = (f"[green]+{row['reward']:.0f}[/green]" if row["reward"] >= 0
                      else f"[red]{row['reward']:.0f}[/red]")
        outcome    = row.get("outcome", "")
        out_col    = ("red" if "BREACH" in outcome or "WRONG" in outcome
                      else "magenta" if "FORK" in outcome
                      else "yellow" if "QUARANTINE" in outcome
                      else "green")
        table.add_row(
            str(row["step"]),
            f"[bold]{row['domain']}[/bold]",
            row["intent"][:20],
            f"[{ACTION_STYLE[action_str]}]{action_str}[/{ACTION_STYLE[action_str]}]",
            f"{row.get('confidence', 0.0):.2f}",
            f"{row.get('uncertainty', 0.0):.2f}",
            f"{row.get('cumulative_risk_score', 0.0):.2f}",
            ", ".join(row.get("missing_evidence", []))[:20],
            row.get("safe_outcome", "")[:28],
            mal_str,
            reward_str,
            f"[{out_col}]{outcome[:22]}[/{out_col}]",
        )

    return Panel(table, title="[bold]DECISION FEED[/bold]", border_style="cyan", padding=(0, 1))


def build_reward_panel(state: DashboardState) -> Panel:
    spark = sparkline(list(state.step_rewards), width=40)
    ep_spark = sparkline(state.ep_rewards[-30:], width=40) if state.ep_rewards else "─" * 40

    t = Text()
    t.append("Step rewards  ", style="dim")
    t.append(spark, style="cyan")
    t.append(f"  cur {(list(state.step_rewards)[-1] if state.step_rewards else 0):+.0f}\n", style="bold")
    t.append("Episode Σ     ", style="dim")
    t.append(ep_spark, style="green")
    ep_mean = (sum(state.ep_rewards[-20:]) / len(state.ep_rewards[-20:])
               if state.ep_rewards else 0)
    t.append(f"  μ {ep_mean:+.1f}", style="bold")

    return Panel(t, title="[bold]REWARD CURVES[/bold]", border_style="green", padding=(0, 1))


def build_action_dist(state: DashboardState) -> Panel:
    total = sum(state.action_counts.values()) or 1
    t = Text()
    for act in ["ALLOW", "BLOCK", "FORK", "QUARANTINE"]:
        cnt  = state.action_counts[act]
        pct  = cnt / total
        bar  = "█" * int(pct * 20)
        pad  = "░" * (20 - len(bar))
        style = ACTION_STYLE[act]
        t.append(f"{act:<12}", style=style)
        t.append(bar, style=style)
        t.append(pad, style="dim")
        t.append(f" {cnt:4d} ({pct:.0%})\n", style="dim")

    return Panel(t, title="[bold]ACTION DISTRIBUTION[/bold]", border_style="yellow", padding=(0, 1))


def build_quarantine_panel(state: DashboardState) -> Panel:
    if not state.latest_signals:
        t = Text("No active quarantine hold.\n\n[dim]SIEM signals appear here during QUARANTINE[/dim]",
                 style="dim")
    else:
        t = Text()
        for sig in state.latest_signals[-4:]:
            conf = sig.get("signal_confidence", 0)
            conf_col = "green" if conf > 0.6 else "yellow"
            t.append(f"Hold step {sig.get('hold_step','?')}  ", style="bold yellow")
            t.append(f"steps left: {sig.get('steps_remaining','?')}\n", style="dim")
            t.append(sig.get("context_signal", ""), style="cyan")
            t.append(f"\n  confidence: [{conf_col}]{conf:.0%}[/{conf_col}]\n\n")

    return Panel(t, title="[bold yellow]⏳ QUARANTINE HOLD / SIEM SIGNALS[/bold yellow]",
                 border_style="yellow", padding=(0, 1))


def build_incident_panel(state: DashboardState) -> Panel:
    if not state.latest_incident:
        t = Text("[dim]No incidents yet. Incidents appear when FORK is executed.[/dim]")
    else:
        inc   = state.latest_incident
        sev   = inc.get("severity", "MEDIUM")
        mitre = inc.get("mitre", {})
        blast = inc.get("blast_radius", {})
        t = Text()
        t.append(f"[{SEVERITY_STYLE[sev]}]{sev}[/{SEVERITY_STYLE[sev]}]")
        t.append(f"  {inc.get('report_id','')}  {inc.get('timestamp','')[:19]}\n")
        t.append(f"Domain: [bold]{inc.get('domain','')}[/bold]  Intent: {inc.get('intent','')}\n")
        t.append(f"MITRE: [cyan]{mitre.get('tactic','')}[/cyan] → {mitre.get('technique_id','')} "
                 f"{mitre.get('technique_name','')}\n")
        t.append(f"Confidence: [yellow]{inc.get('confidence',0):.0%}[/yellow]\n")
        if blast:
            for k, v in list(blast.items())[:3]:
                t.append(f"  {k}: [dim]{v}[/dim]\n")
        t.append(f"[dim]Rec: {inc.get('recommendation','')[:80]}[/dim]\n")

    return Panel(t, title="[bold red]🚨 LATEST INCIDENT REPORT[/bold red]",
                 border_style="red", padding=(0, 1))


def build_layout(state: DashboardState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="bottom", size=10),
    )
    layout["body"].split_row(
        Layout(name="left",  ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(name="feed",    ratio=3),
        Layout(name="health",  ratio=2),
    )
    layout["right"].split_column(
        Layout(name="rewards",    ratio=2),
        Layout(name="dist",       ratio=2),
        Layout(name="quarantine", ratio=3),
    )
    layout["bottom"].split_row(
        Layout(name="incident", ratio=2),
    )

    layout["header"].update(build_header(state))
    layout["feed"].update(build_decision_feed(state))
    layout["health"].update(build_health_panel(state))
    layout["rewards"].update(build_reward_panel(state))
    layout["dist"].update(build_action_dist(state))
    layout["quarantine"].update(build_quarantine_panel(state))
    layout["incident"].update(build_incident_panel(state))

    return layout

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def run_dashboard(n_episodes: int, speed: float, malicious_rate: float, seed: int):
    console = Console()
    state   = DashboardState()

    if not _ENV_AVAILABLE:
        console.print("[bold red]shadowops_env.py not found — running in DEMO mode with synthetic data[/bold red]")
        time.sleep(1)
        _run_demo_mode(n_episodes, speed, state, console)
        return

    env = UniversalShadowEnv(
        malicious_rate=malicious_rate,
        episode_max_length=20,
        mode="training",
        seed=seed,
    )

    with Live(build_layout(state), console=console, refresh_per_second=8,
              screen=True) as live:
        for ep in range(n_episodes):
            obs_text, obs_vec = env.reset()
            done     = False
            ep_r     = 0.0
            state.episode_num = ep + 1

            while not done:
                # ── Decide ────────────────────────────────
                scenario = getattr(env, "_current_scenario", {}) or {}
                risk_vec  = obs_vec[:16]
                ambiguity = compute_ambiguity(risk_vec)
                decision_details = mock_supervisor_details(
                    risk_vec,
                    ambiguity,
                    scenario.get("domain", "SOC"),
                    scenario.get("intent", ""),
                    scenario.get("raw_payload", ""),
                )
                action_str = decision_details["decision"]
                action_int = {"ALLOW": 0, "BLOCK": 1, "FORK": 2, "QUARANTINE": 3}[action_str]

                obs_text, obs_vec, reward, done, info = env.step(action_int)

                # ── Update state ──────────────────────────
                ep_r              += reward
                state.total_reward += reward
                state.step_num    += 1
                state.step_rewards.append(reward)
                state.action_counts[action_str] += 1
                state.health = dict(info["health"])

                is_mal = info["is_malicious"]
                if ((is_mal  and action_str in ("BLOCK", "FORK", "QUARANTINE")) or
                    (not is_mal and action_str == "ALLOW")):
                    state.correct += 1
                state.total_decisions += 1

                if action_str == "FORK":
                    state.fork_count  += 1
                    state.fork_active[info["domain"]] = True
                if action_str == "QUARANTINE":
                    state.quarantine_count += 1
                if info.get("outcome") == "BREACH":
                    state.breach_count += 1

                # ── Quarantine hold display ───────────────
                q_info = info.get("quarantine_hold", {})
                for d in DOMAINS:
                    hold = q_info.get(d, {})
                    state.quarantine_hold[d] = hold.get("steps_remaining", 0)
                    if hold.get("context_signals"):
                        state.latest_signals = hold["context_signals"]

                # ── Incident reports ──────────────────────
                if env.incident_reports:
                    state.latest_incident = env.incident_reports[-1]

                # ── Feed row ──────────────────────────────
                state.feed.append({
                    "step":         state.step_num,
                    "domain":       info["domain"],
                    "intent":       info["intent"],
                    "action":       action_str,
                    "confidence":   decision_details.get("confidence", 0.0),
                    "uncertainty":  decision_details.get("uncertainty", 0.0),
                    "cumulative_risk_score": decision_details.get("cumulative_risk_score", 0.0),
                    "missing_evidence": decision_details.get("missing_evidence", []),
                    "safe_outcome": decision_details.get("safe_outcome", ""),
                    "is_malicious": is_mal,
                    "reward":       reward,
                    "outcome":      info.get("outcome", ""),
                })

                live.update(build_layout(state))
                time.sleep(speed)

            # Episode done
            state.ep_rewards.append(ep_r)
            state.ep_reward = ep_r
            # Reset fork active flags between episodes
            for d in DOMAINS:
                state.fork_active[d] = False

        # Final pause so user can read results
        live.update(build_layout(state))
        time.sleep(5)

    console.print("\n[bold green]✓ Training run complete![/bold green]")
    ep_mean = sum(state.ep_rewards[-20:]) / max(len(state.ep_rewards[-20:]), 1)
    acc = state.correct / max(state.total_decisions, 1) * 100
    console.print(f"  Episodes:        {n_episodes}")
    console.print(f"  Mean reward:     {ep_mean:+.1f}")
    console.print(f"  Accuracy:        {acc:.1f}%")
    console.print(f"  Total forks:     {state.fork_count}")
    console.print(f"  Total breaches:  {state.breach_count}")
    console.print(f"  Quarantines:     {state.quarantine_count}")


# ─────────────────────────────────────────────────────────────
# Demo mode — fully synthetic, no env required
# ─────────────────────────────────────────────────────────────
import math as _math

def _demo_step(rng, state, step_n, ep_n):
    """Generate one synthetic step and push to state."""
    domain  = rng.choice(["SOC", "GITHUB", "AWS"])
    intents = ["UPDATE_FIREWALL", "MERGE_PR", "MODIFY_VPC", "UPDATE_IAM",
               "PUSH_CODE", "ADD_WEBHOOK", "ISOLATE_SERVER"]
    intent  = rng.choice(intents)
    is_mal  = rng.random() < 0.5
    action  = rng.choices(
        ["ALLOW", "BLOCK", "FORK", "QUARANTINE"],
        weights=[30, 25, 25, 20], k=1)[0]

    # synthetic reward
    r_table = {
        ("FORK",  True):  15, ("FORK",  False): -8,
        ("BLOCK", True):  10, ("BLOCK", False): -6,
        ("ALLOW", False):  9, ("ALLOW", True): -20,
        ("QUARANTINE", True): 0, ("QUARANTINE", False): 0,
    }
    reward  = r_table.get((action, is_mal), 0) + rng.uniform(-3, 3)

    outcomes = {
        ("FORK",  True): "FORKED_ATTACK",  ("FORK",  False): "UNNECESSARY_FORK",
        ("BLOCK", True): "BLOCKED_ATTACK", ("BLOCK", False): "FALSE_POSITIVE",
        ("ALLOW", False): "ALLOWED_OK",    ("ALLOW", True):  "BREACH",
        ("QUARANTINE", True): "QUARANTINE_INITIATED",
        ("QUARANTINE", False): "QUARANTINE_INITIATED",
    }
    outcome = outcomes.get((action, is_mal), "UNKNOWN")

    # health drift
    dh = {"SOC": 0, "GITHUB": 0, "AWS": 0}
    if is_mal and action == "ALLOW":
        dh[domain] = -25
        state.breach_count += 1
    elif action == "FORK" and is_mal:
        state.fork_count += 1
    state.health[domain] = max(0, min(100, state.health[domain] + dh[domain]))

    if action == "QUARANTINE":
        state.quarantine_count += 1
        state.quarantine_hold[domain] = rng.randint(1, 3)
        state.latest_signals = [
            {"hold_step": 1, "steps_remaining": 2,
             "context_signal": "[SIEM] Cross-check in progress — no corroborating signals",
             "signal_confidence": 0.33}
        ]
    else:
        state.quarantine_hold[domain] = 0

    # synthetic incident
    if action == "FORK" and is_mal:
        state.latest_incident = {
            "report_id": f"INC-{rng.randint(10000,99999)}",
            "timestamp": "2026-04-21T12:00:00",
            "domain": domain, "intent": intent,
            "severity": "CRITICAL", "confidence": round(rng.uniform(0.7, 0.95), 2),
            "mitre": {"tactic": "Defense Evasion", "technique_id": "T1027",
                      "technique_name": "Obfuscated Files", "confidence": 0.8},
            "blast_radius": {"exposed_ports": ["port_22"], "compromised_iam": []},
            "recommendation": "Revoke rogue roles. Audit S3 ACLs.",
        }

    state.action_counts[action] += 1
    state.step_rewards.append(reward)
    state.total_reward += reward
    state.step_num = step_n
    state.episode_num = ep_n

    if ((is_mal and action in ("BLOCK", "FORK", "QUARANTINE")) or
        (not is_mal and action == "ALLOW")):
        state.correct += 1
    state.total_decisions += 1

    state.feed.append({
        "step": step_n, "domain": domain, "intent": intent,
        "action": action, "is_malicious": is_mal,
        "confidence": 0.70,
        "uncertainty": 0.30,
        "cumulative_risk_score": 0.50 if is_mal else 0.20,
        "missing_evidence": [],
        "safe_outcome": "Synthetic demo outcome.",
        "reward": reward, "outcome": outcome,
    })
    return reward


def _run_demo_mode(n_episodes: int, speed: float, state: DashboardState,
                   console: Console):
    rng       = random.Random(42)
    step_num  = 0
    ep_len    = 20

    with Live(build_layout(state), console=console, refresh_per_second=8,
              screen=True) as live:
        for ep in range(n_episodes):
            ep_r = 0.0
            for _ in range(ep_len):
                step_num += 1
                ep_r += _demo_step(rng, state, step_num, ep + 1)
                live.update(build_layout(state))
                time.sleep(speed)

            state.ep_rewards.append(ep_r)
            # Gradually heal domains between episodes
            for d in DOMAINS:
                state.health[d] = min(100, state.health[d] + 5)

        live.update(build_layout(state))
        time.sleep(5)

    console.print("\n[bold green]✓ Demo complete (synthetic data)[/bold green]")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ShadowOps Rich CLI Dashboard")
    parser.add_argument("--episodes",  type=int,   default=30,   help="Number of episodes")
    parser.add_argument("--speed",     type=float, default=0.15, help="Seconds per step")
    parser.add_argument("--malicious", type=float, default=0.5,  help="Malicious rate 0-1")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    run_dashboard(
        n_episodes    = args.episodes,
        speed         = args.speed,
        malicious_rate = args.malicious,
        seed          = args.seed,
    )

if __name__ == "__main__":
    main()
