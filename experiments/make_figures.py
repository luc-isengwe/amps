"""Figures of Chapter 4, drawn from the AgentDojo result files in runs/.

Every number is read from the JSON files written by AgentDojo; nothing is
typed by hand. Run from the project folder (~/Documents/amps):

    python experiments/make_figures.py          (all figures)
    python experiments/make_figures.py --fig 1  (one figure)

Output in figures/: a .pdf and a .png 

Figures (added one by one):
    1  attack success rate and utility under attack, bar charts
    2  attack success rate of A, B and B+H under three attacks (file-based tasks)
    3  where the attacks succeed: file-based tasks against the other tasks
    4  sink calls blocked (or refused by the user) and executed by the three amps policies
    5  scalability: synthesis and verification time against the number of agents
    6  human confirmation: attentive and naive simulated user (file-based tasks)
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUNS = Path("runs")              # where AgentDojo wrote its results
OUT = Path("figures")            # where the figures are saved

# Configurations: label on the figure -> run folder (same order as the thesis).
CONFIGS = {
    "A\n(no defence)": "qwen3-4b-instruct_A",
    "Repeat\nprompt":  "qwen3-4b-instruct_repeat",
    "Delimiting":      "qwen3-4b-instruct_spotlight",
    "B":               "qwen3-4b-instruct_B",
    "B-field":         "qwen3-4b-instruct_Bfield",
    "B+H":             "qwen3-4b-instruct_BH",
}
NAIVE = "qwen3-4b-instruct_BH_naive"     # B+H answered by the naive user (file tasks only)
RESULTS = Path("results")                # where the confirmation logs are written

# Two colours only: blue = existing configurations, orange = amps policies.
BLUE, ORANGE = "#2a78d6", "#eb6834"
DARK_ORANGE = "#a8471f"          # second shade of orange, for B+H next to B
INK, GREY, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,          # embed fonts properly in the PDF
})


# ----------------------------------------------------------------------
# Reading the results
# ----------------------------------------------------------------------
def load_json(path):
    """Read one result file; a damaged file is treated as missing."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def attacked_results(folder, attack="important_instructions"):
    """Return {(user_task, injection_task): result} for the finished cases of a run."""
    results = {}
    pattern = f"{RUNS}/{folder}/*/banking/user_task_*/{attack}/injection_task_*.json"
    for path in glob.glob(pattern):
        r = load_json(path)
        if "security" in r and "utility" in r:          # skip unfinished cases
            ut = re.search(r"user_task_\d+", path).group()
            it = re.search(r"injection_task_\d+", path).group()
            results[(ut, it)] = r
    return results


def save(fig, name):
    """Save a figure as PDF and PNG in figures/."""
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"written: {OUT}/{name}.pdf and {OUT}/{name}.png")


def figure_title(fig, number, text):
    """Short title at the top of the figure.

    The figure number is not written in the image: LaTeX numbers the figure
    in its caption, and a second number in the image would contradict it.
    """
    fig.suptitle(text, fontsize=10.5, fontweight="bold", color=INK)


def legend_below(fig, entries, ncol=3):
    """Colour legend under the panels: entries are (colour, hatch, text)."""
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=t) for c, h, t in entries]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8)
    rows = -(-len(entries) // ncol)
    fig.tight_layout(rect=(0, 0.045 + 0.045 * rows, 1, 1))


def bars(ax, labels, values, colours, title, xlabel, ylabel, hatches=None):
    """One simple bar chart with the value written on top of each bar."""
    rects = ax.bar(range(len(labels)), values, color=colours, width=0.6, zorder=3,
                   hatch=hatches, edgecolor="white", linewidth=0)
    for r, v in zip(rects, values):
        ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_ylim(0, max(values) * 1.25 + 5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)


# ----------------------------------------------------------------------
# Figure 1: attack success rate (left) and utility under attack (right)
# ----------------------------------------------------------------------
def fig1():
    labels, asr, utility, colours = [], [], [], []
    print(f"{'configuration':16s} {'cases':>5s} {'ASR %':>7s} {'utility %':>10s}")
    for label, folder in CONFIGS.items():
        res = attacked_results(folder)
        if not res:
            print(f"  [skip] no result in {RUNS}/{folder}")
            continue
        n = len(res)
        a = 100 * sum(bool(r["security"]) for r in res.values()) / n   # AgentDojo: security = attack succeeded
        u = 100 * sum(bool(r["utility"]) for r in res.values()) / n
        labels.append(label)
        asr.append(a)
        utility.append(u)
        colours.append(DARK_ORANGE if label == "B+H" else ORANGE if label.startswith("B") else BLUE)
        print(f"{label.replace(chr(10), ' '):16s} {n:5d} {a:7.1f} {u:10.1f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 4.8))
    bars(left, labels, asr, colours, "(a) Security: lower is better",
         "Configuration", "Attack success rate (%)")
    bars(right, labels, utility, colours, "(b) Utility: higher is better",
         "Configuration", "Utility under attack (%)")
    figure_title(fig, 1, "Security and utility under the important_instructions attack (144 cases)")
    legend_below(fig, [(BLUE, None, "Without amps (no defence, prompt-level defences)"),
                       (ORANGE, None, "amps policy (B, B-field)"),
                       (DARK_ORANGE, None, "amps with human confirmation (B+H)")], ncol=1)
    save(fig, "fig1_security_utility")


# ----------------------------------------------------------------------
# Figure 2: three attacks, A, B and B+H, on the four file-based tasks
# ----------------------------------------------------------------------
FILE_TASKS = {"user_task_0", "user_task_2", "user_task_12", "user_task_13"}
ATTACKS = {
    "important_instructions": "important_\ninstructions",
    "tool_knowledge":         "tool_\nknowledge",
    "ignore_previous":        "ignore_\nprevious",
}


def fig2():
    series = [("A (no defence)", "qwen3-4b-instruct_A", BLUE),
              ("B (amps)", "qwen3-4b-instruct_B", ORANGE),
              ("B+H (amps, attentive user)", "qwen3-4b-instruct_BH", DARK_ORANGE)]
    print(f"{'attack':24s}" + "".join(f" {name.split()[0]:>6s} n  ASR %" for name, _, _ in series))
    labels, values = [], {name: [] for name, _, _ in series}
    for attack, label in ATTACKS.items():
        row = []
        for name, folder, _ in series:
            res = {k: r for k, r in attacked_results(folder, attack).items() if k[0] in FILE_TASKS}
            n = len(res)
            row.append((n, 100 * sum(bool(r["security"]) for r in res.values()) / n if n else None))
        if any(v is None for _, v in row):
            print(f"  [skip] {attack}: missing results")
            continue
        labels.append(label)
        for (name, _, _), (_, v) in zip(series, row):
            values[name].append(v)
        print(f"{attack:24s}" + "".join(f" {n:8d} {v:6.1f}" for n, v in row))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = range(len(labels))
    w = 0.26
    for i, (name, _, col) in enumerate(series):
        shift = (i - 1) * w
        rects = ax.bar([j + shift for j in x], values[name], width=w - 0.03, color=col, label=name, zorder=3)
        for r, v in zip(rects, values[name]):
            ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Attack", color=INK)
    ax.set_ylabel("Attack success rate (%)", color=INK)
    figure_title(fig, 2, "Three attacks on the file-based tasks (36 cases per attack)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    save(fig, "fig2_attacks")


# ----------------------------------------------------------------------
# Figure 3: where the attacks succeed (file-based tasks vs. the other tasks)
# ----------------------------------------------------------------------
LIGHT_BLUE = "#9cc3ef"           # second shade of the same blue, for the second group


def fig3():
    labels, file_asr, other_asr = [], [], []
    print(f"{'configuration':16s} {'file tasks':>14s} {'other tasks':>15s}")
    for label, folder in CONFIGS.items():
        res = attacked_results(folder)
        if not res:
            print(f"  [skip] no result in {RUNS}/{folder}")
            continue
        f = [r for k, r in res.items() if k[0] in FILE_TASKS]
        o = [r for k, r in res.items() if k[0] not in FILE_TASKS]
        fs, os_ = sum(bool(r["security"]) for r in f), sum(bool(r["security"]) for r in o)
        labels.append(label)
        file_asr.append(100 * fs / len(f))
        other_asr.append(100 * os_ / len(o))
        print(f"{label.replace(chr(10), ' '):16s} {fs:4d}/{len(f):<3d} {file_asr[-1]:5.1f}%  {os_:4d}/{len(o):<3d} {other_asr[-1]:5.1f}%")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = range(len(labels))
    w = 0.36
    for shift, vals, col, name in ((-w / 2, file_asr, BLUE, "Tasks that read a file (0, 2, 12, 13)"),
                                   (w / 2, other_asr, LIGHT_BLUE, "The other 12 tasks")):
        rects = ax.bar([i + shift for i in x], vals, width=w - 0.04, color=col, label=name, zorder=3)
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Configuration", color=INK)
    ax.set_ylabel("Attack success rate (%)", color=INK)
    figure_title(fig, 3, "Where the attacks succeed (important_instructions)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, "fig3_where_attacks_succeed")


# ----------------------------------------------------------------------
# Figure 4: sink calls blocked (or refused) and executed under B, B-field and B+H
# ----------------------------------------------------------------------
SINKS = {"send_money", "schedule_transaction", "update_scheduled_transaction",
         "update_password", "update_user_info"}


def count_sink_calls(pattern):
    """Count the sink calls blocked by amps and the ones executed, in all result files."""
    blocked = executed = 0
    for path in glob.glob(pattern):
        for m in load_json(path).get("messages", []):
            if m.get("role") == "tool" and (m.get("tool_call") or {}).get("function") in SINKS:
                if "AMPS" in str(m.get("error")) + str(m.get("content")):
                    blocked += 1
                else:
                    executed += 1
    return blocked, executed


def count_confirmations(log_name):
    """B+H: count the calls refused and approved by the simulated user, from its log.

    Requests made outside the benchmark cases (AgentDojo's preliminary checks of the
    injection tasks, logged with an unknown task) are left out.
    """
    path = RESULTS / log_name
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    refused = approved = 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") is None:
            continue
        if r.get("reason") == "rejected_by_human":
            refused += 1
        elif r.get("reason") == "confirmed":
            approved += 1
    return refused, approved


def fig4():
    groups = [("B\nwithout attack", "qwen3-4b-instruct_B", "none"),
              ("B-field\nwithout attack", "qwen3-4b-instruct_Bfield", "none"),
              ("B+H\nwithout attack", "qwen3-4b-instruct_BH_benign_calls.jsonl", "log"),
              ("B\nunder attack", "qwen3-4b-instruct_B", "important_instructions"),
              ("B-field\nunder attack", "qwen3-4b-instruct_Bfield", "important_instructions"),
              ("B+H\nunder attack", "qwen3-4b-instruct_BH_attack_calls.jsonl", "log")]
    labels, blocked, executed = [], [], []
    print(f"{'run':24s} {'blocked or refused':>19s} {'executed':>9s}")
    for label, source, attack in groups:
        if attack == "log":
            counts = count_confirmations(source)
            if counts is None:
                continue
            b, e = counts
        else:
            name = "none" if attack == "none" else "injection_task_*"
            b, e = count_sink_calls(f"{RUNS}/{source}/*/banking/user_task_*/{attack}/{name}.json")
        labels.append(label)
        blocked.append(b)
        executed.append(e)
        print(f"{label.replace(chr(10), ' '):24s} {b:19d} {e:9d}")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = range(len(labels))
    w = 0.36
    for shift, vals, col, name in ((-w / 2, blocked, ORANGE, "Blocked by the policy (B, B-field) or refused by the user (B+H)"),
                                   (w / 2, executed, BLUE, "Executed (for B+H: approved by the user)")):
        rects = ax.bar([i + shift for i in x], vals, width=w - 0.04, color=col, label=name, zorder=3)
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(blocked + executed) * 1.3 + 5)
    ax.set_xlabel("Policy and run", color=INK)
    ax.set_ylabel("Number of sink calls", color=INK)
    figure_title(fig, 4, "Sink calls under the three amps policies")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    save(fig, "fig4_sink_calls")


# ----------------------------------------------------------------------
# Figure 5: scalability (from results/scalability.csv, written by run_scalability.py)
# ----------------------------------------------------------------------
def fig5():
    path = Path("results/scalability.csv")
    if not path.exists():
        print(f"  [skip] {path} not found: run experiments/run_scalability.py first")
        return
    rows = list(csv.DictReader(open(path)))
    agents = sorted({int(r["agents"]) for r in rows})
    # Worst case over sessions and seeds, for each number of agents.
    synth = [max(float(r["t_synth_s"]) for r in rows if int(r["agents"]) == a) for a in agents]
    verif = [1000 * max(float(r["t_verify_s"]) for r in rows if int(r["agents"]) == a) for a in agents]
    nodes = [max(int(r["nodes"]) for r in rows if int(r["agents"]) == a) for a in agents]
    print(f"{'agents':>6s} {'max nodes':>9s} {'synthesis (s)':>14s} {'verification (ms)':>18s}")
    for a, n, s_, v in zip(agents, nodes, synth, verif):
        print(f"{a:6d} {n:9d} {s_:14.3f} {v:18.2f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    figure_title(fig, 5, "Scalability of amps (worst case over seeds and sessions)")
    for ax, vals, title, ylabel, fmt in ((left, synth, "(a) Policy synthesis", "Time (seconds)", "{:.2f}"),
                                         (right, verif, "(b) Verification", "Time (milliseconds)", "{:.1f}")):
        ax.plot(range(len(agents)), vals, color=BLUE, linewidth=2, marker="o", markersize=6, zorder=3,
                label="Worst case over seeds and sessions")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.text(len(agents) - 1, vals[-1], "  " + fmt.format(vals[-1]), va="center",
                fontsize=8.5, color=INK)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([str(a) for a in agents])
        ax.set_xlabel("Number of agents", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_xlim(-0.3, len(agents) - 0.3)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    fig.tight_layout()
    save(fig, "fig5_scalability")


# ----------------------------------------------------------------------
# Figure 6: human confirmation, attentive and naive user (file-based tasks)
# ----------------------------------------------------------------------
def fig6():
    series = [("A\n(no defence)", "qwen3-4b-instruct_A", BLUE, None),
              ("B", "qwen3-4b-instruct_B", ORANGE, None),
              ("B+H\nattentive user", "qwen3-4b-instruct_BH", DARK_ORANGE, None),
              ("B+H\nnaive user", NAIVE, DARK_ORANGE, "///")]
    labels, asr, utility, colours, hatches = [], [], [], [], []
    print(f"{'configuration':24s} {'cases':>5s} {'ASR %':>7s} {'utility %':>10s}")
    for label, folder, col, hatch in series:
        res = {k: r for k, r in attacked_results(folder).items() if k[0] in FILE_TASKS}
        if not res:
            print(f"  [skip] no result in {RUNS}/{folder}")
            continue
        n = len(res)
        a = 100 * sum(bool(r["security"]) for r in res.values()) / n
        u = 100 * sum(bool(r["utility"]) for r in res.values()) / n
        labels.append(label)
        asr.append(a)
        utility.append(u)
        colours.append(col)
        hatches.append(hatch)
        print(f"{label.replace(chr(10), ' '):24s} {n:5d} {a:7.1f} {u:10.1f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 4.6))
    bars(left, labels, asr, colours, "(a) Security: lower is better",
         "Configuration", "Attack success rate (%)", hatches)
    bars(right, labels, utility, colours, "(b) Utility: higher is better",
         "Configuration", "Utility under attack (%)", hatches)
    figure_title(fig, 6, "Human confirmation on the file-based tasks (36 cases)")
    legend_below(fig, [(BLUE, None, "A: no defence"),
                       (ORANGE, None, "B: amps policy, calls refused"),
                       (DARK_ORANGE, None, "B+H: confirmation, attentive user (checks each request)"),
                       (DARK_ORANGE, "///", "B+H: confirmation, naive user (approves every request)")], ncol=2)
    save(fig, "fig6_human")


FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the figures of Chapter 4.")
    parser.add_argument("--fig", type=int, choices=sorted(FIGURES),
                        help="number of the figure to draw (default: all)")
    choice = parser.parse_args().fig
    for number in ([choice] if choice else sorted(FIGURES)):
        print(f"\n=== Figure {number} ===")
        FIGURES[number]()