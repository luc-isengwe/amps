"""Banking study of the extensions: field-level labels and human
confirmation. No LLM is involved: each task is replayed from its declared
flows through the runtime engine (same method as predict_banking.py).

Four configurations, all synthesized by amps and all verified secure:

  B_session      v0.4 manifest (whole transaction list untrusted), no human.
  B_session+H    same labels, human confirmation allowed on actions.
  B_field        field-level labels (projection of transactions), no human.
  B_field+H      field-level labels and human confirmation (examples/banking_fields.yaml).

For each task the script reports whether it completes and how many human
confirmations it needs. A "+H" configuration is secure under the stated
assumption that the human rejects the attacker's action.

Self-check: the 'under' labels written by hand in banking_fields.yaml are
compared with the labels the flows actually carry (computed by simulation).

Usage (from the project folder):
    python experiments/banking_fields_study.py
Outputs:
    results/banking_fields_study.csv     one row per (configuration, task)
    results/banking_fields_summary.json  one entry per configuration
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amps.graph import compile_manifest  # noqa: E402
from amps.labels import fmt  # noqa: E402
from amps.manifest import manifest_from_dict  # noqa: E402
from amps.synthesis import synthesize_ilp  # noqa: E402
from amps.verify import verify  # noqa: E402
from amps.workflows import run_workflow, task_workflows, usage_profile  # noqa: E402

TX = "get_most_recent_transactions"


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "examples" / name).read_text(encoding="utf-8"))


def honest_labels(d: dict) -> dict:
    """Set every requirement's 'under' to the label its flow really carries.

    The labels are computed by simulating each task's flows with no gate
    (usage profile), so they do not depend on any hand-written value.
    """
    d = copy.deepcopy(d)
    for t in d["tasks"]:
        for r in t["requires"]:
            r["under"] = []
    m = manifest_from_dict(d)
    for t, wf in zip(d["tasks"], task_workflows(m)):
        prof = usage_profile(m, wf)
        for r in t["requires"]:
            r["under"] = sorted(set().union(*[lbl for e, lbl in prof if e == r["edge"]] or [set()]))
    return d


def security_first(d: dict) -> dict:
    """No human available: a task may not require acting on untrusted
    content (remove U from call requirements, as in the v0.4 manifest)."""
    d = copy.deepcopy(d)
    d.setdefault("controls", {})["allow_human"] = False
    for t in d["tasks"]:
        for r in t["requires"]:
            if r["edge"].startswith("call:"):
                r["under"] = [x for x in r.get("under", []) if x != "U"]
    return d


def with_human(d: dict) -> dict:
    d = copy.deepcopy(d)
    d.setdefault("controls", {})["allow_human"] = True
    return d


def session_level(fields_manifest: dict) -> dict:
    """The field-level manifest with the transaction list labelled as a whole
    (as in v0.4): no '#trusted' view, every task reads the full output."""
    d = copy.deepcopy(fields_manifest)
    d["system"] = "agentdojo-banking-session"
    for tool in d["tools"]:
        if tool["id"] == TX:
            tool["output"] = {"trust": "untrusted", "data_classes": ["transaction_history"]}
    for t in d["tasks"]:
        for r in t["requires"]:
            r["edge"] = r["edge"].replace(f"{TX}#trusted", TX)
    return honest_labels(d)


def evaluate(name: str, d: dict) -> tuple[dict, list[dict]]:
    m = manifest_from_dict(d)
    cg = compile_manifest(m)
    res = synthesize_ilp(cg)
    secure = verify(cg, res.policy).safe
    rows = []
    for wf in task_workflows(m):
        r = run_workflow(m, res.policy, wf, name)
        rows.append({"config": name, "task": wf.tasks[0], "completes": int(r.success),
                     "confirmations": r.n_confirmations, "first_block": r.first_block or ""})
    summary = {
        "config": name, "status": res.status, "secure": secure,
        "gates": {e: md.value for e, md in res.policy.gates.items()},
        "cost": round(res.cost, 3),
        "conflicts": [f"{c.witness.property}:{c.witness.sink_tool}" for c in res.conflicts],
        "tasks_completed": sum(r["completes"] for r in rows),
        "tasks_with_confirmation": sum(r["confirmations"] > 0 for r in rows),
        "confirmations_total": sum(r["confirmations"] for r in rows),
    }
    return summary, rows


def main() -> None:
    fields_h = load("banking_fields.yaml")

    # Self-check of the hand-written labels.
    computed = honest_labels(fields_h)
    mismatches = [(t["id"], a["edge"], a.get("under", []), b["under"])
                  for t, u in zip(fields_h["tasks"], computed["tasks"])
                  for a, b in zip(t["requires"], u["requires"])
                  if sorted(a.get("under", [])) != b["under"]]
    print("hand-written labels vs computed labels:",
          "all equal" if not mismatches else f"{len(mismatches)} mismatch(es)")
    for mm in mismatches:
        print("   ", mm)

    configs = {
        "B_session": security_first(session_level(fields_h)),
        "B_session+H": with_human(session_level(fields_h)),
        "B_field": security_first(fields_h),
        "B_field+H": with_human(computed),
    }
    summaries, rows = [], []
    for name, d in configs.items():
        s, r = evaluate(name, d)
        summaries.append(s)
        rows += r

    # Readable table: one line per task, one column per configuration.
    names = list(configs)
    print(f"\n{'task':14s}" + "".join(f"{n:>14s}" for n in names))
    for tid in [r["task"] for r in rows if r["config"] == names[0]]:
        cells = []
        for n in names:
            r = next(x for x in rows if x["config"] == n and x["task"] == tid)
            cells.append(("ok" if r["completes"] else "blocked")
                         + (f" ({r['confirmations']}H)" if r["confirmations"] else ""))
        print(f"{tid:14s}" + "".join(f"{c:>14s}" for c in cells))
    print()
    for s in summaries:
        print(f"{s['config']:12s} status={s['status']:8s} secure={s['secure']} cost={s['cost']:<5} "
              f"completed={s['tasks_completed']}/16 with_confirmation={s['tasks_with_confirmation']} "
              f"gates={s['gates']}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with (out / "banking_fields_study.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "banking_fields_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("\nwritten: results/banking_fields_study.csv, results/banking_fields_summary.json")


if __name__ == "__main__":
    main()
