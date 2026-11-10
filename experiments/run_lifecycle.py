"""Lifecycle study (RQ3, hypothesis H6): what happens when a deployed agent
system changes?

For each manifest, every applicable configuration change is generated
systematically (no hand-picking):

  rug_pull          each tool with an output: its provider becomes untrusted
  permission_drift  each (agent, tool) pair the agent did not have
  new_delegation    each ordered pair of agents without a channel
  widen_memory      each memory made persistent; each read / write grant added
  add_sink_tool     each agent receives a new external "export" tool
  neutral_tool      each agent receives a harmless trusted tool (control)
  retire_agent      each agent removed (when the system has more than one)

For each change, starting from the minimal policy deployed for the old
manifest, the impact analysis is run and we record:

  detected          the analysis flags the change (new attack path or changed descriptor)
  new_paths         attack paths opened by the change that the deployed policy does not stop
  cost_old/new      cost of the deployed policy / of the re-synthesized policy
  gates_+/-/~       gates added / removed / changed by the re-synthesis
  stale             gates that refer to edges that no longer exist
  tests             regression scenarios generated
  tests_ok          generated scenarios that the NEW policy blocks (expected: all)
  tests_catch_old   new-path scenarios that the OLD policy lets through (expected: all)
  seconds           time of the impact analysis (including re-synthesis)

Usage (from the project folder):
    python experiments/run_lifecycle.py
Outputs:
    results/lifecycle_mutations.csv     one row per (manifest, change)
    results/lifecycle_summary.json      aggregates per manifest and change type
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amps.graph import compile_manifest  # noqa: E402
from amps.impact import impact_analysis  # noqa: E402
from amps.manifest import manifest_from_dict  # noqa: E402
from amps.mutations import MUTATIONS  # noqa: E402
from amps.policy import Policy  # noqa: E402
from amps.replay import replay_witness  # noqa: E402
from amps.runtime import PolicyEngine  # noqa: E402
from amps.synthesis import synthesize_ilp  # noqa: E402

MANIFESTS = ["demo_4agents.yaml", "banking.yaml", "banking_fields.yaml",
             "support_multitenant.yaml"]


def changes(d: dict):
    """Yield (mutation name, kwargs) for every applicable change of ``d``."""
    agents = [a["id"] for a in d["agents"]]
    tools = [t["id"] for t in d["tools"]]
    mems = [m["id"] for m in d.get("memory", [])]
    for t in d["tools"]:
        if t.get("output"):
            yield "rug_pull", {"tool": t["id"]}
    for a in d["agents"]:
        for t in tools:
            if t not in a.get("tools", []):
                yield "permission_drift", {"agent": a["id"], "tool": t}
    for a in d["agents"]:
        for b in agents:
            if b != a["id"] and b not in a.get("delegates_to", []):
                yield "new_delegation", {"src": a["id"], "dst": b}
    for mem in d.get("memory", []):
        if mem.get("persistence", "session") != "persistent":
            yield "widen_memory", {"memory": mem["id"]}
        for a in d["agents"]:
            acc = a.get("memory", {})
            if mem["id"] not in acc.get("read", []):
                yield "widen_memory", {"memory": mem["id"], "make_persistent": False,
                                       "grant_read_to": [a["id"]]}
            if mem["id"] not in acc.get("write", []):
                yield "widen_memory", {"memory": mem["id"], "make_persistent": False,
                                       "grant_write_to": [a["id"]]}
    for a in agents:
        yield "add_sink_tool", {"agent": a}
        yield "neutral_tool", {"agent": a}
    if len(agents) > 1:
        for a in agents:
            yield "retire_agent", {"agent": a}
    del mems


def label(name: str, kw: dict) -> str:
    return name + "(" + ", ".join(f"{k}={v}" for k, v in kw.items()) + ")"


def study(fname: str) -> list[dict]:
    d = yaml.safe_load((ROOT / "examples" / fname).read_text(encoding="utf-8"))
    old = manifest_from_dict(d)
    deployed = synthesize_ilp(compile_manifest(old)).policy
    rows = []
    for name, kw in changes(d):
        try:
            new = manifest_from_dict(MUTATIONS[name](d, **kw))
        except ValueError as exc:  # the change makes the manifest invalid
            rows.append({"manifest": fname, "change": name, "args": label(name, kw),
                         "valid": 0, "error": str(exc).splitlines()[-1]})
            continue
        t0 = time.perf_counter()
        rep = impact_analysis(old, new, deployed)
        secs = time.perf_counter() - t0
        new_policy = rep.resynthesis.policy
        scenarios = rep.all_test_witnesses()
        ok = sum(not replay_witness(PolicyEngine(new, new_policy), w).violation
                 for w in scenarios)
        carried = Policy.for_manifest(new, {e: m for e, m in deployed.gates.items()
                                            if e in compile_manifest(new).templates})
        caught = sum(replay_witness(PolicyEngine(new, carried), w).violation
                     for w in rep.new_witnesses)
        diff = rep.policy_diff
        rows.append({
            "manifest": fname, "change": name, "args": label(name, kw), "valid": 1, "error": "",
            "detected": int(rep.detected), "new_paths": len(rep.new_witnesses),
            "descriptor_changed": int(bool(rep.changed_descriptors)),
            "status": rep.resynthesis.status,
            "cost_old": round(deployed.cost(old), 3), "cost_new": round(rep.resynthesis.cost, 3),
            "gates_added": len(diff["added"]), "gates_removed": len(diff["removed"]),
            "gates_changed": len(diff["changed"]), "stale": len(rep.stale_gates),
            "tests": len(scenarios), "tests_ok": ok,
            "tests_catch_old": caught, "seconds": round(secs, 4),
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["manifest"], r["change"]) for r in rows})
    for fname, change in keys:
        rs = [r for r in rows if r["manifest"] == fname and r["change"] == change and r["valid"]]
        if not rs:
            continue
        out.append({
            "manifest": fname, "change": change, "n": len(rs),
            "detected": sum(r["detected"] for r in rs),
            "with_new_paths": sum(r["new_paths"] > 0 for r in rs),
            "new_paths_total": sum(r["new_paths"] for r in rs),
            "conflicts": sum(r["status"] == "conflict" for r in rs),
            "mean_cost_increase": round(sum(r["cost_new"] - r["cost_old"] for r in rs) / len(rs), 3),
            "tests_total": sum(r["tests"] for r in rs),
            "tests_ok": sum(r["tests_ok"] for r in rs),
            "tests_catch_old": sum(r["tests_catch_old"] for r in rs),
            "max_seconds": max(r["seconds"] for r in rs),
            "mean_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 4),
        })
    return out


def main() -> None:
    rows = []
    for fname in MANIFESTS:
        rows += study(fname)
    summary = summarize(rows)
    print(f"{'manifest':26s} {'change':17s} {'n':>3s} {'det':>4s} {'paths':>6s} {'confl':>6s} "
          f"{'dcost':>6s} {'tests':>6s} {'ok':>4s} {'old':>4s} {'max_s':>6s}")
    for s in summary:
        print(f"{s['manifest']:26s} {s['change']:17s} {s['n']:3d} {s['detected']:4d} "
              f"{s['new_paths_total']:6d} {s['conflicts']:6d} {s['mean_cost_increase']:6.2f} "
              f"{s['tests_total']:6d} {s['tests_ok']:4d} {s['tests_catch_old']:4d} "
              f"{s['max_seconds']:6.3f}")
    invalid = [r for r in rows if not r["valid"]]
    print(f"\nchanges evaluated: {len(rows) - len(invalid)} (invalid manifests skipped: {len(invalid)})")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                    if k in rows[0] else 99)
    with (out / "lifecycle_mutations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "lifecycle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("written: results/lifecycle_mutations.csv, results/lifecycle_summary.json")


if __name__ == "__main__":
    main()
