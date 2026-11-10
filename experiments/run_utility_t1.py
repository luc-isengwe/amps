"""Tier T1 utility experiment: how much legitimate work does each policy keep?

For each configuration (A, B, B_sink, B_strict) the script runs every
legitimate workflow of the manifest (single tasks and pairs of tasks in the
same session) through the LangGraph application and the guard, and reports:

* UA_task / UA_mixed  success rates with 95 % Wilson intervals
* UA_safe             success rate on label-safe workflows (best possible: 100 %)
* unsafe_passed       share of label-unsafe workflows let through (must be 0 %)
* block_rate          share of actions blocked
* overhead            mean time of one policy check (microseconds)
* top_blockers        edges that blocked workflows most often
* policy cost and number of gates

It also re-states the static security result of each configuration
(verify): number of attack paths left open, per property. The
security claim comes from the verifier and the replay tests; this
script adds the utility side of the trade-off.

Usage:
    python experiments/run_utility_t1.py [manifest.yaml] [--engine langgraph|direct]

Outputs (folder results/):
    utility_t1_workflows.csv   one row per (configuration, workflow)
    utility_t1_summary.json    one entry per configuration
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow running from any folder

from amps.configs import CONFIGS  # noqa: E402
from amps.graph import compile_manifest  # noqa: E402
from amps.guard import Guard  # noqa: E402
from amps.manifest import load_manifest  # noqa: E402
from amps.metrics import utility_summary  # noqa: E402
from amps.runtime import PolicyEngine  # noqa: E402
from amps.verify import verify  # noqa: E402
from amps.workflows import (WorkflowResult, label_violations, mixed_workflows,  # noqa: E402
                            run_workflow, task_workflows)
from amps_demo.tools import TOOLS  # noqa: E402


def run_langgraph(manifest, policy, wf, config) -> WorkflowResult:
    from amps_demo.app import run_plan

    t0 = time.perf_counter()
    guard = Guard(PolicyEngine(manifest, policy, strict_manifest=False), dict(TOOLS))
    log, guard = run_plan(manifest, policy, wf.actions, guard)
    wall = (time.perf_counter() - t0) * 1e6
    blocked = [e for e in log if not e["allowed"]]
    filtered = sum(e["filtered"] for e in log)
    return WorkflowResult(
        workflow_id=wf.id, kind=wf.kind, config=config, n_actions=len(log),
        n_blocked=len(blocked), n_filtered=filtered,
        success=not blocked and filtered == 0,
        first_block=(blocked[0]["edge"] or blocked[0]["reason"]) if blocked else None,
        label_safe=not label_violations(manifest, wf),
        check_us=sum(d.elapsed_us for d in guard.engine.trace), wall_us=wall)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", default=str(ROOT / "examples" / "demo_4agents.yaml"))
    ap.add_argument("--engine", choices=["langgraph", "direct"], default="langgraph")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    cg = compile_manifest(manifest)
    workflows = task_workflows(manifest) + mixed_workflows(manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, summary = [], {}
    for name, make in CONFIGS.items():
        policy = make(cg)
        results = []
        for wf in workflows:
            if args.engine == "langgraph":
                r = run_langgraph(manifest, policy, wf, name)
            else:
                r = run_workflow(manifest, policy, wf, name, TOOLS)
            results.append(r)
            rows.append({"config": name, "workflow": r.workflow_id, "kind": r.kind,
                         "actions": r.n_actions, "blocked": r.n_blocked,
                         "filtered": r.n_filtered, "success": int(r.success),
                         "label_safe": int(r.label_safe),
                         "first_block": r.first_block or "", "check_us": round(r.check_us, 1),
                         "wall_us": round(r.wall_us, 1)})
        s = utility_summary(results)
        open_paths = verify(cg, policy).witnesses
        summary[name] = {
            "gates": {e: m.value for e, m in policy.gates.items()},
            "cost": round(policy.cost(manifest), 3),
            "UA_task": s["UA_task"].fmt(),
            "UA_mixed": s["UA_mixed"].fmt(),
            "UA_safe": s["UA_safe"].fmt(),
            "unsafe_passed": s["unsafe_passed"].fmt(),
            "block_rate": s["block_rate"].fmt(),
            "filtered_entries": s["filtered_entries"],
            "check_us_per_action": round(s["check_us_per_action"], 2),
            "top_blockers": s["top_blockers"],
            "violable_sinks": {
                "integrity": sum(w.property == "integrity" for w in open_paths),
                "confidentiality": sum(w.property == "confidentiality" for w in open_paths),
            },
        }

    with open(out / "utility_t1_workflows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "utility_t1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n_unsafe = sum(bool(label_violations(manifest, wf)) for wf in workflows)
    print(f"Workflows: {len(workflows)} ({sum(wf.kind == 'task' for wf in workflows)} single, "
          f"{sum(wf.kind == 'mixed' for wf in workflows)} mixed); label-unsafe: {n_unsafe} "
          f"| engine: {args.engine}\n")
    print(f"{'config':9} {'cost':>5} {'UA_safe (best=100%)':>32} {'unsafe passed (must be 0%)':>30} "
          f"{'viol. sinks I/C':>15} {'check us':>9}")
    for name, s in summary.items():
        op = s["violable_sinks"]
        print(f"{name:9} {s['cost']:5.1f} {s['UA_safe']:>32} {s['unsafe_passed']:>30} "
              f"{op['integrity']:>7}/{op['confidentiality']:<7} {s['check_us_per_action']:9.1f}")
    print("\nMain blockers of mixed workflows:")
    for name, s in summary.items():
        print(f"  {name:9} {s['top_blockers'] or '-'}")
    print(f"\nResults written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
