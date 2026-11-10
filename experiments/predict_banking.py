"""Static per-task prediction for the AgentDojo banking manifest.

For each of the 16 user tasks, the script answers WITHOUT running any LLM:

* label_violations : does the task itself (ground-truth tool calls, no gate)
                     act on a sink after reading untrusted content?
                     -> "label-unsafe" tasks cannot pass any secure policy.
* pass_<config>    : would the task's tool calls pass the policy of each
                     configuration (A, B, B_util, B_sink, B_strict)?

The prediction is written BEFORE the defended AgentDojo runs, so that the
measured results can later be compared with it (no tuning after the fact).

Usage (from the project folder):
    python experiments/predict_banking.py
Output:
    results/banking_prediction.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow running from any folder

from amps.configs import CONFIGS  # noqa: E402
from amps.graph import compile_manifest  # noqa: E402
from amps.manifest import load_manifest  # noqa: E402
from amps.verify import verify  # noqa: E402
from amps.workflows import (label_violations, succeeds_under,  # noqa: E402
                            task_workflows, usage_profile)

MANIFEST = ROOT / "examples" / "banking.yaml"
OUT = ROOT / "results" / "banking_prediction.csv"


def main() -> None:
    m = load_manifest(MANIFEST)
    cg = compile_manifest(m)

    # 1. Build every configuration and re-check it with the static verifier.
    policies = {name: make(cg) for name, make in CONFIGS.items()}
    print("Static check of each configuration:")
    for name, pol in policies.items():
        res = verify(cg, pol)
        gates = ", ".join(f"{md.value}@{e}" for e, md in sorted(pol.gates.items())) or "-"
        print(f"  {name:9s} open attack paths: {len(res.witnesses):2d} | gates: {gates}")

    # 2. Per-task prediction.
    rows = []
    for wf in task_workflows(m):
        profile = usage_profile(m, wf)               # (edge, label) pairs crossed
        viol = label_violations(m, wf)               # [] = label-safe
        row = {"task": wf.tasks[0],
               "label_violations": "+".join(viol) or "none"}
        for name, pol in policies.items():
            row[f"pass_{name}"] = int(succeeds_under(profile, pol.gates))
        rows.append(row)

    # 3. Print a readable table and save the CSV.
    cols = list(rows[0])
    print("\n" + " | ".join(f"{c:>16s}" if i < 2 else c for i, c in enumerate(cols)))
    for r in rows:
        print(" | ".join(f"{str(r[c]):>16s}" if i < 2 else f"{r[c]:>{len(c)}}"
                         for i, c in enumerate(cols)))
    safe = [r for r in rows if r["label_violations"] == "none"]
    print(f"\nlabel-safe tasks: {len(safe)}/{len(rows)} -> "
          + ", ".join(r["task"].replace("user_task_", "") for r in safe))
    for name in policies:
        k = sum(r[f"pass_{name}"] for r in rows)
        print(f"  predicted benign utility upper bound under {name:9s}: {k}/{len(rows)}")

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
