"""How many labelling errors does the canary check catch?

Starting from a manifest whose labels agree with the observations, every
possible single *weakening* error is introduced, one at a time:
  - an untrusted tool output declared trusted;
  - an untrusted field declared trusted (field-level manifests);
  - a sink (integrity-critical or external tool) declared as a plain tool.
Each mutated manifest is compared with the observations, and the error
counts as detected if the comparison reports an UNSAFE finding for the
mutated tool. Weakening errors are the dangerous ones: they silently remove
attack paths from the analysis.

Usage, from the project folder:
    python experiments/label_mutation_study.py results/banking_observations.json examples/banking.yaml
    python experiments/label_mutation_study.py results/banking_observations.json examples/banking_fields.yaml
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amps.inference import collect_evidence, compare, load_observations, unsafe  # noqa: E402
from amps.manifest import manifest_from_dict  # noqa: E402


def mutations(d: dict):
    """Yield (description, tool id, mutated manifest dict) for every single weakening error."""
    for i, t in enumerate(d["tools"]):
        out = t.get("output")
        if out and out.get("trust") == "untrusted" and not out.get("fields"):
            m = copy.deepcopy(d)
            m["tools"][i]["output"]["trust"] = "trusted"
            yield f"{t['id']}: output declared trusted", t["id"], m
        for j, f in enumerate((out or {}).get("fields") or []):
            if f.get("trust") == "untrusted":
                m = copy.deepcopy(d)
                m["tools"][i]["output"]["fields"][j]["trust"] = "trusted"
                m["tools"][i]["output"].pop("trust", None)
                yield f"{t['id']}.{f['id']}: field declared trusted", t["id"], m
        if t.get("integrity_critical") or t.get("input_destination") == "external":
            m = copy.deepcopy(d)
            m["tools"][i]["integrity_critical"] = False
            m["tools"][i]["input_destination"] = "local"
            yield f"{t['id']}: sink declared as a plain tool", t["id"], m


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ev = collect_evidence(load_observations(sys.argv[1]))
    base = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
    # The comparison only looks at the tools; the tasks are dropped so that a
    # mutation that removes an edge or a view does not invalidate the manifest.
    base["tasks"] = []
    if unsafe(compare(manifest_from_dict(base), ev)):
        print("the original manifest already has UNSAFE findings: fix them first")
        return 1
    rows = []
    for desc, tool, d in mutations(base):
        found = [f for f in unsafe(compare(manifest_from_dict(d), ev)) if f.tool == tool]
        observed = tool in ev
        rows.append((desc, observed, bool(found)))
    print(f"{'single weakening error':58s} {'tool observed':>13s} {'detected':>9s}")
    for desc, obs, det in rows:
        print(f"{desc:58s} {'yes' if obs else 'no':>13s} {'YES' if det else 'no':>9s}")
    det = sum(r[2] for r in rows)
    det_obs = sum(r[2] for r in rows if r[1])
    n_obs = sum(r[1] for r in rows)
    print(f"\ndetected: {det}/{len(rows)} errors in total; "
          f"{det_obs}/{n_obs} among the errors on observed tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())