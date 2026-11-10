"""Does the synthesized policy depend on the utility costs? 

The base cost of each edge kind and the weight of each gate mode are
modelling choices (see amps/manifest.py). This script repeats the synthesis
with many random cost settings and records, for each manifest:
  - how many synthesized policies are secure (verified again from scratch);
  - how many are identical to the policy obtained with the default costs;
  - how many distinct policies appear, and how often.

Random settings:
  - base cost of every edge kind drawn uniformly in [1, 5];
  - mode weights: w_R = 1, w_I in [0.2, 1], w_C in [0.05, w_I], w_H in [0.1, 1.5],
    so that the ordering w_R >= w_I >= w_C > 0 required by amps always holds.

Usage, from the project folder:
    python experiments/cost_sensitivity.py
    python experiments/cost_sensitivity.py --runs 500 --seed 1
Output: a summary on screen and results/cost_sensitivity.json
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amps.graph import compile_manifest  # noqa: E402
from amps.manifest import Manifest  # noqa: E402
from amps.synthesis import synthesize  # noqa: E402
from amps.verify import verify  # noqa: E402

MANIFESTS = [
    "examples/banking.yaml",
    "examples/demo_4agents.yaml",
    "examples/support_multitenant.yaml",
]
EDGE_KINDS = ["prompt", "call", "ret", "msg", "mw", "mr", "persist"]


def policy_key(gates: dict) -> tuple:
    """Hashable, readable description of a policy: sorted (edge, mode) pairs."""
    return tuple(sorted((edge, mode.value) for edge, mode in gates.items()))


def random_controls(rng: random.Random) -> tuple[dict, dict]:
    costs = {kind: round(rng.uniform(1.0, 5.0), 2) for kind in EDGE_KINDS}
    w_i = round(rng.uniform(0.2, 1.0), 2)
    w_c = round(rng.uniform(0.05, w_i), 2)
    w_h = round(rng.uniform(0.1, 1.5), 2)
    weights = {"R": 1.0, "I": w_i, "C": w_c, "H": w_h}
    return costs, weights


def study(path: str, runs: int, rng: random.Random) -> dict:
    with open(ROOT / path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    default = synthesize(compile_manifest(Manifest.model_validate(data)))
    default_key = policy_key(default.policy.gates)

    secure = same = conflicts = 0
    variants: collections.Counter = collections.Counter()
    t0 = time.perf_counter()
    for _ in range(runs):
        d = copy.deepcopy(data)
        controls = d.setdefault("controls", {})
        controls["costs"], controls["mode_weights"] = random_controls(rng)
        manifest = Manifest.model_validate(d)
        cg = compile_manifest(manifest)
        result = synthesize(cg)
        if result.status == "conflict":
            conflicts += 1
        # Independent check: verify the returned gates from scratch.
        if verify(cg, result.policy.gates).safe or result.status == "conflict":
            secure += 1
        key = policy_key(result.policy.gates)
        same += key == default_key
        variants[key] += 1

    return {
        "manifest": path,
        "runs": runs,
        "secure": secure,
        "conflicts": conflicts,
        "same_as_default": same,
        "distinct_policies": len(variants),
        "default_policy": [list(p) for p in default_key],
        "policies": [{"count": n, "gates": [list(p) for p in key]}
                     for key, n in variants.most_common()],
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200, help="random settings per manifest")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--output", default="results/cost_sensitivity.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results = []
    for path in MANIFESTS:
        print(f"Running {path} ({args.runs} settings)...", flush=True)
        r = study(path, args.runs, rng)
        results.append(r)
        print(f"  secure: {r['secure']}/{r['runs']}"
              f" | same policy as default costs: {r['same_as_default']}/{r['runs']}"
              f" | distinct policies: {r['distinct_policies']}"
              f" | {r['seconds']} s")
        for p in r["policies"]:
            gates = ", ".join(f"{mode} on {edge}" for edge, mode in p["gates"])
            print(f"    {p['count']:>4} x  {gates}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "runs": args.runs, "results": results}, fh, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
