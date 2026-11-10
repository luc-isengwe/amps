"""Scalability study: how do graph size and synthesis time grow with the
number of agents, tools and sessions?

Synthetic manifests are generated with a fixed random seed (reproducible):

* n agents organised as a supervisor tree (agent i delegates to 1 or 2
  agents with a larger index), plus a few random extra delegations;
* 3n tools: about 1/3 return untrusted content, 1/3 return sensitive
  data, 1/3 are sinks (external and/or integrity-critical);
* each agent receives 3 to 5 random tools;
* max(1, n // 4) memories, half of them persistent, shared randomly;
* one task per agent that needs one of its tool returns (with its real
  label), so that the synthesizer must respect utility constraints.

For each (n, T, seed) we measure: graph size, verification time without a
policy, synthesis time (ILP + row generation), iterations, number of gates,
and whether the final policy is verified secure.

Usage (from the project folder):
    python experiments/run_scalability.py            # full grid (a few minutes)
    python experiments/run_scalability.py --quick    # small grid (seconds)
Outputs:
    results/scalability.csv
"""

from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amps.graph import compile_manifest  # noqa: E402
from amps.manifest import manifest_from_dict  # noqa: E402
from amps.synthesis import synthesize_ilp  # noqa: E402
from amps.verify import verify  # noqa: E402


def generate(n: int, seed: int) -> dict:
    rng = random.Random(seed * 1000 + n)
    tools = []
    for k in range(3 * n):
        kind = k % 3
        if kind == 0:  # untrusted reader (web page, e-mail, file)
            tools.append({"id": f"t{k}", "output": {"trust": "untrusted", "data_classes": []}})
        elif kind == 1:  # sensitive reader (database, CRM)
            tools.append({"id": f"t{k}", "output": {"trust": "trusted", "data_classes": ["secret"]}})
        else:  # sink
            tools.append({"id": f"t{k}", "input_destination": rng.choice(["external", "internal"]),
                          "integrity_critical": rng.random() < 0.6})
            if tools[-1]["input_destination"] == "internal":
                tools[-1]["integrity_critical"] = True
    mems = [{"id": f"m{j}", "persistence": "persistent" if j % 2 == 0 else "session"}
            for j in range(max(1, n // 4))]
    agents = []
    for i in range(n):
        own = rng.sample([t["id"] for t in tools], k=min(len(tools), rng.randint(3, 5)))
        children = [c for c in (2 * i + 1, 2 * i + 2) if c < n]
        if n > 2 and rng.random() < 0.3:
            extra = rng.randrange(n)
            if extra != i and extra not in children:
                children.append(extra)
        mem_ids = [m["id"] for m in mems]
        agents.append({
            "id": f"a{i}", "tools": own, "delegates_to": [f"a{c}" for c in children],
            "memory": {"read": rng.sample(mem_ids, k=rng.randint(0, len(mem_ids))),
                       "write": rng.sample(mem_ids, k=rng.randint(0, len(mem_ids)))},
        })
    d = {"system": f"synthetic-{n}-{seed}",
         "data_classes": [{"id": "secret", "sensitivity": "confidential"}],
         "tools": tools, "memory": mems, "agents": agents, "tasks": []}
    m = manifest_from_dict(d)
    for a in agents:
        rets = [t for t in a["tools"] if m.tool(t).output is not None]
        if rets:
            t = rng.choice(rets)
            d["tasks"].append({"id": f"task_{a['id']}", "requires": [
                {"edge": f"ret:{t}->{a['id']}", "under": sorted(m.output_label(t))}]})
    return d


def run(n: int, T: int, seed: int, time_limit: float) -> dict:
    d = generate(n, seed)
    d.setdefault("controls", {})["sessions"] = T
    m = manifest_from_dict(d)
    t0 = time.perf_counter()
    cg = compile_manifest(m)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    unsafe = verify(cg)
    t_verify = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = synthesize_ilp(cg, time_limit=time_limit)
    t_synth = time.perf_counter() - t0
    st = cg.stats()
    return {
        "agents": n, "tools": len(m.tools), "memories": len(m.memory), "sessions": T, "seed": seed,
        "templates": st["templates"], "nodes": st["nodes"], "edges": st["edges"],
        "reached_states": unsafe.reached_states, "witnesses_no_policy": len(unsafe.witnesses),
        "status": res.status, "gates": len(res.policy.gates), "cost": round(res.cost, 3),
        "iterations": res.iterations, "constraints": len(res.witnesses),
        "conflicts": len(res.conflicts),
        "secure_after": int(verify(cg, res.policy).safe or res.status == "conflict"),
        "t_compile_s": round(t_compile, 4), "t_verify_s": round(t_verify, 4),
        "t_synth_s": round(t_synth, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60.0, help="per ILP solve (seconds)")
    a = ap.parse_args()
    agents = [2, 4, 8] if a.quick else [2, 4, 8, 16, 32, 64]
    sessions = [1, 2] if a.quick else [1, 2, 4]
    seeds = [1] if a.quick else [1, 2, 3]
    print(f"machine: {platform.processor() or platform.machine()} | python {platform.python_version()}")
    rows = []
    for n in agents:
        for T in sessions:
            for s in seeds:
                r = run(n, T, s, a.time_limit)
                rows.append(r)
                print(f"n={n:3d} T={T} seed={s}  nodes={r['nodes']:5d} edges={r['edges']:6d} "
                      f"wit0={r['witnesses_no_policy']:3d} it={r['iterations']:3d} "
                      f"gates={r['gates']:3d} status={r['status']:8s} "
                      f"verify={r['t_verify_s']:.3f}s synth={r['t_synth_s']:.2f}s", flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    name = "scalability_quick.csv" if a.quick else "scalability.csv"
    with (out / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written: results/{name}")


if __name__ == "__main__":
    main()
