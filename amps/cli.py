"""Command-line interface.

Examples (run from the project folder)::

    amps validate   examples/demo_4agents.yaml
    amps graph      examples/demo_4agents.yaml
    amps verify     examples/demo_4agents.yaml                 # no policy: list attack paths
    amps synthesize examples/demo_4agents.yaml -o policy.json
    amps verify     examples/demo_4agents.yaml -p policy.json  # check a policy
    amps mutate     examples/demo_4agents.yaml permission_drift agent=reviewer tool=send_email -o new.yaml
    amps impact     examples/demo_4agents.yaml new.yaml -p policy.json -o new_policy.json --tests tests/generated

Every command prints a readable report; ``--json`` prints machine-readable
output instead (used by the experiment scripts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__
from .graph import compile_manifest
from .impact import impact_analysis
from .manifest import load_manifest, manifest_from_dict
from .mutations import MUTATIONS
from .policy import Policy
from .synthesis import synthesize
from .testgen import generate_regression_tests
from .verify import verify


def _print(obj, as_json: bool) -> None:
    print(json.dumps(obj, indent=2) if as_json else obj)


def cmd_validate(a) -> int:
    m = load_manifest(a.manifest)
    print(f"OK: '{m.system}' - {len(m.agents)} agents, {len(m.tools)} tools, "
          f"{len(m.memory)} memories, {len(m.tasks)} tasks, "
          f"{len(m.edge_templates())} edge templates")
    return 0


def cmd_graph(a) -> int:
    cg = compile_manifest(load_manifest(a.manifest), a.sessions)
    if a.json:
        _print(cg.stats(), True)
    else:
        for k, v in cg.stats().items():
            print(f"{k:>12}: {v}")
        print("\nEdge templates:")
        for e in cg.templates:
            print(f"  {e}")
    return 0


def cmd_verify(a) -> int:
    m = load_manifest(a.manifest)
    cg = compile_manifest(m, a.sessions)
    policy = Policy.load(a.policy) if a.policy else None
    res = verify(cg, policy)
    if a.json:
        _print({"safe": res.safe, "witnesses": [w.to_dict() for w in res.witnesses]}, True)
    else:
        print("SAFE: no witness found." if res.safe else f"UNSAFE: {len(res.witnesses)} witness(es)")
        for w in res.witnesses:
            print(w.describe())
    return 0 if res.safe else 1


def cmd_synthesize(a) -> int:
    m = load_manifest(a.manifest)
    cg = compile_manifest(m, a.sessions)
    props = ("integrity",) if a.integrity_only else ("integrity", "confidentiality")
    if a.workflow_weighted:
        from .workflows import mixed_workflows, task_workflows, usage_profile
        wfs = task_workflows(m) + mixed_workflows(m)
        res = synthesize(cg, method="ilp", properties=props,
                         workflows=[(1.0, usage_profile(m, w)) for w in wfs])
    else:
        res = synthesize(cg, method=a.method, properties=props)
    if a.output:
        res.policy.save(a.output)
    if a.json:
        _print({**res.summary(), "gates": {e: md.value for e, md in res.policy.gates.items()}}, True)
    else:
        print(f"status: {res.status} | cost: {res.cost:.3f} | iterations: {res.iterations} | "
              f"constraints: {len(res.witnesses)} | time: {res.solve_seconds:.3f}s")
        for e, md in res.policy.gates.items():
            print(f"  gate {md.value}  {e}")
        for c in res.conflicts:
            print("\nCONFLICT (cannot be blocked without breaking a task):")
            print(c.witness.describe())
            for r in c.reasons:
                print(f"    - {r}")
        if a.output:
            print(f"\npolicy written to {a.output}")
    return 0 if res.status == "secure" else 2


def _parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k] = v.split(",") if k.startswith("grant_") else (
            {"true": True, "false": False}.get(v.lower(), v))
    return out


def cmd_mutate(a) -> int:
    with open(a.manifest, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    new = MUTATIONS[a.mutation](data, **_parse_kv(a.args))
    manifest_from_dict(new)  # validate before writing
    Path(a.output).write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"mutated manifest written to {a.output}")
    return 0


def cmd_impact(a) -> int:
    old, new = load_manifest(a.old), load_manifest(a.new)
    deployed = Policy.load(a.policy)
    rep = impact_analysis(old, new, deployed)
    if a.output:
        rep.resynthesis.policy.save(a.output)
    if a.tests:
        if not a.output:
            print("--tests needs -o/--output (the tests run against the new policy)", file=sys.stderr)
            return 2
        name = a.name or Path(a.new).stem
        j, p = generate_regression_tests(rep.all_test_witnesses(), a.new, a.output, a.tests, name)
        print(f"regression tests: {p} ({len(rep.all_test_witnesses())} scenario(s))")
    _print(rep.to_dict() if a.json else rep.describe(), a.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amps", description="Agent Manifest -> Policy Synthesis")
    p.add_argument("--version", action="version", version=f"amps {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="validate a manifest")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("graph", help="compile and describe the attack-path graph")
    s.add_argument("manifest")
    s.add_argument("--sessions", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_graph)

    s = sub.add_parser("verify", help="list attack paths (optionally under a policy)")
    s.add_argument("manifest")
    s.add_argument("-p", "--policy")
    s.add_argument("--sessions", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("synthesize", help="compute a minimal-cost secure policy")
    s.add_argument("manifest")
    s.add_argument("-o", "--output")
    s.add_argument("--method", choices=["auto", "ilp", "mincut"], default="auto")
    s.add_argument("--integrity-only", action="store_true")
    s.add_argument("--workflow-weighted", action="store_true",
                   help="first break as few legitimate workflows as possible, then minimise cost")
    s.add_argument("--sessions", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_synthesize)

    s = sub.add_parser("mutate", help="apply a lifecycle mutation to a manifest")
    s.add_argument("manifest")
    s.add_argument("mutation", choices=sorted(MUTATIONS))
    s.add_argument("args", nargs="*", help="key=value arguments, e.g. agent=reviewer tool=send_email")
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=cmd_mutate)

    s = sub.add_parser("impact", help="impact analysis of a configuration change")
    s.add_argument("old")
    s.add_argument("new")
    s.add_argument("-p", "--policy", required=True, help="policy currently deployed")
    s.add_argument("-o", "--output", help="where to write the re-synthesized policy")
    s.add_argument("--tests", help="folder for generated regression tests")
    s.add_argument("--name", help="name of the generated test module")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_impact)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
