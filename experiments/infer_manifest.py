"""Check (and suggest) the labels of a manifest from canary observations.

Usage, from the project folder:

    python experiments/infer_manifest.py results/banking_observations.json examples/banking.yaml
    python experiments/infer_manifest.py results/banking_observations.json examples/banking_fields.yaml

Prints one line per tool and per aspect (output label, sink), with a verdict:
    UNSAFE ...         the manifest is less strict than what was observed (must be fixed)
    conservative ...   the manifest is stricter than what was observed (allowed, may cost utility)
    not observed       the tool was never called (no evidence either way)
    agrees             the manifest matches the evidence

Also writes results/<manifest>_inferred_tools.yaml: the tool labels suggested
by the evidence, to review before using them.
Exit code 1 if at least one UNSAFE finding is reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amps.inference import (collect_evidence, compare, inferred_output,  # noqa: E402
                            load_observations, unsafe)
from amps.manifest import load_manifest  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    obs = load_observations(sys.argv[1])
    manifest = load_manifest(sys.argv[2])
    ev = collect_evidence(obs)
    findings = compare(manifest, ev)

    print(f"observations: {len(obs['calls'])} calls, {len(obs['canaries'])} seeded locations, "
          f"{len(ev)} tools")
    print(f"manifest    : {sys.argv[2]}\n")
    print(f"{'tool':32s} {'aspect':7s} verdict / detail")
    for f in findings:
        print(f"{f.tool:32s} {f.aspect:7s} {f.verdict}\n{'':41s}{f.detail}")
    bad = unsafe(findings)
    print(f"\nUNSAFE findings: {len(bad)}")

    suggested = []
    for tool in manifest.tools:
        e = ev.get(tool.id)
        entry = {"id": tool.id}
        if e is not None:
            out = inferred_output(e)
            if out is not None:
                entry["output"] = out
            entry["observed_state_change"] = e.is_action
            entry["observed_calls"] = e.calls
        else:
            entry["observed_calls"] = 0
        suggested.append(entry)
    out_path = ROOT / "results" / f"{Path(sys.argv[2]).stem}_inferred_tools.yaml"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(yaml.safe_dump({"tools": suggested}, sort_keys=False), encoding="utf-8")
    print(f"written: {out_path.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
