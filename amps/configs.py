"""Policy configurations compared in the evaluation.

``A``         no gate (only the manifest's default deny of undeclared flows).
``B``         the policy synthesized by amps (minimal cost, verified).
``B_sink``    a common developer heuristic: put an integrity gate directly
              in front of every state-changing tool, and nothing else.
              It operationalises the "hand-written policy" baseline
              (``B_man``) in a reproducible way.
``B_util``    the policy synthesized by amps with the *workflow-weighted*
              objective: among secure policies, first break as few
              legitimate workflows as possible, then minimise gate cost.
``B_strict``  a CaMeL-like strict rule: no tool may be called once the
              caller's context contains untrusted content (integrity gate
              on *every* call edge).

The point of comparing them is the trade-off: how much utility each
configuration keeps (this module's companion, ``workflows.py``) for the
protection it gives (``verify.py``).
"""

from __future__ import annotations

from .graph import CompiledGraph
from .labels import Mode
from .policy import Policy
from .synthesis import synthesize_ilp


def config_A(cg: CompiledGraph) -> Policy:
    return Policy.for_manifest(cg.manifest, {}, {"config": "A"})


def config_B(cg: CompiledGraph) -> Policy:
    p = synthesize_ilp(cg).policy
    p.meta["config"] = "B"
    return p


def config_B_util(cg: CompiledGraph) -> Policy:
    from .workflows import mixed_workflows, task_workflows, usage_profile

    m = cg.manifest
    wfs = task_workflows(m) + mixed_workflows(m)
    p = synthesize_ilp(cg, workflows=[(1.0, usage_profile(m, w)) for w in wfs]).policy
    p.meta["config"] = "B_util"
    return p


def config_B_sink(cg: CompiledGraph) -> Policy:
    m = cg.manifest
    gates = {e: Mode.INTEGRITY for e, t in cg.templates.items()
             if t.kind == "call" and m.tool(t.dst).integrity_critical}
    return Policy.for_manifest(m, gates, {"config": "B_sink"})


def config_B_strict(cg: CompiledGraph) -> Policy:
    gates = {e: Mode.INTEGRITY for e, t in cg.templates.items() if t.kind == "call"}
    return Policy.for_manifest(cg.manifest, gates, {"config": "B_strict"})


CONFIGS = {
    "A": config_A,
    "B": config_B,
    "B_util": config_B_util,
    "B_sink": config_B_sink,
    "B_strict": config_B_strict,
}
