"""Lifecycle impact analysis: what does a configuration change break?

Given the old manifest, the new manifest and the policy currently deployed,
the analysis answers four questions:

1. **What changed structurally?** Edges added or removed, tool descriptors
   whose fingerprint changed (possible rug pull), tool outputs whose label
   changed.
2. **Is the deployed policy still sufficient?** We carry the old gates over
   to the new graph and run the verifier. Every witness found is an attack
   path that the change opened and the old policy does not stop. This set
   is the *delta* Delta W.
3. **What is the new minimal policy?** Re-synthesis with a warm start (the
   old gates and the new witnesses are given to the solver up front).
4. **Which gates are now stale?** Gates on edges that no longer exist (for
   example after an agent is retired) must be revoked.

The regression tests generated from Delta W (see ``testgen.py``) make each
newly found attack path a permanent test of the deployment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .graph import CompiledGraph, compile_manifest
from .labels import fmt
from .manifest import Manifest
from .policy import Policy, diff_policies
from .synthesis import SynthesisResult, synthesize_ilp
from .verify import ALL_PROPERTIES, Property, Witness, verify


@dataclass
class ImpactReport:
    added_edges: list[str]
    removed_edges: list[str]
    changed_descriptors: list[str]
    changed_labels: dict[str, tuple[str, str]]
    stale_gates: dict[str, str]
    new_witnesses: list[Witness]
    resynthesis: SynthesisResult
    policy_diff: dict[str, dict]
    extra_witnesses: list[Witness] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        """True if the change requires action (new attack path or new descriptor)."""
        return bool(self.new_witnesses or self.changed_descriptors)

    def all_test_witnesses(self) -> list[Witness]:
        """Witnesses to turn into regression tests (no duplicates)."""
        out, seen = [], set()
        for w in self.new_witnesses + self.extra_witnesses:
            if w.signature() not in seen:
                seen.add(w.signature())
                out.append(w)
        return out

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "added_edges": self.added_edges,
            "removed_edges": self.removed_edges,
            "changed_descriptors": self.changed_descriptors,
            "changed_labels": {k: list(v) for k, v in self.changed_labels.items()},
            "stale_gates": self.stale_gates,
            "new_witnesses": [w.to_dict() for w in self.new_witnesses],
            "resynthesis": self.resynthesis.summary(),
            "policy_diff": self.policy_diff,
        }

    def describe(self) -> str:
        lines = ["== Impact analysis =="]
        lines.append(f"edges added   : {self.added_edges or '-'}")
        lines.append(f"edges removed : {self.removed_edges or '-'}")
        lines.append(f"descriptors changed (review before re-approval): "
                     f"{self.changed_descriptors or '-'}")
        for tool, (a, b) in self.changed_labels.items():
            lines.append(f"output label of {tool}: {a} -> {b}")
        lines.append(f"stale gates to revoke: {self.stale_gates or '-'}")
        lines.append(f"new attack paths not stopped by the deployed policy: {len(self.new_witnesses)}")
        for w in self.new_witnesses:
            lines.append(w.describe())
        lines.append(f"re-synthesis: {self.resynthesis.summary()}")
        lines.append(f"policy diff : {self.policy_diff}")
        return "\n".join(lines)


def _realizable_in(cg: CompiledGraph, w: Witness) -> bool:
    """Could this witness already happen in the (old) graph ``cg``?

    True if every edge template exists there and every tool output on the
    path already carried at least the label used by the witness.
    """
    m = cg.manifest
    tools = {t.id for t in m.tools}
    for s in w.steps:
        if s.template not in cg.templates:
            return False
        if s.kind == "ret":
            tool = s.src.split(":", 1)[1]  # may name a view, e.g. tool#trusted
            if tool.split("#", 1)[0] not in tools:
                return False
            try:
                if not s.label <= m.output_label(tool):
                    return False
            except KeyError:  # the view did not exist in the old manifest
                return False
    return True


def impact_analysis(old: Manifest, new: Manifest, deployed: Policy,
                    properties: Iterable[Property] = ALL_PROPERTIES) -> ImpactReport:
    properties = tuple(properties)
    old_cg, new_cg = compile_manifest(old), compile_manifest(new)
    old_edges, new_edges = set(old_cg.templates), set(new_cg.templates)

    # 1. Structural diff.
    old_tools = {t.id: t for t in old.tools}
    changed_descriptors = sorted(
        t.id for t in new.tools
        if t.id in old_tools and deployed.registry.get(t.id) != t.descriptor_sha256())
    changed_labels = {
        t.id: (fmt(old.output_label(t.id)), fmt(new.output_label(t.id)))
        for t in new.tools
        if t.id in old_tools and old.output_label(t.id) != new.output_label(t.id)}
    stale = {e: m.value for e, m in deployed.gates.items() if e not in new_edges}

    # 2. Is the deployed policy still sufficient on the new graph?
    carried = {e: m for e, m in deployed.gates.items() if e in new_edges}
    delta = verify(new_cg, carried, properties).witnesses

    # 3. Re-synthesis with warm start.
    result = synthesize_ilp(new_cg, properties, warm_start=deployed, initial_witnesses=delta)
    # Constraints discovered during re-synthesis that were impossible in the
    # old system are new attack paths too: they also become tests.
    extra = [w for w in result.witnesses if not _realizable_in(old_cg, w)]

    return ImpactReport(
        added_edges=sorted(new_edges - old_edges),
        removed_edges=sorted(old_edges - new_edges),
        changed_descriptors=changed_descriptors,
        changed_labels=changed_labels,
        stale_gates=stale,
        new_witnesses=delta,
        resynthesis=result,
        policy_diff=diff_policies(deployed, result.policy),
        extra_witnesses=extra,
    )
