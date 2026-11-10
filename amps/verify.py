"""Static verifier: label propagation over G_T under a given policy.

Question answered
-----------------
"Given this policy, can a flow that violates integrity or confidentiality
reach a sink?" If yes, the verifier returns a *witness*: the exact chain of
flows (edges and labels) that leads to the violation.

Abstract semantics (identical to the runtime engine)
-----------------------------------------------------
For every node we compute the set of labels that can reach it.

* ``out:<tool>`` nodes start with the tool's output label.
* A flow carrying label ``L`` crosses an edge unless the gate on that edge
  blocks ``L`` (see :func:`amps.labels.blocks`).
* **Agent contexts accumulate.** Within one session an agent's context is a
  single conversation, so if ``{U}`` can reach it and ``{S}`` can reach it,
  then ``{U,S}`` can be present at the same time. We therefore close the
  label set of every ``ctx`` node under union. This is the "session-level
  conservative label" of the thesis (FIDES-like).
* **Memory entries do not accumulate.** Each memory entry keeps the label of
  the context that wrote it (origin bound at write time). Two entries with
  labels ``{U}`` and ``{S}`` stay separate inside the store; they are only
  combined when an agent reads both into its context.

The analysis is a breadth-first fixed-point computation over pairs
``(node, label)``. There are at most ``3 x |nodes|`` such pairs, so it runs
in time linear in the size of the graph.

Witnesses
---------
For every pair we record how it was first derived (from a source, through
an edge, or by union of two labels in a context). Following these records
backwards gives a derivation tree, which we flatten into an ordered list of
steps. The order is a valid execution order: every step comes after the
steps that produce its input, and steps are sorted by session. This is what
makes witnesses *replayable* against the runtime engine (``replay.py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Union

from .graph import CompiledGraph, node_session
from .labels import (
    NON_EMPTY_LABELS,
    SENSITIVE,
    UNTRUSTED,
    Label,
    Mode,
    blocks,
    confidentiality_violation,
    effective_modes,
    fmt,
    integrity_violation,
    make_label,
)
from .policy import Policy

Property = Literal["integrity", "confidentiality"]
ALL_PROPERTIES: tuple[Property, ...] = ("integrity", "confidentiality")


@dataclass(frozen=True)
class Step:
    """One flow in a witness: edge instance + label carried."""

    template: str  # edge template id, e.g. "msg:supervisor->executor"
    kind: str  # edge kind (ret, call, mw, mr, msg, persist)
    src: str  # source node, e.g. "ctx:supervisor@1"
    dst: str  # destination node
    label: Label  # label of the flow on this edge

    @property
    def session(self) -> int:
        """Session in which the step happens (used to order replay)."""
        s = node_session(self.src)
        return s if s is not None else (node_session(self.dst) or 1)

    def to_dict(self) -> dict:
        return {"template": self.template, "kind": self.kind, "src": self.src,
                "dst": self.dst, "label": sorted(self.label)}

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(d["template"], d["kind"], d["src"], d["dst"], make_label(d["label"]))


@dataclass
class Witness:
    """A concrete attack path that violates a security property."""

    property: Property
    sink_tool: str
    label: Label  # label reaching the sink
    steps: list[Step] = field(default_factory=list)

    def blocking_options(self, fixed: Iterable[str] = ()) -> set[tuple[str, Mode]]:
        """All (edge, mode) pairs such that one gate would break this witness.

        A step can be broken by any mode that blocks the label on that step.
        Edges listed in ``fixed`` cannot be gated.
        """
        fixed = set(fixed)
        options: set[tuple[str, Mode]] = set()
        for step in self.steps:
            if step.template in fixed:
                continue
            for mode in effective_modes(step.label):
                options.add((step.template, mode))
        return options

    def signature(self) -> tuple:
        """Session-independent identity of the witness.

        Two witnesses with the same property, sink and set of (template,
        label) steps describe the same attack, even if the session numbers
        differ. Used to compute the new witnesses introduced by a change.
        """
        steps = frozenset((s.template, tuple(sorted(s.label))) for s in self.steps)
        return (self.property, self.sink_tool, tuple(sorted(steps)))

    def templates(self) -> list[str]:
        """Edge templates used by the witness, without duplicates, in order."""
        seen: list[str] = []
        for s in self.steps:
            if s.template not in seen:
                seen.append(s.template)
        return seen

    def describe(self) -> str:
        """Multi-line, human-readable explanation (used by the CLI)."""
        head = f"[{self.property}] {fmt(self.label)} reaches sink '{self.sink_tool}'"
        lines = [head]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"  {i:2d}. {s.src:>28} --{s.template}--> {s.dst:<28} {fmt(s.label)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"property": self.property, "sink_tool": self.sink_tool,
                "label": sorted(self.label), "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Witness":
        return cls(d["property"], d["sink_tool"], make_label(d["label"]),
                   [Step.from_dict(s) for s in d["steps"]])


@dataclass
class VerificationResult:
    safe: bool
    witnesses: list[Witness]
    #: Number of (node, label) pairs reached (useful for scalability figures).
    reached_states: int


# A derivation record explains how a (node, label) pair was first reached.
#   ("source",)                         the node is a tool output
#   ("edge", u, template, kind, src_label)  reached through edge u -> node
#                                       (src_label = label before the edge; it
#                                       differs from the node's label only on
#                                       persist edges that cross principals)
#   ("union", L1, L2)                   union of two labels already in the context
_Derivation = tuple


def _gates_of(policy: Union[Policy, Mapping[str, Mode], None]) -> Mapping[str, Mode]:
    if policy is None:
        return {}
    if isinstance(policy, Policy):
        return policy.gates
    return policy


def propagate(cg: CompiledGraph, policy: Union[Policy, Mapping[str, Mode], None] = None
              ) -> dict[tuple[str, Label], _Derivation]:
    """Compute every reachable (node, label) pair and how it was derived."""
    gates = _gates_of(policy)
    g = cg.graph
    reached: dict[tuple[str, Label], _Derivation] = {}
    labels_at: dict[str, set[Label]] = {}
    queue: deque[tuple[str, Label]] = deque()

    def add(node: str, label: Label, how: _Derivation) -> None:
        if (node, label) in reached:
            return
        reached[(node, label)] = how
        labels_at.setdefault(node, set()).add(label)
        queue.append((node, label))
        # Session-level accumulation inside an agent context.
        if g.nodes[node]["kind"] == "ctx":
            for other in list(labels_at[node]):
                union = label | other
                if union != label and union != other:
                    add(node, union, ("union", label, other))

    # Seed: every tool output with a non-empty label.
    for node, data in g.nodes(data=True):
        if data["kind"] == "out" and data["label"]:
            add(node, data["label"], ("source",))

    # Breadth-first propagation.
    while queue:
        u, label = queue.popleft()
        for v in g.successors(u):
            edge = g.edges[u, v]
            if blocks(gates.get(edge["template"]), label):
                continue  # the gate stops this flow
            # ``adds``: labels gained on the edge (cross-principal persistence).
            add(v, label | edge.get("adds", frozenset()),
                ("edge", u, edge["template"], edge["kind"], label))
    return reached


def _steps_for(reached: dict, node: str, label: Label, out: list[Step], seen: set) -> None:
    """Append, in post-order, the steps needed to derive (node, label)."""
    how = reached[(node, label)]
    if how[0] == "source":
        return
    if how[0] == "union":
        _steps_for(reached, node, how[1], out, seen)
        _steps_for(reached, node, how[2], out, seen)
        return
    _, u, template, kind, src_label = how
    _steps_for(reached, u, src_label, out, seen)
    step = Step(template, kind, u, node, src_label)
    if step not in seen:
        seen.add(step)
        out.append(step)


def extract_witness(reached: dict, sink_node: str, label: Label, prop: Property,
                    sink_tool: str) -> Witness:
    steps: list[Step] = []
    _steps_for(reached, sink_node, label, steps, set())
    # Stable sort by session keeps the dependency order inside a session and
    # puts earlier sessions first (cross-session flows only go forward).
    steps.sort(key=lambda s: s.session)
    return Witness(prop, sink_tool, label, steps)


def verify(cg: CompiledGraph, policy: Union[Policy, Mapping[str, Mode], None] = None,
           properties: Iterable[Property] = ALL_PROPERTIES,
           ignore: Optional[set[tuple[str, Property]]] = None) -> VerificationResult:
    """Check a policy. Returns at most one witness per (sink, property).

    ``ignore`` lists (sink_tool, property) pairs that are known to be
    unavoidable (security-utility conflicts); they are not reported again.
    """
    properties = tuple(properties)
    ignore = ignore or set()
    reached = propagate(cg, policy)
    witnesses: list[Witness] = []

    # Prefer the simplest label when several reach the same sink.
    order = {lbl: i for i, lbl in enumerate(NON_EMPTY_LABELS)}
    sink_nodes = sorted(n for n, d in cg.graph.nodes(data=True) if d["kind"] == "in")
    for node in sink_nodes:
        data = cg.graph.nodes[node]
        tool = data["entity"]
        present = sorted((lbl for (n, lbl) in reached if n == node), key=order.get)
        if "integrity" in properties and data["integrity_sink"] and (tool, "integrity") not in ignore:
            hit = next((lbl for lbl in present if integrity_violation(lbl)), None)
            if hit is not None:
                witnesses.append(extract_witness(reached, node, hit, "integrity", tool))
        if ("confidentiality" in properties and data["confidentiality_sink"]
                and (tool, "confidentiality") not in ignore):
            hit = next((lbl for lbl in present if confidentiality_violation(lbl)), None)
            if hit is not None:
                witnesses.append(extract_witness(reached, node, hit, "confidentiality", tool))
    return VerificationResult(safe=not witnesses, witnesses=witnesses,
                              reached_states=len(reached))


__all__ = [
    "Step", "Witness", "VerificationResult", "verify", "propagate", "extract_witness",
    "Property", "ALL_PROPERTIES", "UNTRUSTED", "SENSITIVE",
]
