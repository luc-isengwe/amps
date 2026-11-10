"""Compile a manifest into the attack-path graph G_T.

The graph is a directed graph (``networkx.DiGraph``) whose nodes are places
where information can be, and whose edges are possible information flows.

Node kinds (the node name encodes the kind and, if relevant, the session):

``out:<tool>``          output of a tool. Source node; carries the tool's label.
``out:<tool>#trusted``  projection of a tool output on its trusted fields .
``usr:<principal>``     prompt of a principal (user). Source node .
``in:<tool>``           input of a sink tool. Terminal node (a "sink").
``ctx:<agent>@<s>``     context of an agent during session ``s``.
``mem:<memory>@<s>``    content of a memory store during session ``s``.

Why unroll over sessions? A single session cannot show attacks that are
planted in one session and triggered in a later one (memory poisoning).
We therefore make one copy of every agent context and memory store per
session ``s = 1..T``. The only edges that link session ``s`` to ``s+1`` are
``persist`` edges of persistent memory stores. Agent contexts never cross
sessions, which matches how agent frameworks reset the conversation.

Every graph edge keeps the id of the edge template it was created from, in
the edge attribute ``template``. A policy gates templates, so one gate acts
on all session copies of the same flow.

Heterogeneous sessions
-----------------------------
If the manifest declares principals, session ``s`` belongs to the principal
``controls.principal_of(s)``. Only that principal's ``prompt`` edges exist in
session ``s``. A ``persist`` edge between two sessions of *different*
principals carries the attribute ``adds={"S"}``: what one principal left in
a persistent memory is confidential for the next one (tenant isolation).
The verifier adds this label after the gate check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from .labels import SENSITIVE
from .manifest import EdgeTemplate, Manifest


# --- Node naming helpers ----------------------------------------------------


def ctx(agent: str, session: int) -> str:
    return f"ctx:{agent}@{session}"


def mem(memory: str, session: int) -> str:
    return f"mem:{memory}@{session}"


def out(tool: str) -> str:
    return f"out:{tool}"


def inp(tool: str) -> str:
    return f"in:{tool}"


def usr(principal: str) -> str:
    return f"usr:{principal}"


def node_session(node: str) -> Optional[int]:
    """Session number encoded in a node name, or ``None`` for tool nodes."""
    return int(node.rsplit("@", 1)[1]) if "@" in node else None


# --- Compiled graph ---------------------------------------------------------


@dataclass
class CompiledGraph:
    manifest: Manifest
    graph: nx.DiGraph
    templates: dict[str, EdgeTemplate]
    sessions: int

    def instances(self, template_id: str) -> list[tuple[str, str]]:
        """All graph edges (one per session, usually) created from a template."""
        return [(u, v) for u, v, t in self.graph.edges(data="template") if t == template_id]

    def has_cross_session_edges(self) -> bool:
        """``True`` if some persistent memory links two sessions, or if sessions
        differ by principal (then the session copies are not identical)."""
        if any(k == "persist" for _, _, k in self.graph.edges(data="kind")):
            return True
        return len(set(self.manifest.controls.session_principals)) > 1

    def stats(self) -> dict[str, int]:
        """Size of the graph, reported in the thesis (scalability)."""
        kinds: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            kinds[data["kind"]] = kinds.get(data["kind"], 0) + 1
        return {
            "sessions": self.sessions,
            "templates": len(self.templates),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            **{f"nodes_{k}": v for k, v in sorted(kinds.items())},
        }


def compile_manifest(manifest: Manifest, sessions: Optional[int] = None) -> CompiledGraph:
    """Build G_T from a validated manifest.

    ``sessions`` overrides ``manifest.controls.sessions`` (useful for tests and
    for the scalability experiments).
    """
    T = sessions if sessions is not None else manifest.controls.sessions
    if T < 1:
        raise ValueError("sessions must be >= 1")
    g = nx.DiGraph()
    templates = manifest.edge_templates()

    # 1. Tool nodes (shared by all sessions: a tool is the same object in
    #    every session; only agent contexts and memories are per session).
    for tool in manifest.tools:
        if tool.output is not None:
            g.add_node(out(tool.id), kind="out", entity=tool.id,
                       label=manifest.output_label(tool.id))
            if tool.output.has_trusted_view:
                view = f"{tool.id}#trusted"
                g.add_node(out(view), kind="out", entity=view,
                           label=manifest.output_label(view))
        if tool.is_sink:
            g.add_node(inp(tool.id), kind="in", entity=tool.id,
                       integrity_sink=tool.integrity_critical,
                       confidentiality_sink=tool.is_confidentiality_sink)

    for principal in manifest.principals:
        g.add_node(usr(principal.id), kind="out", entity=principal.id,
                   label=manifest.principal_label(principal.id), principal=True)

    # 2. Per-session nodes.
    for s in range(1, T + 1):
        for agent in manifest.agents:
            g.add_node(ctx(agent.id, s), kind="ctx", entity=agent.id, session=s)
        for m in manifest.memory:
            g.add_node(mem(m.id, s), kind="mem", entity=m.id, session=s)

    # 3. Edges: one instance of each template per session (persist edges
    #    link session s to s+1, so they have T-1 instances).
    def add_edge(u: str, v: str, template: EdgeTemplate, adds: frozenset = frozenset()) -> None:
        g.add_edge(u, v, template=template.id, kind=template.kind, adds=adds)

    who = manifest.controls.principal_of

    for tpl in templates.values():
        for s in range(1, T + 1):
            if tpl.kind == "ret":
                add_edge(out(tpl.src), ctx(tpl.dst, s), tpl)
            elif tpl.kind == "call":
                add_edge(ctx(tpl.src, s), inp(tpl.dst), tpl)
            elif tpl.kind == "mw":
                add_edge(ctx(tpl.src, s), mem(tpl.dst, s), tpl)
            elif tpl.kind == "mr":
                add_edge(mem(tpl.src, s), ctx(tpl.dst, s), tpl)
            elif tpl.kind == "msg":
                add_edge(ctx(tpl.src, s), ctx(tpl.dst, s), tpl)
            elif tpl.kind == "persist":
                if s < T:
                    # Crossing to another principal's session: the content
                    # becomes confidential for the new principal.
                    adds = frozenset({SENSITIVE}) if who(s) != who(s + 1) else frozenset()
                    add_edge(mem(tpl.src, s), mem(tpl.src, s + 1), tpl, adds)
            elif tpl.kind == "prompt":
                if who(s) == tpl.src:  # only the session's own principal speaks
                    add_edge(usr(tpl.src), ctx(tpl.dst, s), tpl)
            else:  # pragma: no cover - EDGE_KINDS is closed
                raise ValueError(f"unknown edge kind {tpl.kind}")

    return CompiledGraph(manifest=manifest, graph=g, templates=templates, sessions=T)
