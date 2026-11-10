"""Declarative security manifest: schema, validation and edge templates.

The manifest is the single input that a developer writes to describe a
multi-agent system. It has six sections (More clear whitin Chapter 4 of the thesis):

``data_classes``  kinds of data and how sensitive they are
``tools``         what each tool returns and where its input goes
``memory``        memory stores and whether they persist across sessions
``agents``        which tools, memories and other agents each agent may use
``tasks``         what the system must still be able to do (utility)
``controls``      synthesis parameters: horizon, costs, non-gateable edges

From the manifest we derive *edge templates*. An edge template is one
possible information flow, independent of the session in which it happens,
for example ``call:executor->send_email``. The policy assigns gate modes to
edge templates, so the same policy applies in every session.

Edge-template identifiers follow a fixed grammar::

    ret:<tool>-><agent>        a tool's output enters the agent's context
    ret:<tool>#trusted-><agent>  only the trusted fields of the output enter
                               the context (projection, v0.5)
    prompt:<principal>-><agent>  the session's user (principal) talks to an
                               entry agent (heterogeneous sessions, v0.5)
    call:<agent>-><tool>       the agent's context flows into a tool's input
    mw:<agent>-><memory>       the agent writes to a memory store
    mr:<memory>-><agent>       the agent reads from a memory store
    msg:<agent>-><agent>       one agent sends a message to another
    persist:<memory>           memory entries survive to the next session

Pydantic is used for parsing and validation. ``extra="forbid"`` makes any
unknown key an error, so a misspelt field in the YAML is reported instead
of being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .labels import SENSITIVE, UNTRUSTED, Label, Mode, make_label

#: All edge kinds, in the order used for display.
EDGE_KINDS = ("ret", "call", "mw", "mr", "msg", "persist", "prompt")

#: Separator between a tool id and a view of its output, e.g.
#: ``get_most_recent_transactions#trusted``.
VIEW_SEP = "#"
#: Name of the projected view that keeps only the trusted fields.
TRUSTED_VIEW = "trusted"


def split_view(name: str) -> tuple[str, Optional[str]]:
    """``"tool#trusted"`` -> ``("tool", "trusted")``; ``"tool"`` -> ``("tool", None)``."""
    if VIEW_SEP in name:
        tool, view = name.split(VIEW_SEP, 1)
        return tool, view
    return name, None


def _strip_v05_defaults(data: Any) -> Any:
    """Remove v0.5 keys that still have their default value.

    Fingerprints (manifest and tool descriptors) are computed on this
    stripped form, so a manifest that does not use any v0.5 feature keeps
    exactly the fingerprint it had in v0.4, and policies saved with v0.4
    remain valid.
    """
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k in ("fields", "principals", "session_principals") and v == []:
                continue
            if k == "allow_human" and v is False:
                continue
            if k == "mode_weights" and isinstance(v, dict) and v.get("H") == 0.8:
                v = {kk: vv for kk, vv in v.items() if kk != "H"}
            if k == "costs" and isinstance(v, dict) and v.get("prompt") == 4.0:
                v = {kk: vv for kk, vv in v.items() if kk != "prompt"}
            out[k] = _strip_v05_defaults(v)
        return out
    if isinstance(data, list):
        return [_strip_v05_defaults(x) for x in data]
    return data


def _fingerprint(model: BaseModel) -> str:
    canonical = json.dumps(_strip_v05_defaults(model.model_dump(mode="json")), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    """Base class: reject unknown fields and make instances immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Section models ---------------------------------------------------------


class DataClass(StrictModel):
    id: str
    sensitivity: Literal["public", "internal", "confidential"]
    description: str = ""


class OutputField(StrictModel):
    """One field of a structured tool output (field-level labels, v0.5).

    Example: a bank transaction has the fields ``amount`` and ``date``
    (produced by the bank, trusted) and ``subject`` (written by the sender,
    untrusted). Declaring fields lets the runtime deliver a *projection* of
    the output that keeps only the trusted fields.
    """

    id: str
    trust: Literal["trusted", "untrusted"]
    data_classes: list[str] = Field(default_factory=list)


class ToolOutput(StrictModel):
    """What a tool returns to the agent that called it.

    Either a single ``trust`` value for the whole output (v0.1 style), or a
    list of ``fields`` with one trust value each (field-level labels). With
    fields, the output as a whole is untrusted if any field is untrusted,
    and its data classes are the union of the fields' data classes.
    """

    #: ``untrusted`` if an attacker may control (part of) the output.
    trust: Optional[Literal["trusted", "untrusted"]] = None
    #: Data classes that may appear in the output.
    data_classes: list[str] = Field(default_factory=list)
    #: Optional field-level description (enables the ``#trusted`` view).
    fields: list[OutputField] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ToolOutput":
        if not self.fields and self.trust is None:
            raise ValueError("tool output: give 'trust' or a list of 'fields'")
        if self.fields:
            derived = "untrusted" if any(f.trust == "untrusted" for f in self.fields) else "trusted"
            if self.trust is not None and self.trust != derived:
                raise ValueError(f"tool output: 'trust: {self.trust}' contradicts its fields "
                                 f"(which make it {derived})")
        return self

    @property
    def overall_trust(self) -> str:
        if self.fields:
            return "untrusted" if any(f.trust == "untrusted" for f in self.fields) else "trusted"
        return self.trust  # type: ignore[return-value]

    @property
    def all_data_classes(self) -> list[str]:
        out = list(self.data_classes)
        for f in self.fields:
            out += [d for d in f.data_classes if d not in out]
        return out

    @property
    def trusted_fields(self) -> list[str]:
        return [f.id for f in self.fields if f.trust == "trusted"]

    @property
    def has_trusted_view(self) -> bool:
        """A projection is useful only if it removes something and keeps something."""
        trusts = {f.trust for f in self.fields}
        return trusts == {"trusted", "untrusted"}


class Tool(StrictModel):
    id: str
    description: str = ""
    #: JSON-schema-like description of the arguments, as shown to the LLM.
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: ``None`` if the tool returns nothing useful to the agent.
    output: Optional[ToolOutput] = None
    #: Where the tool's *input* goes. ``external`` means it leaves the trust
    #: boundary (a search query, an outgoing e-mail): a possible exfiltration
    #: channel. ``local``/``internal`` stay inside the boundary.
    input_destination: Literal["local", "internal", "external"] = "local"
    #: ``True`` if calling the tool changes state that matters
    #: (sending, deleting, paying, updating a record).
    integrity_critical: bool = False

    @property
    def is_sink(self) -> bool:
        """A tool is a sink if its input leaves the boundary or it changes state."""
        return self.input_destination == "external" or self.integrity_critical

    @property
    def is_confidentiality_sink(self) -> bool:
        return self.input_destination == "external"

    def descriptor_sha256(self) -> str:
        """SHA-256 fingerprint of the full tool descriptor.

        Any change to the description, parameters or declared behaviour
        changes the fingerprint. The runtime engine refuses a tool whose
        fingerprint differs from the approved one (protection against
        "rug pull" attacks on tool descriptors).
        """
        return _fingerprint(self)


class Memory(StrictModel):
    id: str
    #: ``session``: wiped at the end of every session.
    #: ``persistent``: entries survive to later sessions.
    persistence: Literal["session", "persistent"] = "session"
    description: str = ""


class Principal(StrictModel):
    """A user (or other caller) who opens sessions (heterogeneous sessions, v0.5).

    ``trust: untrusted`` models a caller who may be the attacker (an
    external customer, a public chat user): their prompt enters the entry
    agents' context with label U (direct prompt injection). Data written to
    persistent memory during one principal's session is *sensitive* for
    every other principal (tenant isolation).
    """

    id: str
    trust: Literal["trusted", "untrusted"] = "trusted"
    #: Agents that receive this principal's prompt at the start of a session.
    entry_agents: list[str] = Field(default_factory=list)
    description: str = ""


class MemoryAccess(StrictModel):
    read: list[str] = Field(default_factory=list)
    write: list[str] = Field(default_factory=list)


class Agent(StrictModel):
    id: str
    role: str = ""
    tools: list[str] = Field(default_factory=list)
    memory: MemoryAccess = Field(default_factory=MemoryAccess)
    #: Agents this agent may delegate to. Delegation creates messages in both
    #: directions (request and response).
    delegates_to: list[str] = Field(default_factory=list)


class Requirement(StrictModel):
    """One edge a task needs, and the label under which it must still work.

    Example: ``{edge: "call:executor->send_email", under: ["S"]}`` means the
    task must be able to send an e-mail whose content contains sensitive
    data. A gate that blocks ``{S}`` on that edge would break the task.
    """

    edge: str
    under: list[Literal["U", "S"]] = Field(default_factory=list)

    @property
    def label(self) -> Label:
        return make_label(self.under)


class Task(StrictModel):
    id: str
    description: str = ""
    requires: list[Requirement]


def _default_costs() -> dict[str, float]:
    # Default utility cost of gating one edge, per edge kind. Tool calls and
    # tool returns are more expensive to gate than memory accesses because
    # they remove more visible functionality. These are defaults only; the
    # thesis evaluation varies them (sensitivity analysis).
    return {"ret": 2.0, "call": 3.0, "mw": 1.0, "mr": 1.0, "msg": 2.0, "persist": 1.0,
            "prompt": 4.0}


def _default_weights() -> dict[Mode, float]:
    # A REMOVE gate costs the full edge cost; conditional gates cost less,
    # because they only act in tainted sessions. A human-confirmation gate
    # keeps the task working but costs a human interruption in tainted
    # sessions: we price it above I by default (a modelling choice that the
    # sensitivity analysis varies).
    return {Mode.REMOVE: 1.0, Mode.INTEGRITY: 0.6, Mode.TRIFECTA: 0.3, Mode.HUMAN: 0.8}


class Controls(StrictModel):
    #: Number of sessions T over which the graph is unrolled. T >= 2 is
    #: needed to see attacks that cross sessions through persistent memory.
    sessions: int = Field(default=2, ge=1, le=20)
    #: Data-class sensitivities that count as "S".
    sensitive_levels: list[Literal["public", "internal", "confidential"]] = Field(
        default_factory=lambda: ["confidential"]
    )
    costs: dict[str, float] = Field(default_factory=_default_costs)
    mode_weights: dict[Mode, float] = Field(default_factory=_default_weights)
    #: Per-edge cost overrides, e.g. ``{"msg:supervisor->executor": 5}``.
    overrides: dict[str, float] = Field(default_factory=dict)
    #: Edges on which no gate can be placed (no enforcement point exists).
    fixed: list[str] = Field(default_factory=list)
    #: Allow human-confirmation gates (mode H) on call edges. Off by default,
    #: so that results computed without this option do not change.
    allow_human: bool = False
    #: Principal of each session, in order (heterogeneous sessions). If the
    #: list is shorter than ``sessions`` it is repeated. Empty: no principal.
    session_principals: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _default_human_weight(cls, data: Any) -> Any:
        # Manifests written before v0.5 give weights for R, I and C only.
        if isinstance(data, dict) and isinstance(data.get("mode_weights"), dict):
            w = dict(data["mode_weights"])
            if not any(str(k) in ("H", "Mode.HUMAN") or k is Mode.HUMAN for k in w):
                w["H"] = _default_weights()[Mode.HUMAN]
            data = {**data, "mode_weights": w}
        return data

    def principal_of(self, session: int) -> Optional[str]:
        """Principal of session ``session`` (1-based), or ``None``."""
        if not self.session_principals:
            return None
        return self.session_principals[(session - 1) % len(self.session_principals)]

    @model_validator(mode="after")
    def _check(self) -> "Controls":
        unknown = set(self.costs) - set(EDGE_KINDS)
        if unknown:
            raise ValueError(f"controls.costs: unknown edge kind(s) {sorted(unknown)}")
        if any(v <= 0 for v in self.costs.values()) or any(
            v <= 0 for v in self.overrides.values()
        ):
            raise ValueError("controls: costs and overrides must be > 0")
        w = {m: self.mode_weights.get(m) for m in Mode}
        if None in w.values():
            raise ValueError("controls.mode_weights must define R, I and C")
        # Keep the cost ordering consistent with the blocking power:
        # R blocks more than I, which blocks more than C.
        if not (w[Mode.REMOVE] >= w[Mode.INTEGRITY] >= w[Mode.TRIFECTA] > 0):
            raise ValueError("controls.mode_weights must satisfy R >= I >= C > 0")
        if not w[Mode.HUMAN] > 0:
            raise ValueError("controls.mode_weights: H must be > 0")
        return self


# --- Edge templates ---------------------------------------------------------


@dataclass(frozen=True)
class EdgeTemplate:
    """A session-independent information flow derived from the manifest."""

    id: str  # e.g. "call:executor->send_email"
    kind: str  # one of EDGE_KINDS
    src: str  # id of the source entity (agent, tool or memory)
    dst: str  # id of the destination entity


# --- Root model -------------------------------------------------------------


class Manifest(StrictModel):
    version: Literal[1] = 1
    system: str
    description: str = ""
    data_classes: list[DataClass]
    tools: list[Tool]
    memory: list[Memory] = Field(default_factory=list)
    agents: list[Agent]
    tasks: list[Task] = Field(default_factory=list)
    principals: list[Principal] = Field(default_factory=list)
    controls: Controls = Field(default_factory=Controls)

    # -- lookup helpers -------------------------------------------------

    def tool(self, tool_id: str) -> Tool:
        """Tool by id (a view suffix such as ``#trusted`` is ignored)."""
        return {t.id: t for t in self.tools}[split_view(tool_id)[0]]

    def principal(self, principal_id: str) -> Principal:
        return {p.id: p for p in self.principals}[principal_id]

    def principal_label(self, principal_id: Optional[str]) -> Label:
        """Label of a principal's prompt: {U} if the principal is untrusted."""
        if principal_id is None:
            return frozenset()
        return frozenset({UNTRUSTED}) if self.principal(principal_id).trust == "untrusted" \
            else frozenset()

    def projected_fields(self, tool_id: str) -> list[str]:
        """Fields kept by the ``#trusted`` view of a tool (empty if none)."""
        tool = self.tool(tool_id)
        if tool.output is None or not tool.output.has_trusted_view:
            return []
        return tool.output.trusted_fields

    def agent(self, agent_id: str) -> Agent:
        return {a.id: a for a in self.agents}[agent_id]

    def memory_store(self, memory_id: str) -> Memory:
        return {m.id: m for m in self.memory}[memory_id]

    def output_label(self, tool_id: str) -> Label:
        """Label carried by the output of ``tool_id`` (may be empty).

        ``tool_id`` may name a view: ``"tool#trusted"`` gives the label of the
        projection that keeps only the trusted fields.
        """
        base, view = split_view(tool_id)
        tool = self.tool(base)
        if tool.output is None:
            return frozenset()
        levels = {d.id: d.sensitivity for d in self.data_classes}
        sensitive = lambda classes: any(levels[d] in self.controls.sensitive_levels  # noqa: E731
                                        for d in classes)
        label = set()
        if view == TRUSTED_VIEW:
            kept = [f for f in tool.output.fields if f.trust == "trusted"]
            if sensitive(tool.output.data_classes) or any(sensitive(f.data_classes) for f in kept):
                label.add(SENSITIVE)
            return frozenset(label)
        if view is not None:
            raise KeyError(f"unknown view '{view}' of tool '{base}'")
        if tool.output.overall_trust == "untrusted":
            label.add(UNTRUSTED)
        if sensitive(tool.output.all_data_classes):
            label.add(SENSITIVE)
        return frozenset(label)

    def sha256(self) -> str:
        """Fingerprint of the whole manifest (stored in the policy file)."""
        return _fingerprint(self)

    def edge_templates(self) -> dict[str, EdgeTemplate]:
        """Derive every possible information flow declared by the manifest.

        Returns a dictionary ``{edge_id: EdgeTemplate}`` sorted by id, so that
        iteration order (and therefore every downstream result) is
        deterministic.
        """
        edges: dict[str, EdgeTemplate] = {}

        def add(kind: str, src: str, dst: str) -> None:
            edge_id = f"{kind}:{src}" if kind == "persist" else f"{kind}:{src}->{dst}"
            edges[edge_id] = EdgeTemplate(edge_id, kind, src, dst)

        for agent in self.agents:
            for tool_id in agent.tools:
                tool = self.tool(tool_id)
                if tool.output is not None:
                    add("ret", tool_id, agent.id)  # tool output -> agent
                    if tool.output.has_trusted_view:  # projection (trusted fields only)
                        add("ret", f"{tool_id}{VIEW_SEP}{TRUSTED_VIEW}", agent.id)
                if tool.is_sink:
                    add("call", agent.id, tool_id)  # agent context -> tool input
            for mem_id in agent.memory.read:
                add("mr", mem_id, agent.id)
            for mem_id in agent.memory.write:
                add("mw", agent.id, mem_id)
            for other in agent.delegates_to:
                add("msg", agent.id, other)  # request
                add("msg", other, agent.id)  # response
        for mem in self.memory:
            if mem.persistence == "persistent":
                add("persist", mem.id, mem.id)
        for principal in self.principals:
            for agent_id in principal.entry_agents:
                add("prompt", principal.id, agent_id)  # user prompt -> entry agent
        return dict(sorted(edges.items()))

    def edge_cost(self, edge_id: str, mode: Mode) -> float:
        """Utility cost of placing a gate in ``mode`` on ``edge_id``."""
        kind = edge_id.split(":", 1)[0]
        base = self.controls.overrides.get(edge_id, self.controls.costs.get(kind, 1.0))
        return base * self.controls.mode_weights[mode]

    # -- validation -----------------------------------------------------

    @model_validator(mode="after")
    def _check_references(self) -> "Manifest":
        errors: list[str] = []

        # 1. Identifiers must be unique across all sections, because they are
        #    used as node names in the graph.
        seen: dict[str, str] = {}
        for section, items in (
            ("data_classes", self.data_classes),
            ("tools", self.tools),
            ("memory", self.memory),
            ("agents", self.agents),
            ("tasks", self.tasks),
            ("principals", self.principals),
        ):
            for item in items:
                if item.id in seen:
                    errors.append(f"duplicate id '{item.id}' (in {seen[item.id]} and {section})")
                seen[item.id] = section
                if not item.id or any(c in item.id for c in ":@-> #"):
                    errors.append(f"invalid id '{item.id}': must not contain ':', '@', '-', '>', "
                                  f"'#' or spaces")

        data_ids = {d.id for d in self.data_classes}
        tool_ids = {t.id for t in self.tools}
        mem_ids = {m.id for m in self.memory}
        agent_ids = {a.id for a in self.agents}

        # 2. Every reference must point to something declared.
        for tool in self.tools:
            if tool.output:
                for d in tool.output.all_data_classes:
                    if d not in data_ids:
                        errors.append(f"tool '{tool.id}': unknown data class '{d}'")
                names = [f.id for f in tool.output.fields]
                if len(names) != len(set(names)):
                    errors.append(f"tool '{tool.id}': duplicate field names")
        principal_ids = {p.id for p in self.principals}
        for principal in self.principals:
            for a in principal.entry_agents:
                if a not in {ag.id for ag in self.agents}:
                    errors.append(f"principal '{principal.id}': unknown entry agent '{a}'")
        for pid in self.controls.session_principals:
            if pid not in principal_ids:
                errors.append(f"controls.session_principals: unknown principal '{pid}'")
        if self.principals and not self.controls.session_principals:
            errors.append("principals are declared: list them in controls.session_principals")
        for agent in self.agents:
            for t in agent.tools:
                if t not in tool_ids:
                    errors.append(f"agent '{agent.id}': unknown tool '{t}'")
            for m in agent.memory.read + agent.memory.write:
                if m not in mem_ids:
                    errors.append(f"agent '{agent.id}': unknown memory '{m}'")
            for b in agent.delegates_to:
                if b not in agent_ids:
                    errors.append(f"agent '{agent.id}': unknown delegate '{b}'")
                if b == agent.id:
                    errors.append(f"agent '{agent.id}': cannot delegate to itself")

        if errors:
            raise ValueError("Invalid manifest:\n  - " + "\n  - ".join(errors))

        # 3. Edge ids used in tasks and controls must exist. This check needs
        #    the edge templates, which are only valid once 1-2 passed.
        edge_ids = set(self.edge_templates())
        for task in self.tasks:
            for req in task.requires:
                if req.edge not in edge_ids:
                    errors.append(f"task '{task.id}': unknown edge '{req.edge}'")
        for e in list(self.controls.overrides) + self.controls.fixed:
            if e not in edge_ids:
                errors.append(f"controls: unknown edge '{e}'")
        if errors:
            raise ValueError("Invalid manifest:\n  - " + "\n  - ".join(errors))
        return self


# --- Loading ----------------------------------------------------------------


def load_manifest(path: str | Path) -> Manifest:
    """Read a YAML manifest from disk and validate it."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Manifest.model_validate(data)


def manifest_from_dict(data: dict) -> Manifest:
    """Validate a manifest given as a Python dictionary (used by mutations)."""
    return Manifest.model_validate(data)
