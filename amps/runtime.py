"""Runtime enforcement engine (policy points P1-P4).

The engine sits between the agents and everything they touch. The
orchestrator (LangGraph in the demonstrator) must call the engine *before*
performing an action and must respect the returned :class:`Decision`.
The engine never looks at the content of prompts or tool outputs: it only
tracks labels, so its decisions do not depend on an LLM and cannot be
talked out of by a prompt injection.

Policy enforcement points
-------------------------
P0  user-prompt gate     :meth:`PolicyEngine.user_prompt`   (principals)
P1  tool gate            :meth:`PolicyEngine.before_tool_call`
P2  memory write gate    :meth:`PolicyEngine.memory_write`
P3  memory read gate     :meth:`PolicyEngine.memory_read`
P4  inter-agent gate     :meth:`PolicyEngine.send_message`

Plus two lifecycle hooks: :meth:`start_session` and :meth:`end_session`
(the latter applies ``persist`` gates to persistent memory).

Label tracking
--------------
* Each agent context has a label that only grows during a session
  (session-level conservative labelling). It is reset at session start.
* Each memory entry stores the label of the context that wrote it (origin
  bound at write time); a read adds the labels of the returned entries to
  the reader's context.

Default deny
------------
Any flow that is not declared in the manifest (an undeclared tool, memory
or delegation) is denied, whatever the policy says. A tool whose descriptor
fingerprint differs from the approved one is denied as well.

Human confirmation 
-------------------------
On an edge gated in mode ``H``, a tool call made from an untrusted context is
not refused: the engine calls ``confirm(agent, tool, label)``, a function
supplied by the application (a dialog box, a chat message to the user...).
The call proceeds only if it returns ``True``. Without a ``confirm`` function
the engine refuses (fail-safe: no human, no action). Every request is
counted in ``self.confirmations``: the number of interruptions is the cost
of this mode for the user.

Field-level views
------------------------
If a tool declares output fields, the engine can deliver a *projection* that
keeps only the trusted fields (view ``#trusted``). The view used for a call
is stored in the decision (``details["view"]`` and ``details["keep_fields"]``);
:func:`project_output` applies it to the actual tool result.

Every decision is appended to ``self.trace`` (and optionally to a JSONL
file) so that each experiment can be audited afterwards.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .labels import (
    EMPTY,
    SENSITIVE,
    Label,
    Mode,
    blocks,
    confidentiality_violation,
    fmt,
    integrity_violation,
    needs_confirmation,
)
from .manifest import TRUSTED_VIEW, VIEW_SEP, Manifest, Tool, split_view

#: Signature of a human-confirmation callback: (agent, tool, label) -> approve?
ConfirmFn = Callable[[str, str, Label], bool]
from .policy import Policy


@dataclass
class Decision:
    """Outcome of one check by the engine."""

    allowed: bool
    point: str  # "P1", "P2", "P3", "P4" or "LIFECYCLE"
    action: str  # human-readable action, e.g. "call send_email"
    agent: Optional[str]
    edge: Optional[str]  # edge template id checked, if any
    mode: Optional[str]  # gate mode on that edge, if any
    label: str  # label of the flow checked, e.g. "{S,U}"
    reason: str  # "allowed", "gate", "undeclared", "descriptor_mismatch", ...
    session: int
    seq: int  # position in the trace
    elapsed_us: float = 0.0  # time spent in the check (overhead metric)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    content: Any
    label: Label
    writer: str
    session: int
    principal: Optional[str] = None


@dataclass
class ConfirmationRequest:
    """One request sent to the human by a gate in mode H."""

    agent: str
    tool: str
    label: str
    approved: bool
    session: int
    seq: int

@dataclass
class SinkEvent:
    """A sink was invoked with a label that violates a property.

    With a correct policy this list stays empty. In experiments without a
    policy (configuration A) it measures successful attacks at the label
    level (T1 tier).
    """

    tool: str
    agent: str
    label: str
    properties: list[str]
    session: int
    seq: int


class PolicyEngine:
    def __init__(self, manifest: Manifest, policy: Policy,
                 trace_path: Optional[str | Path] = None, strict_manifest: bool = True,
                 confirm: Optional[ConfirmFn] = None, prefer_projection: bool = False):
        """
        ``confirm``: human-confirmation callback for gates in mode H (default:
        refuse). ``prefer_projection``: deliver the ``#trusted`` view of a tool
        whenever it exists and is allowed, even if the full output is allowed
        too (use it when the agent never needs the untrusted fields).
        """
        if strict_manifest and policy.manifest_sha256 and policy.manifest_sha256 != manifest.sha256():
            raise ValueError("policy was synthesized for a different manifest "
                             "(run the impact analysis and re-synthesize)")
        self.manifest = manifest
        self.policy = policy
        self.edges = manifest.edge_templates()
        self.trace: list[Decision] = []
        self.sink_events: list[SinkEvent] = []
        self.trace_path = Path(trace_path) if trace_path else None
        self.session = 0
        self.ctx_labels: dict[str, Label] = {}
        self.memory: dict[str, list[MemoryEntry]] = {m.id: [] for m in manifest.memory}
        self.quarantine: list[MemoryEntry] = []
        self.confirm = confirm
        self.prefer_projection = prefer_projection
        self.confirmations: list[ConfirmationRequest] = []
        self.principal: Optional[str] = None
        self._view: dict[tuple[str, str], Optional[str]] = {}
        self.start_session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, decision: Decision) -> Decision:
        self.trace.append(decision)
        if self.trace_path:
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(decision)) + "\n")
        return decision

    def _decide(self, t0: float, point: str, action: str, agent: Optional[str],
                edge: Optional[str], label: Label, allowed: bool, reason: str,
                **details: Any) -> Decision:
        mode = self.policy.mode(edge) if edge else None
        return self._record(Decision(
            allowed=allowed, point=point, action=action, agent=agent, edge=edge,
            mode=mode.value if mode else None, label=fmt(label), reason=reason,
            session=self.session, seq=len(self.trace),
            elapsed_us=(time.perf_counter() - t0) * 1e6, details=details))

    def _gate(self, edge: str, label: Label, agent: str = "", tool: str = ""
              ) -> tuple[bool, str]:
        """Check one declared edge. Returns (allowed, reason)."""
        if edge not in self.edges:
            return False, "undeclared"
        mode = self.policy.mode(edge)
        if needs_confirmation(mode, label):
            approved = bool(self.confirm(agent, tool, label)) if self.confirm else False
            self.confirmations.append(ConfirmationRequest(
                agent, tool, fmt(label), approved, self.session, len(self.trace)))
            return (True, "confirmed") if approved else (False, "rejected_by_human")
        if blocks(mode, label):
            return False, "gate"
        return True, "allowed"

    def label_of(self, agent: str) -> Label:
        return self.ctx_labels.get(agent, EMPTY)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a new session: every agent context starts clean."""
        self.session += 1
        self.ctx_labels = {a.id: EMPTY for a in self.manifest.agents}
        self.principal = self.manifest.controls.principal_of(self.session)
        self._view = {}

    def end_session(self) -> None:
        """Close the session.

        * Session-scoped memories are wiped.
        * Persistent memories keep their entries, except those whose label is
          blocked by the ``persist:<memory>`` gate: these are moved to
          quarantine (kept for audit, never readable by agents).
        """
        t0 = time.perf_counter()
        for mem in self.manifest.memory:
            entries = self.memory[mem.id]
            if mem.persistence == "session":
                self.memory[mem.id] = []
                continue
            edge = f"persist:{mem.id}"
            kept, dropped = [], []
            for e in entries:
                (dropped if blocks(self.policy.mode(edge), e.label) else kept).append(e)
            # Tenant isolation: entries left by another principal are
            # confidential for the principal of the next session.
            nxt = self.manifest.controls.principal_of(self.session + 1)
            if nxt != self.principal:
                for e in kept:
                    e.label = e.label | {SENSITIVE}
            self.memory[mem.id] = kept
            self.quarantine.extend(dropped)
            if dropped:
                self._decide(t0, "LIFECYCLE", f"persist {mem.id}", None, edge, EMPTY, True,
                             "quarantined", quarantined=len(dropped))
        self.start_session()

    # ------------------------------------------------------------------
    # P0 - user prompt gate (principals)
    # ------------------------------------------------------------------

    def user_prompt(self, agent: str) -> Decision:
        """P0: the session's principal sends its request to ``agent``.

        If allowed, the agent's context takes the principal's label ({U} for
        an untrusted principal: direct prompt injection is possible).
        """
        t0 = time.perf_counter()
        if self.principal is None:
            return self._decide(t0, "P0", f"prompt {agent}", agent, None, EMPTY, True,
                                "no_principal")
        label = self.manifest.principal_label(self.principal)
        edge = f"prompt:{self.principal}->{agent}"
        ok, reason = self._gate(edge, label, agent)
        if ok:
            self.ctx_labels[agent] = self.label_of(agent) | label
        return self._decide(t0, "P0", f"prompt {agent}", agent, edge, label, ok, reason,
                            principal=self.principal)

    # ------------------------------------------------------------------
    # P1 - tool gate
    # ------------------------------------------------------------------

    def before_tool_call(self, agent: str, tool: str,
                         descriptor: Optional[Tool | dict] = None,
                         view: Optional[str] = None) -> Decision:
        """Check whether ``agent`` may call ``tool`` now.

        ``descriptor`` is the tool descriptor as currently exposed by the
        tool server (MCP or local). If given, its fingerprint must match the
        approved registry entry.

        ``view`` asks for a view of the output (``"trusted"``); ``tool`` may
        also be written ``"tool#trusted"``. Without a view, the engine
        delivers the full output if its return edge allows it, otherwise the
        trusted projection if that one is allowed (or always the projection
        when ``prefer_projection`` is set).
        """
        t0 = time.perf_counter()
        base, named_view = split_view(tool)
        view = view or named_view
        tool = base
        label = self.label_of(agent)
        action = f"call {tool}"
        try:
            agent_decl = self.manifest.agent(agent)
        except KeyError:
            return self._decide(t0, "P1", action, agent, None, label, False, "unknown_agent")
        if tool not in agent_decl.tools:
            return self._decide(t0, "P1", action, agent, None, label, False, "undeclared")

        if descriptor is not None:
            if isinstance(descriptor, dict):
                descriptor = Tool.model_validate(descriptor)
            if descriptor.descriptor_sha256() != self.policy.registry.get(tool):
                return self._decide(t0, "P1", action, agent, None, label, False,
                                    "descriptor_mismatch")

        decl = self.manifest.tool(tool)
        # (a) Does the agent's context flow into the tool's input?
        reason_call = "allowed"
        if decl.is_sink:
            edge = f"call:{agent}->{tool}"
            ok, reason_call = self._gate(edge, label, agent, tool)
            if not ok:
                return self._decide(t0, "P1", action, agent, edge, label, False, reason_call)
        # (b) Would the tool's output (or a view of it) be accepted into the context?
        chosen_view: Optional[str] = None
        if decl.output is not None:
            ok, ret_edge, out_label, reason, chosen_view = self._choose_view(agent, tool, view)
            if not ok:
                return self._decide(t0, "P1", action, agent, ret_edge, out_label, False, reason)
        self._view[(agent, tool)] = chosen_view

        # Allowed. If this is a sink invoked with a violating label, record it
        # (this can only happen when the policy is incomplete or absent, or
        # when a human approved the action).
        violated = []
        if decl.integrity_critical and integrity_violation(label):
            violated.append("integrity")
        if decl.is_confidentiality_sink and confidentiality_violation(label):
            violated.append("confidentiality")
        if violated:
            self.sink_events.append(SinkEvent(tool, agent, fmt(label), violated,
                                              self.session, len(self.trace)))
        edge = f"call:{agent}->{tool}" if decl.is_sink else self._ret_edge(agent, tool, chosen_view)
        details: dict[str, Any] = {"violated": violated}
        if chosen_view:
            details.update(view=chosen_view, keep_fields=self.manifest.projected_fields(tool))
        reason = "confirmed" if reason_call == "confirmed" else "allowed"
        return self._decide(t0, "P1", action, agent, edge, label, True, reason, **details)

    @staticmethod
    def _ret_edge(agent: str, tool: str, view: Optional[str]) -> str:
        return f"ret:{tool}{VIEW_SEP}{view}->{agent}" if view else f"ret:{tool}->{agent}"

    def _choose_view(self, agent: str, tool: str, view: Optional[str]):
        """Pick the full output or the trusted projection. Returns
        (allowed, edge, label, reason, view)."""
        has_view = bool(self.manifest.projected_fields(tool))
        candidates: list[Optional[str]]
        if view is not None:
            candidates = [view]
        elif has_view and self.prefer_projection:
            candidates = [TRUSTED_VIEW]
        elif has_view:
            candidates = [None, TRUSTED_VIEW]
        else:
            candidates = [None]
        last = None
        for v in candidates:
            name = f"{tool}{VIEW_SEP}{v}" if v else tool
            edge = self._ret_edge(agent, tool, v)
            try:
                out_label = self.manifest.output_label(name)
            except KeyError:
                return False, edge, EMPTY, "unknown_view", v
            ok, reason = self._gate(edge, out_label, agent, tool)
            if ok:
                return True, edge, out_label, reason, v
            last = (False, edge, out_label, reason, v)
        return last

    def after_tool_result(self, agent: str, tool: str, view: Optional[str] = None) -> None:
        """Taint the agent's context with the label of what it received.

        Call this after an *allowed* tool call has returned. The view chosen
        by :meth:`before_tool_call` is used unless ``view`` is given.
        """
        base, named_view = split_view(tool)
        view = view or named_view or self._view.get((agent, base))
        name = f"{base}{VIEW_SEP}{view}" if view else base
        self.ctx_labels[agent] = self.label_of(agent) | self.manifest.output_label(name)

    # ------------------------------------------------------------------
    # P2 / P3 - memory gates
    # ------------------------------------------------------------------

    def memory_write(self, agent: str, memory: str, content: Any) -> Decision:
        """P2: store ``content`` with the writer's current label (if allowed)."""
        t0 = time.perf_counter()
        label = self.label_of(agent)
        edge = f"mw:{agent}->{memory}"
        ok, reason = self._gate(edge, label, agent)
        if ok:
            self.memory[memory].append(MemoryEntry(content, label, agent, self.session,
                                                   self.principal))
        return self._decide(t0, "P2", f"write {memory}", agent, edge, label, ok, reason)

    def memory_read(self, agent: str, memory: str) -> tuple[Decision, list[Any]]:
        """P3: return the entries the agent may see; taint its context.

        The gate is applied *per entry*: blocked entries are filtered out
        rather than failing the whole read, so clean entries stay usable.
        """
        t0 = time.perf_counter()
        edge = f"mr:{memory}->{agent}"
        if edge not in self.edges:
            return self._decide(t0, "P3", f"read {memory}", agent, None, EMPTY, False,
                                "undeclared"), []
        visible, filtered = [], 0
        gained: Label = EMPTY
        for entry in self.memory.get(memory, []):
            if blocks(self.policy.mode(edge), entry.label):
                filtered += 1
            else:
                visible.append(entry.content)
                gained = gained | entry.label
        self.ctx_labels[agent] = self.label_of(agent) | gained
        return self._decide(t0, "P3", f"read {memory}", agent, edge, gained, True,
                            "allowed", returned=len(visible), filtered=filtered), visible

    # ------------------------------------------------------------------
    # P4 - inter-agent gate
    # ------------------------------------------------------------------

    def send_message(self, sender: str, receiver: str) -> Decision:
        """P4: may ``sender`` send a message to ``receiver`` now?

        If allowed, the receiver's context inherits the sender's label.
        """
        t0 = time.perf_counter()
        label = self.label_of(sender)
        edge = f"msg:{sender}->{receiver}"
        ok, reason = self._gate(edge, label, sender)
        if ok:
            self.ctx_labels[receiver] = self.label_of(receiver) | label
        return self._decide(t0, "P4", f"message {receiver}", sender, edge, label, ok, reason)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        denied = [d for d in self.trace if not d.allowed]
        return {
            "decisions": len(self.trace),
            "denied": len(denied),
            "denied_by_reason": _count(d.reason for d in denied),
            "sink_violations": len(self.sink_events),
            "quarantined_entries": len(self.quarantine),
            "confirmations": len(self.confirmations),
            "confirmations_approved": sum(c.approved for c in self.confirmations),
            "mean_check_us": (sum(d.elapsed_us for d in self.trace) / len(self.trace))
            if self.trace else 0.0,
        }


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def project_output(manifest: Manifest, tool: str, value: Any) -> Any:
    """Apply the ``#trusted`` view to an actual tool result.

    Works on the two shapes structured tools return: a dictionary (one
    record) or a list of dictionaries (several records, e.g. transactions).
    Keys that are not trusted fields are removed. Objects with a
    ``model_dump()`` method (Pydantic models) are converted first. Any other
    shape cannot be projected safely and raises ``TypeError`` (fail closed).
    """
    keep = set(manifest.projected_fields(tool))
    if not keep:
        raise ValueError(f"tool '{tool}' has no trusted view")

    def one(record: Any) -> dict:
        if hasattr(record, "model_dump"):
            record = record.model_dump()
        if not isinstance(record, dict):
            raise TypeError(f"cannot project a {type(record).__name__}")
        return {k: v for k, v in record.items() if k in keep}

    if isinstance(value, (list, tuple)):
        return [one(r) for r in value]
    return one(value)
