"""Legitimate workflows and the utility runner.

A *workflow* is a sequence of actions that the system must be able to
perform for its users. Workflows are built from the ``tasks`` section of the
manifest, so they come from the same declaration the developer already
wrote; nothing is invented by hand.

Two families are produced:

``task``   one workflow per task: its required edges, in declaration order.
``mixed``  two different tasks run one after the other *in the same
           session* (for example "summarise the inbox" and then "reply to a
           customer"). This is the realistic case where session-level labels
           can cost utility: content read for the first task stays in the
           agents' context during the second one. Measuring how often these
           workflows are blocked quantifies the utility cost of the policy
           (hypotheses H3 and H5 of the thesis).

A workflow *succeeds* when every action is allowed and no memory entry is
hidden from the reader.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from .guard import Action, Guard, ToolFn
from .labels import (EMPTY, SENSITIVE, Label, breaks_task, confidentiality_violation,
                     integrity_violation, needs_confirmation)
from .manifest import Manifest, split_view
from .policy import Policy
from .runtime import PolicyEngine


@dataclass
class Workflow:
    id: str
    kind: Literal["task", "mixed"]
    actions: list[Action]
    tasks: list[str]

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "tasks": self.tasks,
                "actions": [a.to_dict() for a in self.actions]}


def label_violations(manifest: Manifest, wf: Workflow) -> list[str]:
    """Properties the workflow itself would violate with no gate at all.

    A workflow that, for example, sends an e-mail after the context was
    exposed to untrusted content *is* an integrity-violating flow at the
    label level, even if it was meant to be legitimate: no secure policy can
    let it pass. Separating such workflows from label-safe ones is
    essential to read the utility results correctly.
    """
    engine = PolicyEngine(manifest, Policy.for_manifest(manifest), strict_manifest=False)
    run_on_guard(Guard(engine), wf, "probe")
    return sorted({p for e in engine.sink_events for p in e.properties})


@dataclass
class WorkflowResult:
    workflow_id: str
    kind: str
    config: str
    n_actions: int
    n_blocked: int
    n_filtered: int
    success: bool
    first_block: Optional[str]  # edge of the first blocked action, if any
    label_safe: bool  # False if the workflow itself violates a property
    check_us: float  # time spent in policy checks
    wall_us: float  # total time of the workflow
    outcomes: list = field(default_factory=list, repr=False)
    n_confirmations: int = 0  # human confirmations requested (mode H)


def requirement_to_action(edge: str) -> Action:
    """Map a required edge (e.g. ``msg:a->b``) to the action that uses it."""
    kind, body = edge.split(":", 1)
    if kind == "persist":
        return Action("end_session")
    src, dst = body.split("->")
    return {
        "prompt": Action("prompt", dst, src),
        "ret": Action("tool", dst, src),  # ``src`` may name a view (tool#trusted)
        "call": Action("tool", src, dst),
        "mw": Action("write", src, dst),
        "mr": Action("read", dst, src),
        "msg": Action("send", src, dst),
    }[kind]


def _task_actions(manifest: Manifest, task_id: str) -> list[Action]:
    task = next(t for t in manifest.tasks if t.id == task_id)
    actions: list[Action] = []
    for req in task.requires:
        a = requirement_to_action(req.edge)
        # A tool both called and returning (e.g. web_search) appears twice in
        # the requirements but is a single call: do not repeat it. If one of
        # the two names a view, keep the more specific one.
        prev = actions[-1] if actions else None
        if (prev is not None and a.op == prev.op == "tool" and a.agent == prev.agent
                and split_view(a.target)[0] == split_view(prev.target)[0]):
            if "#" in a.target:
                actions[-1] = a
            continue
        actions.append(a)
    return actions


def task_workflows(manifest: Manifest) -> list[Workflow]:
    return [Workflow(f"task-{t.id}", "task", _task_actions(manifest, t.id), [t.id])
            for t in manifest.tasks]


def mixed_workflows(manifest: Manifest) -> list[Workflow]:
    """Every ordered pair of tasks run in the same session.

    A task paired with itself is included (for example two web searches in
    a row, the second refining the first): repeating a task is a common,
    legitimate pattern.
    """
    out = []
    for a, b in itertools.product([t.id for t in manifest.tasks], repeat=2):
        acts = _task_actions(manifest, a) + _task_actions(manifest, b)
        out.append(Workflow(f"mixed-{a}+{b}", "mixed", acts, [a, b]))
    return out


def run_workflow(manifest: Manifest, policy: Policy, wf: Workflow, config: str,
                 tools: Optional[dict[str, ToolFn]] = None) -> WorkflowResult:
    """Execute a workflow through a fresh engine and guard.

    The workflows are legitimate, so the simulated human approves every
    confirmation request (mode H); the number of requests is reported.
    """
    engine = PolicyEngine(manifest, policy, strict_manifest=False,
                          confirm=lambda agent, tool, label: True)
    guard = Guard(engine, dict(tools or {}))
    return run_on_guard(guard, wf, config, not label_violations(manifest, wf))


def run_on_guard(guard: Guard, wf: Workflow, config: str,
                 label_safe: bool = True) -> WorkflowResult:
    t0 = time.perf_counter()
    outcomes = [guard.execute(a) for a in wf.actions]
    wall = (time.perf_counter() - t0) * 1e6
    blocked = [o for o in outcomes if not o.allowed]
    filtered = sum(o.filtered for o in outcomes)
    first = None
    if blocked:
        d = blocked[0].decision
        first = (d.edge or d.reason) if d else blocked[0].action.describe()
    return WorkflowResult(
        workflow_id=wf.id, kind=wf.kind, config=config, n_actions=len(outcomes),
        n_blocked=len(blocked), n_filtered=filtered,
        success=not blocked and filtered == 0, first_block=first, label_safe=label_safe,
        check_us=sum(d.elapsed_us for d in guard.engine.trace), wall_us=wall,
        outcomes=outcomes, n_confirmations=len(guard.engine.confirmations))


# ---------------------------------------------------------------------------
# Usage profile: which (edge, label) pairs a workflow needs
# ---------------------------------------------------------------------------

#: A usage profile: the set of (edge id, label) pairs a workflow crosses.
Profile = frozenset


def usage_profile(manifest: Manifest, wf: Workflow) -> Profile:
    """Every (edge, label) pair the workflow crosses when nothing is gated.

    Key property (tested in ``test_weighted_synthesis.py``): a workflow
    succeeds under a policy **if and only if** no gate of the policy blocks
    one of these pairs. Reason: as long as nothing is blocked, the run is
    identical to the ungated run, so it crosses exactly these pairs; and as
    soon as one pair is blocked, an action is denied or a memory entry is
    hidden, which counts as a failure.

    The simulation mirrors the runtime engine exactly (same label rules).
    """
    ctx: dict[str, Label] = {}
    memory: dict[str, list[Label]] = {m.id: [] for m in manifest.memory}
    used: set[tuple[str, Label]] = set()

    def label(agent: str) -> Label:
        return ctx.get(agent, EMPTY)

    who = manifest.controls.principal_of
    session = 1
    for a in wf.actions:
        if a.op == "prompt":
            lbl = manifest.principal_label(who(session))
            used.add((f"prompt:{who(session)}->{a.agent}", lbl))
            ctx[a.agent] = label(a.agent) | lbl
        elif a.op == "tool":
            base, _view = split_view(a.target)
            tool = manifest.tool(base)
            if tool.is_sink:  # the context flows into the tool's input
                used.add((f"call:{a.agent}->{base}", label(a.agent)))
            if tool.output is not None:  # the output (or its view) flows into the context
                out = manifest.output_label(a.target)
                used.add((f"ret:{a.target}->{a.agent}", out))
                ctx[a.agent] = label(a.agent) | out
        elif a.op == "write":
            used.add((f"mw:{a.agent}->{a.target}", label(a.agent)))
            memory[a.target].append(label(a.agent))
        elif a.op == "read":
            for entry in memory[a.target]:  # the read gate acts per entry
                used.add((f"mr:{a.target}->{a.agent}", entry))
                ctx[a.agent] = label(a.agent) | entry
        elif a.op == "send":
            used.add((f"msg:{a.agent}->{a.target}", label(a.agent)))
            ctx[a.target] = label(a.target) | label(a.agent)
        elif a.op == "end_session":
            crossing = who(session) != who(session + 1)
            for m in manifest.memory:
                if m.persistence == "persistent":
                    for entry in memory[m.id]:
                        used.add((f"persist:{m.id}", entry))
                    if crossing:  # tenant isolation (see graph.py)
                        memory[m.id] = [e | {SENSITIVE} for e in memory[m.id]]
                else:
                    memory[m.id] = []
            ctx = {}
            session += 1
    return frozenset(used)


def succeeds_under(profile: Profile, gates) -> bool:
    """Predict, without running it, whether a workflow survives ``gates``.

    (Human-confirmation gates never make a legitimate workflow fail.)
    """
    return not any(breaks_task(gates.get(edge), lbl) for edge, lbl in profile)


def confirmations_under(profile: Profile, gates) -> int:
    """Number of distinct (edge, label) pairs of the workflow that a gate in
    mode H submits to the human (upper bound of the interruptions)."""
    return sum(needs_confirmation(gates.get(edge), lbl) for edge, lbl in profile)


# ---------------------------------------------------------------------------
# Lower bound on human confirmations (v0.6)
# ---------------------------------------------------------------------------


def unsafe_sink_calls(manifest: Manifest, profile: Profile) -> list[tuple[str, Label]]:
    """The (call edge, label) pairs of a workflow that violate a sink property.

    A pair is listed if the call reaches an integrity-critical tool with an
    untrusted label, or an external tool with an untrusted and sensitive label.
    """
    out = []
    for edge, lbl in profile:
        if not edge.startswith("call:"):
            continue
        tool = manifest.tool(edge.split("->", 1)[1])
        if (tool.integrity_critical and integrity_violation(lbl)) or \
                (tool.is_confidentiality_sink and confidentiality_violation(lbl)):
            out.append((edge, lbl))
    return sorted(out, key=lambda p: (p[0], sorted(p[1])))


def confirmation_lower_bound(manifest: Manifest) -> dict[str, int]:
    """Minimum number of human confirmations each task needs, under ANY
    secure policy that lets the task complete.

    Proposition (proved in the thesis, checked by the tests): let p be a
    (call edge, label) pair of a task's usage profile that violates a sink
    property. Every edge on the derivation that leads to p carries an
    untrusted label, so a gate in mode R, I or C placed on any of them
    (C when the label is also sensitive) would break the task; and the
    derivation is a witness, which a secure policy must break. The only gate
    that breaks it without breaking the task is H on the call edge itself.
    Hence every such pair costs one confirmation, whatever the policy:

        confirmations(task) >= | unsafe_sink_calls(task) |

    The bound depends only on the labels of the manifest. It is reached by
    the policies amps synthesizes, so finer labels (for instance field-level
    labels) are the only way to lower the number of confirmations.
    """
    return {wf.tasks[0]: len(unsafe_sink_calls(manifest, usage_profile(manifest, wf)))
            for wf in task_workflows(manifest)}
