"""Replay a static witness against the runtime engine.

A witness is a list of steps (edge + label) produced by the verifier. Each
step corresponds to one engine call:

=========  ======================================================
step kind  engine call(s)
=========  ======================================================
prompt     user_prompt(agent)
ret        before_tool_call(agent, tool) + after_tool_result(...)
call       before_tool_call(agent, tool)
mw         memory_write(agent, memory, ...)
mr         memory_read(memory -> agent)
msg        send_message(sender, receiver)
persist    end_session()  (applied when the replay moves to the next session)
=========  ======================================================

The replay answers one question: *if an attacker forced the agents to
perform exactly this sequence of actions, would the engine let the
violating flow reach the sink? It is the bridge between the static
guarantee (verifier) and the runtime enforcement (engine): the generated
regression tests are replays.

Note: the runtime engine is at least as strict as the verifier (for
example, a tool call is denied if *either* its call edge or its return edge
is gated), so a witness blocked statically is always blocked at run time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .graph import node_session
from .runtime import Decision, PolicyEngine
from .verify import Witness


@dataclass
class ReplayResult:
    blocked: bool  # True if the engine stopped the attack
    violation: bool  # True if the sink was reached with a violating label
    decisions: list[Decision] = field(default_factory=list)
    stopped_at: str | None = None  # edge where the flow was stopped


def _entity(node: str) -> str:
    """'ctx:executor@2' -> 'executor', 'out:web_search' -> 'web_search'."""
    return node.split(":", 1)[1].split("@", 1)[0]


def _deps(witness: Witness) -> list[set[int]]:
    """Step i must come after the steps that bring (part of) its label into
    its source node: incoming steps whose delivered label is a subset of the
    label step i carries."""
    deps: list[set[int]] = []
    for i, s in enumerate(witness.steps):
        d = set()
        for j, o in enumerate(witness.steps):
            if j != i and o.dst == s.src and o.label <= s.label:
                d.add(j)
        deps.append(d)
    return deps


def _execute(engine: PolicyEngine, step) -> Decision | None:
    """Perform one non-persist step on ``engine``; return its decision."""
    if step.kind == "prompt":
        return engine.user_prompt(_entity(step.dst))
    if step.kind in ("ret", "call"):
        if step.kind == "ret":
            tool, agent = _entity(step.src), _entity(step.dst)
        else:
            agent, tool = _entity(step.src), _entity(step.dst)
        d = engine.before_tool_call(agent, tool)  # tool may carry a view
        if d.allowed and step.kind == "ret":
            engine.after_tool_result(agent, tool)
        return d
    if step.kind == "mw":
        return engine.memory_write(_entity(step.src), _entity(step.dst), content=f"<{step.template}>")
    if step.kind == "mr":
        d, entries = engine.memory_read(_entity(step.dst), _entity(step.src))
        if d.allowed and d.details.get("filtered") and not entries:
            # Every entry was hidden by the read gate: nothing reached the reader.
            d = Decision(**{**d.__dict__, "allowed": False, "reason": "label-filtered"})
        return d
    if step.kind == "msg":
        return engine.send_message(_entity(step.src), _entity(step.dst))
    raise ValueError(f"unknown step kind {step.kind}")  # pragma: no cover


def _violates(decisions: list[Decision], prop: str) -> bool:
    return any(d.point == "P1" and d.allowed and prop in d.details.get("violated", [])
               for d in decisions)


def _clone(engine: PolicyEngine) -> PolicyEngine:
    # Copy the mutable state only; manifest and policy are shared (read-only).
    memo = {id(engine.manifest): engine.manifest, id(engine.policy): engine.policy,
            id(engine.edges): engine.edges, id(engine.confirm): engine.confirm}
    return copy.deepcopy(engine, memo)


def replay_witness(engine: PolicyEngine, witness: Witness,
                   max_states: int = 20000) -> ReplayResult:
    """Replay *witness*. The engine's *confirm* callback plays the human:
    for an attack it should refuse (the default when no callback is set).

    The verifier proves that the labels of the witness can meet; it does not
    fix the order in which the agents act. Because agent contexts accumulate
    labels at run time, the order matters (an agent may forward sensitive
    data *before* it reads untrusted content). The replay therefore searches
    the interleavings allowed by the dependencies between steps, in session
    order, and reports a violation if any interleaving reaches the sink.
    The witness order itself is tried first.
    """
    steps = witness.steps
    deps = _deps(witness)
    sess = [s.session for s in steps]
    persist = {i for i, s in enumerate(steps) if s.kind == "persist"}
    states = 0
    first_failure: tuple[list[Decision], str | None] | None = None

    def dfs(eng: PolicyEngine, done: frozenset, current: int, decisions: list[Decision]):
        nonlocal states, first_failure
        states += 1
        if states > max_states:
            return None
        if len(done) == len(steps):
            if _violates(decisions, witness.property):
                return decisions
            if first_failure is None:
                first_failure = (decisions, "label-filtered")
            return None
        pending_here = [i for i in range(len(steps)) if i not in done and sess[i] == current
                        and i not in persist]
        if not pending_here:
            # Session finished: its persist steps happen when the session ends
            # (end_session applies the persist gates), then move on.
            done = done | {i for i in persist if sess[i] <= current}
            later = [sess[i] for i in range(len(steps)) if i not in done]
            if not later:
                return dfs(eng, frozenset(done), current, decisions)
            eng2 = _clone(eng)
            while current < min(later):
                eng2.end_session()
                current += 1
            return dfs(eng2, frozenset(done), current, decisions)
        ready = [i for i in pending_here if deps[i] - persist <= done]
        for i in ready:  # witness order first
            eng2 = _clone(eng)
            d = _execute(eng2, steps[i])
            if not d.allowed:
                if first_failure is None:
                    first_failure = (decisions + [d], steps[i].template)
                continue
            found = dfs(eng2, done | {i}, current, decisions + [d])
            if found is not None:
                return found
        return None

    start = min(sess) if steps else 1
    eng0 = _clone(engine)
    current = 1
    while current < start:
        eng0.end_session()
        current += 1
    found = dfs(eng0, frozenset(i for i in persist if sess[i] < current), current, [])
    if found is not None:
        return ReplayResult(blocked=False, violation=True, decisions=found, stopped_at=None)
    decisions, stopped = first_failure if first_failure else ([], None)
    return ReplayResult(blocked=True, violation=False, decisions=decisions, stopped_at=stopped)
