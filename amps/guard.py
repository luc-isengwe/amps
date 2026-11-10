"""Guard: the adapter between an agent orchestrator and the policy engine.

The policy engine (``runtime.py``) only *decides*. The guard *executes*: it
asks the engine first, and performs the action (calls the tool function,
stores the memory entry, delivers the message) only if the decision allows
it. An orchestrator such as LangGraph therefore never touches a tool,
a memory or another agent directly; it always goes through the guard.

This is the "complete mediation" principle of Saltzer and Schroeder: every
access is checked, and there is no path around the check.

Actions
-------
An :class:`Action` is the smallest unit of agent behaviour that matters for
security:

``prompt``       the session's principal sends its request to an agent (P0)
``tool``         agent calls a tool             (P1); ``target`` may name a
                 view, e.g. ``get_most_recent_transactions#trusted``
``write``        agent writes to a memory store (P2)
``read``         agent reads from a memory store (P3)
``send``         agent sends a message to another agent (P4)
``end_session``  the current session ends (persist gates are applied)

Both the deterministic planner (tier T1) and an LLM planner (tiers T2/T3)
produce actions; the guard treats them identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from .manifest import split_view
from .runtime import Decision, PolicyEngine, project_output

Op = Literal["prompt", "tool", "write", "read", "send", "end_session"]

#: A tool implementation: receives the arguments, returns any value.
ToolFn = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class Action:
    op: Op
    agent: str = ""  # acting agent (the sender for "send")
    target: str = ""  # tool, memory or receiving agent
    args: tuple = ()  # tool arguments as (key, value) pairs (hashable)
    #: True for the action an adversary is trying to cause (used only for
    #: reporting; the guard never looks at it when deciding).
    adversarial: bool = False

    def describe(self) -> str:
        if self.op == "end_session":
            return "end_session"
        arrow = {"tool": "calls", "write": "writes", "read": "reads", "send": "messages",
                 "prompt": "receives the prompt of"}[self.op]
        return f"{self.agent} {arrow} {self.target}"

    def to_dict(self) -> dict:
        return {"op": self.op, "agent": self.agent, "target": self.target,
                "args": [list(a) for a in self.args], "adversarial": self.adversarial}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(d["op"], d.get("agent", ""), d.get("target", ""),
                   tuple(tuple(a) for a in d.get("args", [])), d.get("adversarial", False))


@dataclass
class Outcome:
    action: Action
    allowed: bool
    value: Any = None  # tool result, memory entries or delivered message
    decision: Optional[Decision] = None
    #: Number of memory entries hidden by the read gate (reads only).
    filtered: int = 0


@dataclass
class Guard:
    engine: PolicyEngine
    tools: dict[str, ToolFn] = field(default_factory=dict)
    #: Messages delivered to each agent (its inbox), in order.
    inbox: dict[str, list[Any]] = field(default_factory=dict)

    # -- single operations ----------------------------------------------

    def tool(self, agent: str, tool: str, args: Optional[dict] = None,
             action: Optional[Action] = None) -> Outcome:
        action = action or Action("tool", agent, tool, tuple(sorted((args or {}).items())))
        d = self.engine.before_tool_call(agent, tool)  # ``tool`` may carry a view
        if not d.allowed:
            return Outcome(action, False, None, d)
        base, _ = split_view(tool)
        fn = self.tools.get(base)
        value = fn(dict(args or {})) if fn else None
        if value is not None and d.details.get("view"):
            value = project_output(self.engine.manifest, base, value)  # keep trusted fields only
        self.engine.after_tool_result(agent, base)  # taint the context with the delivered label
        return Outcome(action, True, value, d)

    def prompt(self, agent: str, action: Optional[Action] = None) -> Outcome:
        action = action or Action("prompt", agent, self.engine.principal or "")
        d = self.engine.user_prompt(agent)
        return Outcome(action, d.allowed, None, d)

    def write(self, agent: str, memory: str, content: Any,
              action: Optional[Action] = None) -> Outcome:
        action = action or Action("write", agent, memory)
        d = self.engine.memory_write(agent, memory, content)
        return Outcome(action, d.allowed, None, d)

    def read(self, agent: str, memory: str, action: Optional[Action] = None) -> Outcome:
        action = action or Action("read", agent, memory)
        d, entries = self.engine.memory_read(agent, memory)
        return Outcome(action, d.allowed, entries, d, d.details.get("filtered", 0))

    def send(self, sender: str, receiver: str, content: Any,
             action: Optional[Action] = None) -> Outcome:
        action = action or Action("send", sender, receiver)
        d = self.engine.send_message(sender, receiver)
        if d.allowed:
            self.inbox.setdefault(receiver, []).append(content)
        return Outcome(action, d.allowed, content if d.allowed else None, d)

    def end_session(self) -> Outcome:
        self.engine.end_session()
        self.inbox.clear()
        return Outcome(Action("end_session"), True)

    # -- generic entry point --------------------------------------------

    def execute(self, action: Action, content: Any = None) -> Outcome:
        """Execute one action through the policy engine."""
        if action.op == "prompt":
            return self.prompt(action.agent, action)
        if action.op == "tool":
            return self.tool(action.agent, action.target, dict(action.args), action)
        if action.op == "write":
            return self.write(action.agent, action.target,
                              content if content is not None else f"note by {action.agent}", action)
        if action.op == "read":
            return self.read(action.agent, action.target, action)
        if action.op == "send":
            return self.send(action.agent, action.target,
                             content if content is not None else f"message from {action.agent}", action)
        if action.op == "end_session":
            return self.end_session()
        raise ValueError(f"unknown op {action.op!r}")
