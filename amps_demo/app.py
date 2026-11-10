"""LangGraph application of the four-agent demonstrator.

Graph structure
---------------
::

                    ┌────────────┐
          START ──► │ supervisor │ ◄──────────────┐
                    └─────┬──────┘                │
            route to the agent of the next action │
          ┌───────────┬───┴────────┬──────────┐   │
          ▼           ▼            ▼          ▼   │
     retrieval    executor     reviewer      END  │
          └───────────┴────────────┴──────────────┘

* The **supervisor** node owns the plan. It executes its own actions (for
  example delegating with a message, or reading memory) and routes control
  to the agent that performs the next action.
* Each **agent node** executes the consecutive actions that belong to it,
  then hands control back to the supervisor.
* Every action goes through the :class:`amps.guard.Guard`. No node calls a
  tool, a memory or another agent directly.

Who decides the actions?
------------------------
A *planner* gives the list of actions. At tier T1 the planner is the
workflow itself (deterministic). At tier T2/T3 the agent nodes will ask a
language model which action to take next; the graph and the guard stay the
same, only the source of the actions changes. Keeping that seam explicit is
what makes the T1 and T2 results comparable.

The guard is kept outside the LangGraph state (captured by the node
functions) because it holds live objects (engine, tools) that must not be
copied or serialised by LangGraph.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from amps.guard import Action, Guard, Outcome
from amps.manifest import Manifest
from amps.policy import Policy
from amps.runtime import PolicyEngine

from .tools import TOOLS

AGENTS = ("supervisor", "retrieval", "executor", "reviewer")


class State(TypedDict):
    actions: list[Action]  # the plan
    cursor: int  # index of the next action to execute
    log: list[dict[str, Any]]  # one entry per executed action


def _owner(action: Action) -> str:
    """Agent node that must execute this action."""
    if action.op == "end_session":
        return "supervisor"  # session boundaries are handled centrally
    return action.agent


def build_app(guard: Guard, agents: tuple[str, ...] = AGENTS):
    """Compile the LangGraph application around a guard."""

    def make_node(name: str):
        def node(state: State) -> dict:
            cursor, log = state["cursor"], list(state["log"])
            actions = state["actions"]
            # Execute every consecutive action that belongs to this agent.
            while cursor < len(actions) and _owner(actions[cursor]) == name:
                out: Outcome = guard.execute(actions[cursor])
                d = out.decision
                log.append({
                    "node": name, "action": out.action.describe(), "allowed": out.allowed,
                    "edge": d.edge if d else None, "reason": d.reason if d else "lifecycle",
                    "filtered": out.filtered,
                })
                cursor += 1
            return {"cursor": cursor, "log": log}
        return node

    def route(state: State) -> str:
        if state["cursor"] >= len(state["actions"]):
            return END
        nxt = _owner(state["actions"][state["cursor"]])
        return nxt if nxt != "supervisor" else "supervisor"

    g = StateGraph(State)
    for name in agents:
        g.add_node(name, make_node(name))
    g.add_edge(START, "supervisor")
    # From the supervisor, go to the owner of the next action (or finish).
    g.add_conditional_edges("supervisor", route,
                            {**{a: a for a in agents if a != "supervisor"},
                             "supervisor": "supervisor", END: END})
    # Every worker returns to the supervisor, which routes again.
    for name in agents:
        if name != "supervisor":
            g.add_edge(name, "supervisor")
    return g.compile()


def run_plan(manifest: Manifest, policy: Policy, actions: list[Action],
             guard: Optional[Guard] = None) -> tuple[list[dict], Guard]:
    """Run a plan through the LangGraph app. Returns (log, guard)."""
    guard = guard or Guard(PolicyEngine(manifest, policy, strict_manifest=False), dict(TOOLS))
    app = build_app(guard)
    final = app.invoke({"actions": list(actions), "cursor": 0, "log": []},
                       {"recursion_limit": 10 * len(actions) + 20})
    return final["log"], guard
