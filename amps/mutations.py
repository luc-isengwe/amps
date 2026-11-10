"""Configuration mutations used in the lifecycle study.

Each function takes a manifest (as a Python dictionary, the same structure
as the YAML file) and returns a new dictionary describing the system after
a realistic change. The original is never modified (we work on a deep copy).

The mutations mirror common lifecycle events:

``rug_pull``           a tool silently changes its descriptor/behaviour
``permission_drift``   an agent receives a tool it did not have before
``new_delegation``     a new agent-to-agent channel appears
``widen_memory``       a memory becomes persistent, or an agent gets access
``neutral_tool``       a harmless tool is added (should create no new witness;
                         used to measure false alarms of the impact analysis)
``add_sink_tool``      a new tool that sends data out (e.g. a new MCP server
                         offering "share" or "export") is given to an agent
``retire_agent``       an agent is decommissioned (lifecycle "retirement"
                         stage; its gates and registrations must be revoked)
"""

from __future__ import annotations

import copy
from typing import Any, Callable

Manifest = dict[str, Any]


def _find(items: list[dict], item_id: str) -> dict:
    for item in items:
        if item["id"] == item_id:
            return item
    raise KeyError(item_id)


def rug_pull(m: Manifest, tool: str, new_description: str | None = None,
             make_untrusted: bool = True) -> Manifest:
    """The tool's descriptor changes after approval; optionally its output
    becomes attacker-influenced (e.g. a compromised MCP server)."""
    m = copy.deepcopy(m)
    t = _find(m["tools"], tool)
    t["description"] = new_description or (t.get("description", "") + " [updated by provider]")
    if make_untrusted:
        t.setdefault("output", {"trust": "untrusted", "data_classes": []})
        if t["output"].get("fields"):
            for f in t["output"]["fields"]:  # every field becomes attacker-controlled
                f["trust"] = "untrusted"
            t["output"].pop("trust", None)
            # The trusted view no longer exists: tasks that relied on it now
            # receive the full (untrusted) output.
            for task in m.get("tasks", []):
                for r in task["requires"]:
                    if r["edge"].startswith(f"ret:{tool}#"):
                        r["edge"] = f"ret:{tool}->" + r["edge"].split("->", 1)[1]
                        r["under"] = sorted(set(r.get("under", [])) | {"U"})
        else:
            t["output"]["trust"] = "untrusted"
    return m


def permission_drift(m: Manifest, agent: str, tool: str) -> Manifest:
    """Grant an existing tool to an agent that did not have it."""
    m = copy.deepcopy(m)
    a = _find(m["agents"], agent)
    if tool not in a.setdefault("tools", []):
        a["tools"].append(tool)
    return m


def new_delegation(m: Manifest, src: str, dst: str) -> Manifest:
    """Allow ``src`` to delegate to ``dst`` (creates messages both ways)."""
    m = copy.deepcopy(m)
    a = _find(m["agents"], src)
    if dst not in a.setdefault("delegates_to", []):
        a["delegates_to"].append(dst)
    return m


def widen_memory(m: Manifest, memory: str, make_persistent: bool = True,
                 grant_read_to: list[str] | None = None,
                 grant_write_to: list[str] | None = None) -> Manifest:
    """Make a memory persistent and/or give more agents access to it."""
    m = copy.deepcopy(m)
    if make_persistent:
        _find(m["memory"], memory)["persistence"] = "persistent"
    for agent in grant_read_to or []:
        acc = _find(m["agents"], agent).setdefault("memory", {})
        if memory not in acc.setdefault("read", []):
            acc["read"].append(memory)
    for agent in grant_write_to or []:
        acc = _find(m["agents"], agent).setdefault("memory", {})
        if memory not in acc.setdefault("write", []):
            acc["write"].append(memory)
    return m


def neutral_tool(m: Manifest, agent: str, tool: str = "get_time") -> Manifest:
    """Add a trusted, non-sensitive, local tool (expected: no new witness)."""
    m = copy.deepcopy(m)
    m["tools"].append({
        "id": tool,
        "description": "Return the current server time.",
        "parameters": {},
        "output": {"trust": "trusted", "data_classes": []},
        "input_destination": "local",
    })
    _find(m["agents"], agent).setdefault("tools", []).append(tool)
    return m


def add_sink_tool(m: Manifest, agent: str, tool: str = "export_data",
                  external: bool = True, integrity_critical: bool = False) -> Manifest:
    """Add a new tool whose input leaves the trust boundary (or changes state)."""
    m = copy.deepcopy(m)
    m["tools"].append({
        "id": tool,
        "description": "Send the given content to an external address.",
        "parameters": {},
        "input_destination": "external" if external else "internal",
        "integrity_critical": integrity_critical,
    })
    _find(m["agents"], agent).setdefault("tools", []).append(tool)
    return m


def retire_agent(m: Manifest, agent: str) -> Manifest:
    """Remove an agent and every reference to it (delegations, tasks, controls)."""
    m = copy.deepcopy(m)
    m["agents"] = [a for a in m["agents"] if a["id"] != agent]
    for a in m["agents"]:
        if agent in a.get("delegates_to", []):
            a["delegates_to"].remove(agent)

    def mentions(edge: str) -> bool:
        body = edge.split(":", 1)[1]
        return agent in body.replace("->", " ").split()

    for task in m.get("tasks", []):
        task["requires"] = [r for r in task["requires"] if not mentions(r["edge"])]
    m["tasks"] = [t for t in m.get("tasks", []) if t["requires"]]
    controls = m.get("controls", {})
    if "overrides" in controls:
        controls["overrides"] = {e: c for e, c in controls["overrides"].items() if not mentions(e)}
    if "fixed" in controls:
        controls["fixed"] = [e for e in controls["fixed"] if not mentions(e)]
    return m


#: Registry used by the CLI: name -> function.
MUTATIONS: dict[str, Callable[..., Manifest]] = {
    "rug_pull": rug_pull,
    "permission_drift": permission_drift,
    "new_delegation": new_delegation,
    "widen_memory": widen_memory,
    "neutral_tool": neutral_tool,
    "add_sink_tool": add_sink_tool,
    "retire_agent": retire_agent,
}
