"""Inference and checking of manifest labels from observations.

Why
---
Every guarantee of amps rests on the labels of the manifest: a tool that
brings third-party content into the agent's context must be labelled
``untrusted``, and a tool that changes the state of the world must be a
sink. Written by hand, these labels can be wrong. This module derives them
from *observations* of the real tools and compares them with a manifest.

Method (canary-based discovery)
-------------------------------
1. Every place where a third party can write data (a file, the subject of an
   incoming transfer, a web page...) is filled with a unique, harmless
   marker string, a *canary* (for instance ``AMPS_CANARY_3f9a``). A canary
   is an inert token, not an instruction.
2. Each tool is called with ordinary arguments, without any language model,
   and its output is recorded, together with whether the call changed the
   state of the environment.
3. A tool whose output contains a canary brings third-party content into the
   context: it must be ``untrusted``. If the output is structured (records
   with named fields), the fields that contain a canary are the untrusted
   fields, which gives field-level labels. A tool whose call changed the
   state of the environment is an action: it must be a sink.

The observations are stored as a JSON file (format below), produced by a
small script that drives the tools of the system under study. This module
only reads that file.

Observation file format::

    {
      "system": "agentdojo-banking-v1.2.2",
      "canaries": {"<third-party location>": "<canary string>", ...},
      "calls": [
        {"task": "user_task_0", "tool": "read_file",
         "output": <the tool output, as JSON>, "error": null,
         "state_changed": false},
        ...
      ]
    }

Soundness and limits
--------------------
* A canary found in an output is a *proof* that third-party content reaches
  the agent through this tool (no false positive, as long as the canary
  strings are unique).
* The absence of a canary is only evidence for the calls observed: a tool
  never called, or called with arguments that never touch a third-party
  location, may still be untrusted. The comparison therefore reports such
  cases as "not confirmed", never as "trusted".
* Third-party locations that are not seeded with a canary are invisible to
  the method: its coverage is that of the list of seeded locations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .manifest import Manifest

# Verdicts of the comparison, from the most to the least serious.
UNSAFE_MISSING_UNTRUSTED = "UNSAFE: untrusted content observed, manifest says trusted"
UNSAFE_MISSING_FIELD = "UNSAFE: untrusted field observed, manifest says trusted"
UNSAFE_MISSING_SINK = "UNSAFE: state change observed, manifest says not a sink"
CONSERVATIVE_UNTRUSTED = "conservative: manifest says untrusted, no canary observed"
CONSERVATIVE_SINK = "conservative: manifest says sink, no state change observed"
NOT_OBSERVED = "not observed: tool never called"
AGREES = "agrees"


@dataclass
class ToolEvidence:
    """What the observations say about one tool."""

    tool: str
    calls: int = 0
    calls_with_output: int = 0
    canary_hits: set[str] = field(default_factory=set)      # third-party locations seen
    fields_seen: set[str] = field(default_factory=set)       # keys of structured records
    untrusted_fields: set[str] = field(default_factory=set)  # keys that carried a canary
    unstructured_hit: bool = False                           # canary outside named fields
    state_changes: int = 0

    @property
    def untrusted(self) -> bool:
        return bool(self.canary_hits)

    @property
    def is_action(self) -> bool:
        return self.state_changes > 0


# ---------------------------------------------------------------------------
# Reading observations
# ---------------------------------------------------------------------------


def load_observations(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("canaries", "calls"):
        if key not in data:
            raise ValueError(f"observation file: missing '{key}'")
    values = list(data["canaries"].values())
    if len(set(values)) != len(values):
        raise ValueError("observation file: canary strings must be unique")
    return data


def _hits(value: Any, canaries: dict[str, str]) -> set[str]:
    """Third-party locations whose canary appears anywhere in ``value``."""
    text = json.dumps(value, default=str, ensure_ascii=False)
    return {loc for loc, c in canaries.items() if c in text}


def _records(output: Any) -> Optional[list[dict]]:
    """The output as a list of records (dicts), or None if it is not structured."""
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list) and output and all(isinstance(r, dict) for r in output):
        return output
    return None


def collect_evidence(observations: dict) -> dict[str, ToolEvidence]:
    """Aggregate the observed calls, tool by tool."""
    canaries = observations["canaries"]
    ev: dict[str, ToolEvidence] = {}
    for call in observations["calls"]:
        e = ev.setdefault(call["tool"], ToolEvidence(call["tool"]))
        e.calls += 1
        if call.get("state_changed"):
            e.state_changes += 1
        out = call.get("output")
        if out is None or call.get("error"):
            continue
        e.calls_with_output += 1
        hits = _hits(out, canaries)
        e.canary_hits |= hits
        recs = _records(out)
        if recs is None:
            if hits:
                e.unstructured_hit = True
            continue
        for r in recs:
            e.fields_seen |= set(r)
            for k, v in r.items():
                if _hits(v, canaries):
                    e.untrusted_fields.add(k)
    return ev


# ---------------------------------------------------------------------------
# Inferred labels and comparison with a manifest
# ---------------------------------------------------------------------------


def inferred_output(e: ToolEvidence) -> Optional[dict]:
    """Manifest ``output`` entry suggested by the evidence (None: no output seen)."""
    if e.calls_with_output == 0:
        return None
    if e.untrusted and e.fields_seen and not e.unstructured_hit \
            and e.untrusted_fields < e.fields_seen:
        return {"fields": [{"id": f, "trust": "untrusted" if f in e.untrusted_fields else "trusted"}
                           for f in sorted(e.fields_seen)]}
    return {"trust": "untrusted" if e.untrusted else "trusted"}


@dataclass
class Finding:
    tool: str
    aspect: str     # "output" or "sink"
    verdict: str
    detail: str


def compare(manifest: Manifest, evidence: dict[str, ToolEvidence]) -> list[Finding]:
    """Compare the labels of ``manifest`` with the evidence, tool by tool."""
    findings: list[Finding] = []
    for tool in manifest.tools:
        e = evidence.get(tool.id)
        if e is None or e.calls == 0:
            findings.append(Finding(tool.id, "both", NOT_OBSERVED, "no call recorded"))
            continue
        # --- output label
        if tool.output is not None:
            declared = tool.output.overall_trust
            declared_untrusted_fields = {f.id for f in tool.output.fields if f.trust == "untrusted"}
            if e.untrusted and declared == "trusted":
                findings.append(Finding(tool.id, "output", UNSAFE_MISSING_UNTRUSTED,
                                        f"canaries from {sorted(e.canary_hits)}"))
            elif e.untrusted and tool.output.fields and \
                    not e.untrusted_fields <= declared_untrusted_fields:
                extra = sorted(e.untrusted_fields - declared_untrusted_fields)
                findings.append(Finding(tool.id, "output", UNSAFE_MISSING_FIELD,
                                        f"untrusted field(s) {extra} declared trusted"))
            elif not e.untrusted and declared == "untrusted":
                findings.append(Finding(tool.id, "output", CONSERVATIVE_UNTRUSTED,
                                        f"{e.calls_with_output} call(s) with output, no canary"))
            else:
                detail = (f"untrusted, canaries from {sorted(e.canary_hits)}" if e.untrusted
                          else f"trusted in {e.calls_with_output} call(s)")
                if e.untrusted and e.untrusted_fields:
                    detail += f"; untrusted field(s) {sorted(e.untrusted_fields)}"
                findings.append(Finding(tool.id, "output", AGREES, detail))
        elif e.untrusted:
            findings.append(Finding(tool.id, "output", UNSAFE_MISSING_UNTRUSTED,
                                    "output not declared, canaries observed"))
        # --- sink
        if e.is_action and not tool.is_sink:
            findings.append(Finding(tool.id, "sink", UNSAFE_MISSING_SINK,
                                    f"{e.state_changes} call(s) changed the state"))
        elif not e.is_action and tool.is_sink:
            findings.append(Finding(tool.id, "sink", CONSERVATIVE_SINK,
                                    f"{e.calls} call(s), no state change"))
        elif e.is_action:
            findings.append(Finding(tool.id, "sink", AGREES,
                                    f"{e.state_changes} state change(s), declared sink"))
    for name in sorted(set(evidence) - {t.id for t in manifest.tools}):
        findings.append(Finding(name, "both", "UNSAFE: tool not in the manifest",
                                f"{evidence[name].calls} call(s) observed"))
    return findings


def unsafe(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.verdict.startswith("UNSAFE")]
