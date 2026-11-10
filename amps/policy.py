"""The enforcement policy produced by synthesis and consumed at run time.

A policy is deliberately small and readable:

* ``gates``: a dictionary ``{edge_id: Mode}``. Edges that are not listed
  are not gated (but they must still be declared in the manifest: the
  runtime engine denies any undeclared flow by default).
* ``registry``: the approved SHA-256 fingerprint of every tool descriptor.
* ``manifest_sha256``: fingerprint of the manifest the policy was built
  for, so that a policy is never silently used with another manifest.
* ``meta``: free-form information (cost, solver, number of iterations...).

The policy is saved as JSON so that it can be reviewed, versioned with git
and compared between two versions of a system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .labels import Mode
from .manifest import Manifest


@dataclass
class Policy:
    gates: dict[str, Mode] = field(default_factory=dict)
    registry: dict[str, str] = field(default_factory=dict)
    manifest_sha256: str = ""
    system: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------

    @classmethod
    def for_manifest(cls, manifest: Manifest, gates: Optional[dict[str, Mode]] = None,
                     meta: Optional[dict[str, Any]] = None) -> "Policy":
        """Create a policy bound to ``manifest`` (registry filled automatically)."""
        return cls(
            gates=dict(sorted((gates or {}).items())),
            registry={t.id: t.descriptor_sha256() for t in manifest.tools},
            manifest_sha256=manifest.sha256(),
            system=manifest.system,
            meta=dict(meta or {}),
        )

    # -- queries --------------------------------------------------------

    def mode(self, edge_id: str) -> Optional[Mode]:
        """Gate mode on ``edge_id``, or ``None`` if the edge is not gated."""
        return self.gates.get(edge_id)

    def cost(self, manifest: Manifest) -> float:
        """Total utility cost of this policy under the manifest's cost model."""
        return sum(manifest.edge_cost(e, m) for e, m in self.gates.items())

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "manifest_sha256": self.manifest_sha256,
            "gates": {e: m.value for e, m in sorted(self.gates.items())},
            "registry": dict(sorted(self.registry.items())),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        return cls(
            gates={e: Mode(m) for e, m in data.get("gates", {}).items()},
            registry=dict(data.get("registry", {})),
            manifest_sha256=data.get("manifest_sha256", ""),
            system=data.get("system", ""),
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def diff_policies(old: Policy, new: Policy) -> dict[str, dict]:
    """Compare two policies gate by gate.

    Returns three dictionaries: gates ``added``, ``removed`` and ``changed``
    (``changed`` maps an edge to ``(old_mode, new_mode)``).
    """
    added = {e: m.value for e, m in new.gates.items() if e not in old.gates}
    removed = {e: m.value for e, m in old.gates.items() if e not in new.gates}
    changed = {
        e: (old.gates[e].value, m.value)
        for e, m in new.gates.items()
        if e in old.gates and old.gates[e] is not m
    }
    return {"added": added, "removed": removed, "changed": changed}
