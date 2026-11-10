"""Shared fixtures and small helper manifests for the test suite."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_4agents.yaml"


@pytest.fixture
def demo_path() -> Path:
    return DEMO


@pytest.fixture
def demo_dict() -> dict:
    with open(DEMO, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def tiny(**overrides) -> dict:
    """A minimal one-agent system: web (untrusted) + crm (sensitive) + e-mail.

    Small enough for brute-force comparisons, rich enough to contain both an
    integrity and a confidentiality attack path.
    """
    base = {
        "version": 1,
        "system": "tiny",
        "data_classes": [
            {"id": "pub", "sensitivity": "public"},
            {"id": "pii", "sensitivity": "confidential"},
        ],
        "tools": [
            {"id": "web", "output": {"trust": "untrusted", "data_classes": ["pub"]},
             "input_destination": "external"},
            {"id": "crm", "output": {"trust": "trusted", "data_classes": ["pii"]}},
            {"id": "email", "input_destination": "external", "integrity_critical": True},
        ],
        "memory": [{"id": "notes", "persistence": "persistent"}],
        "agents": [
            {"id": "reader", "tools": ["web", "crm"], "memory": {"write": ["notes"]},
             "delegates_to": ["actor"]},
            {"id": "actor", "tools": ["email"], "memory": {"read": ["notes"]}},
        ],
        "tasks": [],
        "controls": {"sessions": 1},
    }
    out = copy.deepcopy(base)
    out.update(overrides)
    return out
