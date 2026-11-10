"""Canary-based checking of manifest labels."""

import json
from pathlib import Path

import pytest
import yaml

from amps.inference import (AGREES, CONSERVATIVE_SINK, CONSERVATIVE_UNTRUSTED, NOT_OBSERVED,
                            UNSAFE_MISSING_FIELD, UNSAFE_MISSING_SINK, UNSAFE_MISSING_UNTRUSTED,
                            collect_evidence, compare, inferred_output, load_observations, unsafe)
from amps.manifest import load_manifest, manifest_from_dict

ROOT = Path(__file__).resolve().parents[1]
C_FILE, C_TX = "AMPS_CANARY_file_01", "AMPS_CANARY_tx_02"

# Simulated observations shaped like the AgentDojo banking suite.
OBS = {
    "system": "simulated-banking",
    "canaries": {"bill_text": C_FILE, "incoming_transaction": C_TX},
    "calls": [
        {"task": "t0", "tool": "read_file", "output": f"Bill ... {C_FILE} ...", "state_changed": False},
        {"task": "t1", "tool": "get_most_recent_transactions", "state_changed": False,
         "output": [{"id": 1, "sender": "me", "recipient": "x", "amount": 10.0,
                     "subject": "rent", "date": "2022-01-01", "recurring": False},
                    {"id": 2, "sender": "y", "recipient": "me", "amount": 5.0,
                     "subject": C_TX, "date": "2022-02-01", "recurring": False}]},
        {"task": "t2", "tool": "get_scheduled_transactions", "state_changed": False,
         "output": [{"id": 6, "recipient": "z", "amount": 50.0, "subject": "spotify"}]},
        {"task": "t3", "tool": "get_iban", "output": "DE89", "state_changed": False},
        {"task": "t4", "tool": "get_balance", "output": 1810.0, "state_changed": False},
        {"task": "t5", "tool": "get_user_info", "output": {"first_name": "Emma"}, "state_changed": False},
        {"task": "t6", "tool": "send_money", "output": {"message": "ok"}, "state_changed": True},
        {"task": "t7", "tool": "schedule_transaction", "output": {"message": "ok"}, "state_changed": True},
        {"task": "t8", "tool": "update_scheduled_transaction", "output": {"message": "ok"},
         "state_changed": True},
        {"task": "t9", "tool": "update_password", "output": {"message": "ok"}, "state_changed": True},
        {"task": "t10", "tool": "update_user_info", "output": {"first_name": "Emma"},
         "state_changed": True},
    ],
}


def _verdicts(manifest):
    return {(f.tool, f.aspect): f.verdict for f in compare(manifest, collect_evidence(OBS))}


def test_evidence_finds_untrusted_tools_and_fields():
    ev = collect_evidence(OBS)
    assert ev["read_file"].untrusted and ev["read_file"].unstructured_hit
    assert ev["get_most_recent_transactions"].untrusted_fields == {"subject"}
    assert not ev["get_scheduled_transactions"].untrusted
    assert {t for t, e in ev.items() if e.is_action} == {
        "send_money", "schedule_transaction", "update_scheduled_transaction",
        "update_password", "update_user_info"}


def test_inferred_output_gives_field_level_labels():
    out = inferred_output(collect_evidence(OBS)["get_most_recent_transactions"])
    trust = {f["id"]: f["trust"] for f in out["fields"]}
    assert trust["subject"] == "untrusted"
    assert {k for k, v in trust.items() if v == "untrusted"} == {"subject"}
    assert inferred_output(collect_evidence(OBS)["read_file"]) == {"trust": "untrusted"}


@pytest.mark.parametrize("name", ["banking.yaml", "banking_fields.yaml"])
def test_example_manifests_have_no_unsafe_finding(name):
    m = load_manifest(ROOT / "examples" / name)
    findings = compare(m, collect_evidence(OBS))
    assert unsafe(findings) == []
    v = _verdicts(m)
    assert v[("read_file", "output")] == AGREES
    assert v[("send_money", "sink")] == AGREES


def test_mislabelled_manifest_is_caught():
    d = yaml.safe_load((ROOT / "examples" / "banking.yaml").read_text())
    for t in d["tools"]:
        if t["id"] == "read_file":
            t["output"]["trust"] = "trusted"             # error 1: files declared trusted
        if t["id"] == "update_password":
            t["integrity_critical"] = False              # error 2: action not a sink
            t["input_destination"] = "local"
    for task in d["tasks"]:  # the call edge no longer exists: drop it from the tasks
        task["requires"] = [r for r in task["requires"]
                            if r["edge"] != "call:assistant->update_password"]
    v = _verdicts(manifest_from_dict(d))
    assert v[("read_file", "output")] == UNSAFE_MISSING_UNTRUSTED
    assert v[("update_password", "sink")] == UNSAFE_MISSING_SINK


def test_wrong_field_label_is_caught():
    d = yaml.safe_load((ROOT / "examples" / "banking_fields.yaml").read_text())
    for t in d["tools"]:
        if t["id"] == "get_most_recent_transactions":
            for f in t["output"]["fields"]:
                f["trust"] = "untrusted" if f["id"] == "amount" else "trusted"  # subject wrongly trusted
    v = _verdicts(manifest_from_dict(d))
    assert v[("get_most_recent_transactions", "output")] == UNSAFE_MISSING_FIELD


def test_conservative_and_unobserved_cases():
    obs = {"canaries": OBS["canaries"], "calls": [
        {"tool": "get_iban", "output": "DE89", "state_changed": False},
        {"tool": "send_money", "output": {"message": "no"}, "state_changed": False}]}
    d = yaml.safe_load((ROOT / "examples" / "banking.yaml").read_text())
    for t in d["tools"]:
        if t["id"] == "get_iban":
            t["output"]["trust"] = "untrusted"
    v = {(f.tool, f.aspect): f.verdict
         for f in compare(manifest_from_dict(d), collect_evidence(obs))}
    assert v[("get_iban", "output")] == CONSERVATIVE_UNTRUSTED
    assert v[("send_money", "sink")] == CONSERVATIVE_SINK
    assert v[("read_file", "both")] == NOT_OBSERVED


def test_canaries_must_be_unique(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"canaries": {"a": "X", "b": "X"}, "calls": []}))
    with pytest.raises(ValueError):
        load_observations(p)
