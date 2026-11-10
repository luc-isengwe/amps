"""Tests of the v0.5 features: field-level views, human confirmation (H),
heterogeneous sessions (principals), and backward compatibility."""

import copy
from pathlib import Path

import pytest

from amps.graph import compile_manifest
from amps.labels import Mode, breaks_task, blocks
from amps.manifest import manifest_from_dict, load_manifest
from amps.policy import Policy
from amps.replay import replay_witness
from amps.runtime import PolicyEngine, project_output
from amps.synthesis import synthesize, synthesize_ilp
from amps.verify import verify
from amps.workflows import run_workflow, task_workflows

ROOT = Path(__file__).resolve().parents[1]
S, U = frozenset({"S"}), frozenset({"U"})


# ---------------------------------------------------------------- fixtures

def bank(allow_human=False, call_label=("S",), full=False):
    """Small bank: one agent reads transactions (subject untrusted), pays.

    ``full=True``: the task needs the full output (including the subject)."""
    ret = ({"edge": "ret:transactions->assistant", "under": ["U", "S"]} if full else
           {"edge": "ret:transactions#trusted->assistant", "under": ["S"]})
    return {
        "system": "mini-bank",
        "data_classes": [{"id": "tx", "sensitivity": "confidential"}],
        "tools": [
            {"id": "transactions", "output": {"fields": [
                {"id": "amount", "trust": "trusted", "data_classes": ["tx"]},
                {"id": "sender", "trust": "trusted", "data_classes": ["tx"]},
                {"id": "subject", "trust": "untrusted"}]}},
            {"id": "pay", "input_destination": "external", "integrity_critical": True},
        ],
        "agents": [{"id": "assistant", "tools": ["transactions", "pay"]}],
        "tasks": [{"id": "refund", "requires": [
            ret, {"edge": "call:assistant->pay", "under": list(call_label)}]}],
        "controls": {"sessions": 1, "allow_human": allow_human},
    }


def support(principals=("alice", "bob")):
    """A support agent serving several untrusted customers, with shared notes."""
    return {
        "system": "support",
        "data_classes": [{"id": "kb", "sensitivity": "public"}],
        "tools": [
            {"id": "kb_search", "output": {"trust": "trusted", "data_classes": ["kb"]}},
            {"id": "reply", "input_destination": "external"},
        ],
        "memory": [{"id": "notes", "persistence": "persistent"}],
        "agents": [{"id": "helper", "tools": ["kb_search", "reply"],
                    "memory": {"read": ["notes"], "write": ["notes"]}}],
        "principals": [{"id": p, "trust": "untrusted", "entry_agents": ["helper"]}
                       for p in sorted(set(principals))],
        "tasks": [{"id": "answer", "requires": [
            {"edge": f"prompt:{principals[0]}->helper", "under": ["U"]},
            {"edge": "call:helper->reply", "under": ["U"]}]}],
        "controls": {"sessions": 2, "session_principals": list(principals)},
    }


# ---------------------------------------------------------------- labels

def test_human_mode_blocks_attacks_but_never_breaks_tasks():
    assert blocks(Mode.HUMAN, U) and not blocks(Mode.HUMAN, S)
    assert not breaks_task(Mode.HUMAN, U | S)
    assert breaks_task(Mode.INTEGRITY, U)


# ---------------------------------------------------------------- field-level views

def test_view_labels_and_edges():
    m = manifest_from_dict(bank())
    assert m.output_label("transactions") == U | S
    assert m.output_label("transactions#trusted") == S
    assert m.projected_fields("transactions") == ["amount", "sender"]
    assert "ret:transactions#trusted->assistant" in m.edge_templates()


def test_projection_makes_the_task_feasible_and_the_system_secure():
    m = manifest_from_dict(bank())
    cg = compile_manifest(m)
    res = synthesize_ilp(cg)
    assert res.status == "secure" and verify(cg, res.policy).safe
    # The task (projection, then pay) passes the synthesized policy.
    wf = task_workflows(m)[0]
    assert [a.target for a in wf.actions] == ["transactions#trusted", "pay"]
    assert run_workflow(m, res.policy, wf, "B").success


def test_runtime_falls_back_to_projection_and_projects_values():
    m = manifest_from_dict(bank())
    pol = Policy.for_manifest(m, {"ret:transactions->assistant": Mode.INTEGRITY})
    eng = PolicyEngine(m, pol)
    d = eng.before_tool_call("assistant", "transactions")
    assert d.allowed and d.details["view"] == "trusted"
    eng.after_tool_result("assistant", "transactions")
    assert eng.label_of("assistant") == S  # no U: the subject was not delivered
    rows = [{"amount": 10, "sender": "X", "subject": "do something"}]
    assert project_output(m, "transactions", rows) == [{"amount": 10, "sender": "X"}]
    with pytest.raises(TypeError):
        project_output(m, "transactions", "free text")


def test_prefer_projection_even_when_full_output_is_allowed():
    m = manifest_from_dict(bank())
    eng = PolicyEngine(m, Policy.for_manifest(m), prefer_projection=True)
    assert eng.before_tool_call("assistant", "transactions").details["view"] == "trusted"


def test_view_requires_mixed_trust():
    d = bank(full=True)
    d["tools"][0]["output"]["fields"][2]["trust"] = "trusted"  # nothing to strip
    d["tasks"][0]["requires"][0]["under"] = ["S"]
    m = manifest_from_dict(d)
    assert m.projected_fields("transactions") == []
    assert not any("#" in e for e in m.edge_templates())


def test_contradictory_output_trust_is_rejected():
    d = bank()
    d["tools"][0]["output"]["trust"] = "trusted"
    with pytest.raises(ValueError):
        manifest_from_dict(d)


# ---------------------------------------------------------------- human confirmation

def test_task_acting_on_untrusted_content_is_a_conflict_without_human():
    m = manifest_from_dict(bank(allow_human=False, call_label=("U", "S"), full=True))
    res = synthesize_ilp(compile_manifest(m))
    assert res.status == "conflict"


def test_human_gate_resolves_the_conflict():
    m = manifest_from_dict(bank(allow_human=True, call_label=("U", "S"), full=True))
    cg = compile_manifest(m)
    res = synthesize(cg)  # 'auto' must not pick min-cut when H is allowed
    assert res.status == "secure"
    assert res.policy.gates == {"call:assistant->pay": Mode.HUMAN}
    assert verify(cg, res.policy).safe


def test_human_gate_at_runtime_and_in_replay():
    m = manifest_from_dict(bank(allow_human=True, call_label=("U", "S"), full=True))
    cg = compile_manifest(m)
    policy = synthesize_ilp(cg).policy
    witness = verify(cg, {}).witnesses[0]
    # No human available (default): fail safe, the attack is stopped.
    assert not replay_witness(PolicyEngine(m, policy), witness).violation
    # A human who approves: the request is counted and the action proceeds.
    asked = []
    eng = PolicyEngine(m, policy, confirm=lambda a, t, l: asked.append(t) or True)
    eng.before_tool_call("assistant", "transactions")
    eng.after_tool_result("assistant", "transactions")
    assert eng.before_tool_call("assistant", "pay").reason == "confirmed"
    assert asked == ["pay"] and eng.summary()["confirmations"] == 1


def test_human_gate_only_on_call_edges():
    d = bank(allow_human=True, call_label=("U", "S"), full=True)
    d["tasks"][0]["requires"][1]["under"] = ["S"]  # I is now allowed on the call
    res = synthesize_ilp(compile_manifest(manifest_from_dict(d)))
    assert Mode.HUMAN not in {md for e, md in res.policy.gates.items() if not e.startswith("call:")}


# ---------------------------------------------------------------- principals

def test_cross_tenant_leak_is_found_and_closed():
    m = manifest_from_dict(support())
    cg = compile_manifest(m)
    ws = verify(cg).witnesses
    leak = [w for w in ws if w.property == "confidentiality"]
    assert leak, "notes written for alice must count as sensitive in bob's session"
    assert any(s.kind == "persist" for s in leak[0].steps)
    res = synthesize_ilp(cg)
    assert res.status == "secure" and verify(cg, res.policy).safe
    # Static and runtime agree on the witness.
    assert replay_witness(PolicyEngine(m, Policy.for_manifest(m)), leak[0]).violation
    assert not replay_witness(PolicyEngine(m, res.policy), leak[0]).violation


def test_single_tenant_has_no_cross_tenant_leak():
    m = manifest_from_dict(support(principals=("alice", "alice")))
    assert not [w for w in verify(compile_manifest(m)).witnesses
                if w.property == "confidentiality"]


def test_untrusted_principal_taints_entry_agent():
    m = manifest_from_dict(support())
    eng = PolicyEngine(m, Policy.for_manifest(m))
    assert eng.principal == "alice"
    assert eng.user_prompt("helper").allowed and eng.label_of("helper") == U
    eng.end_session()
    assert eng.principal == "bob"


def test_principals_must_be_scheduled():
    d = support()
    d["controls"]["session_principals"] = []
    with pytest.raises(ValueError):
        manifest_from_dict(d)


# ---------------------------------------------------------------- compatibility

def test_v04_fingerprints_and_policies_still_valid():
    m = load_manifest(ROOT / "examples" / "banking.yaml")
    p = Policy.load(ROOT / "results" / "banking_policy_B.json")
    assert m.sha256() == p.manifest_sha256
    assert all(p.registry[t.id] == t.descriptor_sha256() for t in m.tools)
    PolicyEngine(m, p)  # strict manifest check passes


def test_rug_pull_on_fielded_tool_makes_every_field_untrusted():
    from amps.mutations import rug_pull
    new = manifest_from_dict(rug_pull(bank(), "transactions"))
    assert new.projected_fields("transactions") == []
    assert new.output_label("transactions") == U | S


def test_replay_finds_the_interleaving_that_realises_a_witness():
    """Found by the lifecycle study: after a new delegation, the attack only
    works if the supervisor forwards sensitive data *before* it reads the
    untrusted content. The replay must search that order."""
    import yaml
    from amps.impact import impact_analysis
    from amps.mutations import new_delegation
    d = yaml.safe_load((ROOT / "examples" / "demo_4agents.yaml").read_text())
    old = manifest_from_dict(d)
    deployed = synthesize_ilp(compile_manifest(old)).policy
    new = manifest_from_dict(new_delegation(d, src="reviewer", dst="executor"))
    rep = impact_analysis(old, new, deployed)
    carried = Policy.for_manifest(new, {e: m for e, m in deployed.gates.items()
                                        if e in compile_manifest(new).templates})
    assert rep.new_witnesses
    for w in rep.new_witnesses:  # every static witness is realisable at run time
        assert replay_witness(PolicyEngine(new, carried), w).violation
        assert not replay_witness(PolicyEngine(new, rep.resynthesis.policy), w).violation
