import subprocess
import sys

import yaml

from amps.graph import compile_manifest
from amps.impact import impact_analysis
from amps.manifest import manifest_from_dict
from amps.mutations import (neutral_tool, new_delegation, permission_drift, retire_agent,
                            rug_pull, widen_memory)
from amps.synthesis import synthesize_ilp
from amps.testgen import generate_regression_tests

from conftest import ROOT


def _deploy(d):
    m = manifest_from_dict(d)
    return m, synthesize_ilp(compile_manifest(m)).policy


def test_neutral_tool_raises_no_alarm(demo_dict):
    old, policy = _deploy(demo_dict)
    new = manifest_from_dict(neutral_tool(demo_dict, "reviewer"))
    rep = impact_analysis(old, new, policy)
    assert not rep.detected
    assert rep.policy_diff == {"added": {}, "removed": {}, "changed": {}}


def test_permission_drift_is_detected_and_fixed(demo_dict):
    old, policy = _deploy(demo_dict)
    new = manifest_from_dict(permission_drift(demo_dict, "reviewer", "send_email"))
    rep = impact_analysis(old, new, policy)
    assert rep.detected and rep.new_witnesses
    assert "call:reviewer->send_email" in rep.added_edges
    assert rep.resynthesis.status == "secure"
    assert rep.policy_diff["added"]                    # at least one new gate


def test_new_delegation_is_detected(demo_dict):
    old, policy = _deploy(demo_dict)
    new = manifest_from_dict(new_delegation(demo_dict, "reviewer", "executor"))
    rep = impact_analysis(old, new, policy)
    assert rep.detected
    assert any("msg:reviewer->executor" in w.templates() for w in rep.new_witnesses)


def test_rug_pull_flags_descriptor_and_label(demo_dict):
    old, policy = _deploy(demo_dict)
    new = manifest_from_dict(rug_pull(demo_dict, "kb_search"))
    rep = impact_analysis(old, new, policy)
    assert rep.changed_descriptors == ["kb_search"]
    assert rep.changed_labels["kb_search"] == ("{}", "{U}")
    assert rep.detected


def test_widen_memory_is_detected(demo_dict):
    old, policy = _deploy(demo_dict)
    new = manifest_from_dict(widen_memory(demo_dict, "shared_notes", make_persistent=False,
                                          grant_read_to=["executor"]))
    rep = impact_analysis(old, new, policy)
    assert rep.detected
    assert "mr:shared_notes->executor" in rep.added_edges


def test_retired_agent_leaves_no_stale_capability(demo_dict):
    # Gate something on the executor first, then retire it.
    old, policy = _deploy(demo_dict)
    new = manifest_from_dict(retire_agent(demo_dict, "executor"))
    rep = impact_analysis(old, new, policy)
    assert all("executor" not in e for e in rep.resynthesis.policy.gates)
    assert any("executor" in e for e in rep.removed_edges)
    assert set(rep.stale_gates) <= set(rep.removed_edges)


def test_generated_tests_pass_against_new_policy(demo_dict, tmp_path):
    old, policy = _deploy(demo_dict)
    new_dict = permission_drift(demo_dict, "reviewer", "send_email")
    new = manifest_from_dict(new_dict)
    rep = impact_analysis(old, new, policy)

    manifest_path = tmp_path / "new.yaml"
    manifest_path.write_text(yaml.safe_dump(new_dict, sort_keys=False))
    policy_path = tmp_path / "new_policy.json"
    rep.resynthesis.policy.save(policy_path)
    _, test_file = generate_regression_tests(rep.all_test_witnesses(), manifest_path,
                                             policy_path, tmp_path / "generated", "drift")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test_file)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # The same tests must FAIL against the old (insufficient) policy.
    old_policy_for_new = type(policy).for_manifest(new, policy.gates)
    old_policy_for_new.save(policy_path)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test_file)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
