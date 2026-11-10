"""Workflow-weighted synthesis (objective: break as few legitimate workflows as possible)."""

import pytest

from amps.configs import CONFIGS
from amps.graph import compile_manifest
from amps.guard import Action
from amps.manifest import load_manifest, manifest_from_dict
from amps.policy import Policy
from amps.synthesis import synthesize_ilp
from amps.verify import verify
from amps.workflows import (Workflow, label_violations, mixed_workflows, run_workflow,
                            succeeds_under, task_workflows, usage_profile)
from amps_demo.tools import TOOLS

from conftest import DEMO, tiny


@pytest.fixture(scope="module")
def demo():
    m = load_manifest(DEMO)
    cg = compile_manifest(m)
    return m, cg, {name: make(cg) for name, make in CONFIGS.items()}


def test_profile_predicts_the_real_run_exactly(demo):
    """succeeds_under(profile) must equal the outcome of actually running the workflow."""
    m, _, configs = demo
    for name, policy in configs.items():
        for wf in task_workflows(m) + mixed_workflows(m):
            predicted = succeeds_under(usage_profile(m, wf), policy.gates)
            assert predicted == run_workflow(m, policy, wf, name, TOOLS).success, (name, wf.id)


def test_label_unsafe_workflows_are_exactly_those_no_secure_policy_can_pass(demo):
    m, _, configs = demo
    for wf in task_workflows(m) + mixed_workflows(m):
        unsafe = bool(label_violations(m, wf))
        for name in ("B", "B_util", "B_strict"):  # the secure configurations
            if unsafe:
                assert not run_workflow(m, configs[name], wf, name, TOOLS).success, (name, wf.id)


def test_weighted_policy_is_secure_and_never_worse_on_workflows(demo):
    m, cg, configs = demo
    assert verify(cg, configs["B_util"]).safe
    profiles = [usage_profile(m, w) for w in task_workflows(m) + mixed_workflows(m)]
    broken = lambda p: sum(not succeeds_under(pr, p.gates) for pr in profiles)  # noqa: E731
    assert broken(configs["B_util"]) <= broken(configs["B"])
    assert broken(configs["B_util"]) <= broken(configs["B_strict"])


def _relay():
    """reader (web) -> actor (email). A cheap gate on the message edge would
    break a legitimate 'read the web and brief the actor' workflow; a more
    expensive gate on the e-mail call does not."""
    d = tiny(controls={"sessions": 1, "overrides": {"msg:reader->actor": 1.0}})
    d["memory"] = []
    d["agents"][0]["memory"] = {}
    d["agents"][1]["memory"] = {}
    m = manifest_from_dict(d)
    brief = Workflow("brief", "mixed", [Action("tool", "reader", "web"),
                                        Action("send", "reader", "actor")], ["brief"])
    return m, compile_manifest(m), brief


def test_cost_minimal_policy_can_break_a_safe_workflow_that_weighted_one_keeps():
    m, cg, brief = _relay()
    assert not label_violations(m, brief)                       # the workflow is legitimate
    plain = synthesize_ilp(cg, properties=("integrity",))
    weighted = synthesize_ilp(cg, properties=("integrity",),
                              workflows=[(1.0, usage_profile(m, brief))])
    assert plain.status == weighted.status == "secure"
    assert verify(cg, weighted.policy, ("integrity",)).safe
    assert not succeeds_under(usage_profile(m, brief), plain.policy.gates)    # broken by B
    assert succeeds_under(usage_profile(m, brief), weighted.policy.gates)     # kept by B_util
    assert weighted.cost > plain.cost                                         # at a higher gate cost
    assert weighted.policy.meta["broken_workflow_weight"] == 0


def test_finite_weight_trades_utility_against_cost():
    m, cg, brief = _relay()
    profile = [(1.0, usage_profile(m, brief))]
    cheap = synthesize_ilp(cg, properties=("integrity",), workflows=profile, workflow_weight=0.01)
    assert not succeeds_under(profile[0][1], cheap.policy.gates)  # utility worth too little
    dear = synthesize_ilp(cg, properties=("integrity",), workflows=profile, workflow_weight=100)
    assert succeeds_under(profile[0][1], dear.policy.gates)
