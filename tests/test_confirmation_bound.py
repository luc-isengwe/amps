"""Lower bound on human confirmations.

Checks, on every example manifest with human confirmation allowed, that the
policy synthesized by amps asks for exactly the minimum number of
confirmations given by ``confirmation_lower_bound`` while completing every
task, both by prediction (usage profiles) and by execution (runtime engine).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from amps.graph import compile_manifest  # noqa: E402
from amps.manifest import manifest_from_dict  # noqa: E402
from amps.synthesis import synthesize_ilp  # noqa: E402
from amps.verify import verify  # noqa: E402
from amps.workflows import (confirmation_lower_bound, confirmations_under,  # noqa: E402
                            run_workflow, succeeds_under, task_workflows, usage_profile)
from banking_fields_study import honest_labels, load, session_level, with_human  # noqa: E402


def _cases():
    fields = load("banking_fields.yaml")
    return {
        "banking_session_H": with_human(session_level(fields)),
        "banking_fields_H": with_human(honest_labels(fields)),
    }


@pytest.mark.parametrize("name", sorted(_cases()))
def test_synthesized_policy_reaches_the_bound(name):
    m = manifest_from_dict(_cases()[name])
    cg = compile_manifest(m)
    res = synthesize_ilp(cg)
    assert verify(cg, res.policy.gates).safe
    bound = confirmation_lower_bound(m)
    wfs = task_workflows(m)
    for wf in wfs:
        prof = usage_profile(m, wf)
        assert succeeds_under(prof, res.policy.gates)
        assert confirmations_under(prof, res.policy.gates) == bound[wf.tasks[0]]
        assert run_workflow(m, res.policy, wf, name).n_confirmations == bound[wf.tasks[0]]


def test_field_labels_lower_the_bound():
    cases = _cases()
    session = sum(confirmation_lower_bound(manifest_from_dict(cases["banking_session_H"])).values())
    field = sum(confirmation_lower_bound(manifest_from_dict(cases["banking_fields_H"])).values())
    assert (session, field) == (12, 8)


def test_bound_is_zero_for_label_safe_tasks():
    # The four-agent demonstrator has only label-safe tasks: no confirmation needed.
    import yaml
    d = yaml.safe_load((ROOT / "examples" / "demo_4agents.yaml").read_text())
    assert sum(confirmation_lower_bound(manifest_from_dict(d)).values()) == 0


def _random_manifest_with_actions(n: int, seed: int) -> dict:
    """Synthetic multi-agent manifest (generator of the scalability study) in
    which each agent also has a task that reads one of its tools and then
    acts through one of its sinks; labels are the honest ones."""
    import random
    from run_scalability import generate
    rng = random.Random(seed)
    d = generate(n, seed)
    m = manifest_from_dict(d)
    tasks = []
    for a in d["agents"]:
        # Prefer an untrusted reader, so that the task really needs a confirmation.
        readers = [t for t in a["tools"] if m.tool(t).output is not None
                   and m.tool(t).output.overall_trust == "untrusted"] or \
                  [t for t in a["tools"] if m.tool(t).output is not None]
        sinks = [t for t in a["tools"] if m.tool(t).is_sink]
        if readers and sinks:
            r, s = rng.choice(readers), rng.choice(sinks)
            tasks.append({"id": f"act_{a['id']}", "requires": [
                {"edge": f"ret:{r}->{a['id']}"}, {"edge": f"call:{a['id']}->{s}"}]})
    d["tasks"] = d["tasks"] + tasks
    d.setdefault("controls", {})["allow_human"] = True
    return honest_labels(d)


@pytest.mark.parametrize("n,seed", [(4, 1), (4, 2), (8, 1), (8, 2), (8, 3)])
def test_bound_reached_on_random_multi_agent_manifests(n, seed):
    m = manifest_from_dict(_random_manifest_with_actions(n, seed))
    cg = compile_manifest(m)
    res = synthesize_ilp(cg)
    assert res.status == "secure" and verify(cg, res.policy.gates).safe
    bound = confirmation_lower_bound(m)
    for wf in task_workflows(m):
        prof = usage_profile(m, wf)
        assert succeeds_under(prof, res.policy.gates)
        assert confirmations_under(prof, res.policy.gates) == bound[wf.tasks[0]]


def test_random_cases_are_not_vacuous():
    # Over the random cases above, the bound is positive for many tasks.
    total = sum(sum(confirmation_lower_bound(manifest_from_dict(
        _random_manifest_with_actions(n, seed))).values())
        for n, seed in [(4, 1), (4, 2), (8, 1), (8, 2), (8, 3)])
    assert total >= 5
