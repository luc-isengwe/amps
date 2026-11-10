import itertools

import pytest

from amps.graph import compile_manifest
from amps.labels import Mode, blocks
from amps.manifest import load_manifest, manifest_from_dict
from amps.synthesis import synthesize, synthesize_ilp, synthesize_mincut
from amps.verify import verify

from conftest import tiny


def test_demo_policy_is_secure_and_respects_tasks(demo_path):
    m = load_manifest(demo_path)
    cg = compile_manifest(m)
    res = synthesize_ilp(cg)
    assert res.status == "secure"
    assert verify(cg, res.policy).safe
    # No gate may block a task requirement.
    for task in m.tasks:
        for req in task.requires:
            assert not blocks(res.policy.mode(req.edge), req.label), (task.id, req.edge)
    # Conditional gates are preferred to removal here.
    assert Mode.REMOVE not in res.policy.gates.values()


def test_synthesis_is_deterministic(demo_path):
    cg = compile_manifest(load_manifest(demo_path))
    assert synthesize_ilp(cg).policy.gates == synthesize_ilp(cg).policy.gates


def _brute_force_optimum(cg):
    """Try every assignment of {none, R, I, C} to every edge (tiny graphs only)."""
    m = cg.manifest
    edges = sorted(cg.templates)
    forbidden = {}
    for t in m.tasks:
        for r in t.requires:
            forbidden.setdefault(r.edge, set()).update(
                {Mode.REMOVE} | {md for md in Mode if blocks(md, r.label)})
    best = None
    for combo in itertools.product([None, *Mode], repeat=len(edges)):
        gates = {e: md for e, md in zip(edges, combo) if md is not None}
        if any(md in forbidden.get(e, ()) for e, md in gates.items()):
            continue
        cost = sum(m.edge_cost(e, md) for e, md in gates.items())
        if best is not None and cost >= best - 1e-9:
            continue
        if verify(cg, gates).safe:
            best = cost
    return best


@pytest.mark.parametrize("tasks", [
    [],
    [{"id": "t1", "requires": [{"edge": "msg:reader->actor", "under": ["U"]}]}],
    [{"id": "t2", "requires": [{"edge": "ret:crm->reader", "under": ["S"]},
                               {"edge": "call:actor->email", "under": ["S"]}]}],
])
def test_ilp_matches_brute_force(tasks):
    d = tiny(tasks=tasks)
    d["memory"] = []                                   # keep the search space small
    d["agents"][0]["memory"] = {}
    d["agents"][1]["memory"] = {}
    cg = compile_manifest(manifest_from_dict(d))
    res = synthesize_ilp(cg)
    assert res.status == "secure"
    assert res.cost == pytest.approx(_brute_force_optimum(cg), abs=1e-6)


@pytest.mark.parametrize("sessions", [1, 3])
def test_mincut_matches_ilp_on_integrity(demo_dict, sessions):
    d = dict(demo_dict)
    d["memory"] = [{"id": "scratchpad", "persistence": "session"},
                   {"id": "shared_notes", "persistence": "session"}]  # no cross-session flow
    d["tasks"] = [t for t in d["tasks"] if t["id"] != "remember_kb_facts"]
    cg = compile_manifest(manifest_from_dict(d), sessions=sessions)
    cut = synthesize_mincut(cg)
    ilp = synthesize_ilp(cg, properties=("integrity",))
    assert cut.status == ilp.status == "secure"
    assert cut.cost == pytest.approx(ilp.cost, abs=1e-4)
    assert verify(cg, cut.policy, ("integrity",)).safe


def test_mincut_refuses_cross_session_graph(demo_path):
    cg = compile_manifest(load_manifest(demo_path))
    with pytest.raises(ValueError):
        synthesize_mincut(cg)
    # 'auto' falls back to the ILP in that case.
    assert synthesize(cg, method="auto", properties=("integrity",)).method == "ilp"


def test_conflict_is_reported_not_hidden():
    # A task that needs untrusted content to drive the e-mail tool.
    d = tiny(tasks=[{"id": "auto_reply", "requires": [
        {"edge": "ret:web->reader", "under": ["U"]},
        {"edge": "msg:reader->actor", "under": ["U"]},
        {"edge": "call:actor->email", "under": ["U"]},
        {"edge": "mw:reader->notes", "under": ["U"]},
        {"edge": "mr:notes->actor", "under": ["U"]},
    ]}])
    res = synthesize_ilp(compile_manifest(manifest_from_dict(d)))
    assert res.status == "conflict"
    c = next(c for c in res.conflicts if c.witness.property == "integrity")
    assert any("auto_reply" in r for r in c.reasons)


def test_fixed_edges_are_never_gated():
    d = tiny(controls={"sessions": 1, "fixed": ["call:actor->email", "msg:reader->actor"]})
    res = synthesize_ilp(compile_manifest(manifest_from_dict(d)))
    assert "call:actor->email" not in res.policy.gates
    assert "msg:reader->actor" not in res.policy.gates
    assert res.status == "secure"
