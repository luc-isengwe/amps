import pytest

from amps.configs import CONFIGS
from amps.graph import compile_manifest
from amps.guard import Action, Guard
from amps.manifest import load_manifest
from amps.metrics import utility_summary, wilson
from amps.runtime import PolicyEngine
from amps.verify import verify
from amps.workflows import mixed_workflows, run_workflow, task_workflows
from amps_demo.app import run_plan
from amps_demo.tools import TOOLS


@pytest.fixture(scope="module")
def setup(demo_path=None):
    from conftest import DEMO
    m = load_manifest(DEMO)
    cg = compile_manifest(m)
    return m, cg, {name: make(cg) for name, make in CONFIGS.items()}


def test_wilson_interval_known_values():
    lo, hi = wilson(5, 5)
    assert lo == pytest.approx(0.5655, abs=1e-3) and hi == 1.0
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-3)
    lo, hi = wilson(15, 20)
    assert lo < 0.75 < hi


def test_workflows_are_built_from_tasks(setup):
    m, _, _ = setup
    tasks = task_workflows(m)
    assert [w.tasks[0] for w in tasks] == [t.id for t in m.tasks]
    assert len(mixed_workflows(m)) == len(m.tasks) ** 2  # ordered pairs, repeats included
    reply = next(w for w in tasks if w.tasks == ["reply_to_customer"])
    assert reply.actions == [
        Action("tool", "retrieval", "crm_lookup"),
        Action("send", "retrieval", "supervisor"),
        Action("send", "supervisor", "executor"),
        Action("tool", "executor", "send_email"),
    ]


def test_every_single_task_succeeds_under_synthesized_policy(setup):
    m, _, configs = setup
    for wf in task_workflows(m):
        assert run_workflow(m, configs["B"], wf, "B", TOOLS).success, wf.id


def test_configuration_A_blocks_nothing(setup):
    m, _, configs = setup
    for wf in task_workflows(m) + mixed_workflows(m):
        assert run_workflow(m, configs["A"], wf, "A", TOOLS).success


def test_synthesized_policy_is_secure_but_sink_heuristic_is_not(setup):
    _, cg, configs = setup
    assert verify(cg, configs["B"]).safe
    assert verify(cg, configs["B_strict"]).safe
    open_sinks = verify(cg, configs["B_sink"]).witnesses
    assert [w.property for w in open_sinks] == ["confidentiality"]


def test_langgraph_and_direct_runner_agree(setup):
    m, _, configs = setup
    for name in ("A", "B", "B_strict"):
        for wf in task_workflows(m) + mixed_workflows(m):
            direct = run_workflow(m, configs[name], wf, name, TOOLS)
            log, _ = run_plan(m, configs[name], wf.actions)
            assert [e["allowed"] for e in log] == [o.allowed for o in direct.outcomes], (name, wf.id)


def test_langgraph_routes_actions_to_their_agent(setup):
    m, _, configs = setup
    wf = next(w for w in task_workflows(m) if w.tasks == ["reply_to_customer"])
    log, guard = run_plan(m, configs["B"], wf.actions)
    assert [e["node"] for e in log] == ["retrieval", "retrieval", "supervisor", "executor"]
    assert guard.inbox["executor"] == ["message from supervisor"]


def test_utility_summary_counts(setup):
    m, _, configs = setup
    results = [run_workflow(m, configs["B"], wf, "B", TOOLS)
               for wf in task_workflows(m) + mixed_workflows(m)]
    s = utility_summary(results)
    assert s["UA_task"].k == s["UA_task"].n == 5
    assert s["UA_mixed"].n == 25 and s["UA_mixed"].k < 25  # unsafe pairs are blocked
    assert s["UA_safe"].k == s["UA_safe"].n          # but every label-safe workflow passes
    assert s["unsafe_passed"].k == 0
    assert s["check_us_per_action"] > 0


def test_guard_never_runs_a_denied_tool():
    from conftest import DEMO
    m = load_manifest(DEMO)
    calls = []
    guard = Guard(PolicyEngine(m, CONFIGS["B_strict"](compile_manifest(m)), strict_manifest=False),
                  {"web_search": lambda a: calls.append(a) or {}, "send_email": lambda a: calls.append(a)})
    guard.execute(Action("tool", "retrieval", "web_search"))      # allowed, taints retrieval
    out = guard.execute(Action("tool", "retrieval", "web_search"))  # strict: blocked now
    assert not out.allowed and len(calls) == 1
