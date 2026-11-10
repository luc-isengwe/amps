"""The AgentDojo banking manifest validates, and its synthesized policy is secure."""
from pathlib import Path

from amps.configs import CONFIGS
from amps.graph import compile_manifest
from amps.manifest import load_manifest
from amps.verify import verify
from amps.workflows import label_violations, task_workflows

BANKING = Path(__file__).resolve().parents[1] / "examples" / "banking.yaml"


def test_banking_policy_is_secure_and_label_safe_tasks_are_read_only():
    m = load_manifest(BANKING)
    cg = compile_manifest(m)
    assert not verify(cg, CONFIGS["A"](cg)).safe      # undefended: attack paths exist
    assert verify(cg, CONFIGS["B"](cg)).safe          # synthesized: none left
    safe = [wf.tasks[0] for wf in task_workflows(m) if not label_violations(m, wf)]
    assert safe == ["user_task_1", "user_task_7", "user_task_8", "user_task_10"]
