"""Replay the AgentDojo ground-truth tool calls of each banking user task
through the amps runtime engine, WITHOUT any LLM.

Purpose (validation, no attack content involved):
1. check that examples/banking.yaml describes the real AgentDojo tasks
   (every ground-truth tool is declared for the agent);
2. tell, for each task, which call policy B blocks (if any). This must agree
   with experiments/predict_banking.py and is the reference your AgentDojo
   integration must reproduce on benign runs.

Usage (from the project folder, environment `amps` active):
    python experiments/check_groundtruth_banking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from amps.manifest import load_manifest  # noqa: E402
from amps.policy import Policy  # noqa: E402
from amps.runtime import PolicyEngine  # noqa: E402

AGENT = "assistant"


def main() -> None:
    manifest = load_manifest(ROOT / "examples" / "banking.yaml")
    policy = Policy.load(ROOT / "results" / "banking_policy_B.json")
    suite = get_suite("v1.2.2", "banking")
    # Benign environment: injection placeholders filled with their default (harmless) text.
    env = suite.load_and_inject_default_environment(suite.get_injection_vector_defaults())

    print(f"{'task':14s} {'B':6s} ground-truth calls (X = blocked by B)")
    for tid, task in sorted(suite.user_tasks.items(), key=lambda kv: int(kv[0].split('_')[-1])):
        engine = PolicyEngine(manifest, policy)   # fresh engine = fresh episode
        engine.start_session()
        shown, passed = [], True
        for call in task.ground_truth(env.model_copy(deep=True)):
            d = engine.before_tool_call(AGENT, call.function)
            if d.allowed:
                engine.after_tool_result(AGENT, call.function)  # output enters the context
                shown.append(call.function)
            else:
                passed = False
                shown.append(f"X {call.function} [{d.reason}]")
                break                                             # the episode stops here
        print(f"{tid:14s} {'pass' if passed else 'BLOCK':6s} " + " -> ".join(shown))


if __name__ == "__main__":
    main()
