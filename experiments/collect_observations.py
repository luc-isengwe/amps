import json
from pathlib import Path
from typing import Any

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from pydantic import BaseModel

# Read tools that no ground-truth call of the banking user tasks uses.
# They take no argument, so they are called directly (step 4).
DIRECT_TOOLS = ["get_iban", "get_balance", "get_user_info"]


def to_json_serializable(val: Any) -> Any:
    """Recursively convert AgentDojo outputs (Pydantic models, lists, dicts) to plain JSON types."""
    if isinstance(val, BaseModel) or hasattr(val, "model_dump"):
        return to_json_serializable(val.model_dump())
    elif isinstance(val, (list, tuple)):
        return [to_json_serializable(item) for item in val]
    elif isinstance(val, dict):
        return {str(k): to_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (int, float, str, bool)) or val is None:
        return val
    else:
        return str(val)


def record_call(calls, task_id, env, runtime, function, args):
    """Run one tool call without any LLM and record the observation."""
    state_before = env.model_dump_json()
    output, error = runtime.run_function(env, function, args)
    state_after = env.model_dump_json()
    calls.append({
        "task": task_id,
        "tool": function,
        "output": to_json_serializable(output),
        "error": str(error) if error is not None else None,
        "state_changed": state_before != state_after,
    })


def main():
    # 1. Load the AgentDojo suite
    suite = get_suite("v1.2.2", "banking")
    runtime = FunctionsRuntime(suite.tools)

    # 2. One unique, harmless marker per location written by a third party
    default_vectors = suite.get_injection_vector_defaults()
    canaries = {}
    for i, vector_name in enumerate(sorted(default_vectors.keys())):
        canaries[vector_name] = f"AMPS_CANARY_{i:02d}"

    calls = []

    # 3. Ground-truth calls of every user task
    for task_id, task in suite.user_tasks.items():
        # Fresh marked environment for each task
        env = suite.load_and_inject_default_environment(canaries)
        for call in task.ground_truth(env):
            record_call(calls, task_id, env, runtime, call.function, call.args)

    # 4. Direct calls of the tools that no task uses
    env = suite.load_and_inject_default_environment(canaries)
    for name in DIRECT_TOOLS:
        record_call(calls, "direct_call", env, runtime, name, {})

    # 5. Write the observations to results/banking_observations.json
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "banking_observations.json"

    data = {
        "system": "agentdojo-banking-v1.2.2",
        "canaries": canaries,
        "calls": calls,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    tools = sorted({c["tool"] for c in calls})
    errors = [c for c in calls if c["error"]]
    print(f"Observations collected: {len(calls)} calls, {len(tools)} tools, "
          f"{len(errors)} error(s) -> {output_file}")


if __name__ == "__main__":
    main()