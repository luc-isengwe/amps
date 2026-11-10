import os
import sys
import time
import json
import logging
import yaml
from pathlib import Path
from typing import Any
from rich.logging import RichHandler

# Ensure experiments directory is in sys.path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

# =====================================================================
# SIMULATED HUMAN IMPORTS
# =====================================================================
from simulated_human import (
    load_ground_truth,
    find_task,
    attentive_approves,
    naive_approves,
)

GT = load_ground_truth()

# =====================================================================
# AMPS IMPORTS
# =====================================================================
from amps.manifest import load_manifest
from amps.policy import Policy
from amps.runtime import PolicyEngine, project_output

# =====================================================================
# AGENTDOJO 0.1.35 IMPORTS
# =====================================================================
from agentdojo.agent_pipeline import (
    AgentPipeline,
    BasePipelineElement,
    SystemMessage,
    InitQuery,
    ToolsExecutionLoop,
)
from agentdojo.agent_pipeline.agent_pipeline import (
    load_system_message,
    get_llm,
)
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime, Env
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.benchmark import benchmark_suite_with_injections
from agentdojo.attacks import load_attack
from agentdojo.logging import OutputLogger

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# =====================================================================
# HUMAN TOGGLE AND DYNAMIC CONSTANTS
# =====================================================================
HUMAN = "attentive"  # Options: "attentive" or "naive"

MANIFEST_PATH = "examples/banking_session_H.yaml"
POLICY_PATH = "results/banking_policy_BH.json"
OLLAMA_MODEL_ID = "qwen3:4b-instruct-2507-q4_K_M"

suffix = "" if HUMAN == "attentive" else "_naive"
RUN_FOLDER_NAME = f"qwen3-4b-instruct_BH{suffix}"
PIPELINE_NAME = f"local-amps_BH{suffix}"
LOG_JSONL_NAME = f"{RUN_FOLDER_NAME}_tool_knowledge_attack_calls.jsonl"
PREFER_PROJECTION = False

MANIFEST = load_manifest(MANIFEST_PATH)
POLICY = Policy.load(POLICY_PATH)

# Global episode state variables
current_engine: PolicyEngine = None
current_task_id: str = "unknown_task"
pending_call: dict[str, Any] = {"tool": "", "args": {}}


# =====================================================================
# CONFIRMATION CALLBACK WITH TOGGLE
# =====================================================================
def simulated_human_confirmation_callback(agent: str, tool_name: str, label: str) -> bool:
    tool = pending_call.get("tool", tool_name)
    args = pending_call.get("args", {})
    
    if HUMAN == "naive":
        return naive_approves(current_task_id, tool, args, GT)
    return attentive_approves(current_task_id, tool, args, GT)


# =====================================================================
# ELEMENT 1: AMPS EPISODE INITIALIZER
# =====================================================================
class AmpsEpisodeInitializer(BasePipelineElement):
    """Instantiates a fresh PolicyEngine for each episode."""

    def __init__(self, prefer_projection: bool = False, enable_human_mode: bool = True):
        super().__init__()
        self.prefer_projection = prefer_projection
        self.enable_human_mode = enable_human_mode

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: list[dict[str, Any]],
        extra_args: dict[str, Any],
    ) -> tuple[str, FunctionsRuntime, Env, list[dict[str, Any]], dict[str, Any]]:
        global current_engine, current_task_id

        current_task_id = find_task(query, GT)
        if current_task_id is None:
            logging.warning("[AMPS WARNING] Unable to match prompt query to a known user_task ID in Ground Truth!")

        confirm_func = simulated_human_confirmation_callback if self.enable_human_mode else None
        current_engine = PolicyEngine(
            manifest=MANIFEST,
            policy=POLICY,
            confirm=confirm_func,
            prefer_projection=self.prefer_projection,
        )

        return query, runtime, env, messages, extra_args


# =====================================================================
# ELEMENT 2: AMPS SECURE TOOL INTERCEPTOR
# =====================================================================
class AmpsToolExecutionElement(BasePipelineElement):
    """Intercepts tool calls, records arguments, and logs call history."""

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: list[dict[str, Any]],
        extra_args: dict[str, Any],
    ) -> tuple[str, FunctionsRuntime, Env, list[dict[str, Any]], dict[str, Any]]:
        global current_engine, pending_call

        last_message = messages[-1] if messages else None
        if not last_message or last_message.get("role") != "assistant" or not last_message.get("tool_calls"):
            return query, runtime, env, messages, extra_args

        tool_calls = last_message["tool_calls"]
        tool_result_messages = []

        for tool_call in tool_calls:
            tool_call_id = getattr(tool_call, "id", None)
            tool_name = getattr(tool_call, "function", "") or ""
            raw_args = getattr(tool_call, "args", {}) or {}

            tool_args = {}
            if isinstance(raw_args, dict):
                for k, v in raw_args.items():
                    if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                        try:
                            tool_args[k] = yaml.safe_load(v)
                        except Exception:
                            tool_args[k] = v
                    else:
                        tool_args[k] = v
            else:
                tool_args = raw_args

            pending_call = {"tool": tool_name, "args": tool_args}

            if not tool_name:
                err_msg = "Tool name is empty."
                tool_result_messages.append(
                    ChatToolResultMessage(
                        role="tool",
                        tool_call_id=tool_call_id,
                        tool_call=tool_call,
                        content=[text_content_block_from_string("")],
                        error=err_msg,
                    )
                )
                continue

            if not hasattr(runtime, "functions") or tool_name not in runtime.functions:
                err_msg = f"Tool '{tool_name}' not found."
                tool_result_messages.append(
                    ChatToolResultMessage(
                        role="tool",
                        tool_call_id=tool_call_id,
                        tool_call=tool_call,
                        content=[text_content_block_from_string("")],
                        error=err_msg,
                    )
                )
                continue

            decision = current_engine.before_tool_call("assistant", tool_name)

            log_dir = Path("results")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / LOG_JSONL_NAME

            call_log = {
                "task": current_task_id,
                "tool": tool_name,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "args": tool_args,
            }
            # Add default=str to prevent JSON serialization errors
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(call_log, default=str) + "\n")

            if not decision.allowed:
                error_msg = "Denied by security policy (AMPS Policy Block)."
                logging.info(f"[AMPS BLOCK] Tool '{tool_name}' blocked. Reason: {decision.reason}")

                result_msg = ChatToolResultMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    content=[text_content_block_from_string(error_msg)],
                    error=error_msg,
                )
                tool_result_messages.append(result_msg)
            else:
                raw_output, err_from_runtime = runtime.run_function(env, tool_name, tool_args)

                if err_from_runtime is None and "view" in getattr(decision, "details", {}):
                    projected_data = project_output(MANIFEST, tool_name, raw_output)
                    formatted_output = yaml.safe_dump(projected_data, default_flow_style=False)
                else:
                    formatted_output = tool_result_to_str(raw_output)

                result_msg = ChatToolResultMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    content=[text_content_block_from_string(formatted_output)],
                    error=err_from_runtime,
                )
                tool_result_messages.append(result_msg)

                current_engine.after_tool_result("assistant", tool_name)

        new_messages = list(messages) + tool_result_messages
        return query, runtime, env, new_messages, extra_args


# =====================================================================
# PIPELINE CREATION
# =====================================================================
def create_local_amps_pipeline(
    ollama_model_id: str = OLLAMA_MODEL_ID,
) -> AgentPipeline:
    llm = get_llm("local", "local", ollama_model_id, "tool")
    default_sys_msg = SystemMessage(load_system_message(None))

    elements = [
        default_sys_msg,
        InitQuery(),
        AmpsEpisodeInitializer(prefer_projection=PREFER_PROJECTION, enable_human_mode=True),
        llm,
        ToolsExecutionLoop([AmpsToolExecutionElement(), llm]),
    ]

    pipeline = AgentPipeline(elements)
    pipeline.name = PIPELINE_NAME
    return pipeline


# =====================================================================
# MAIN ATTACK BENCHMARK EXECUTION (144 Cases)
# =====================================================================
if __name__ == "__main__":
    OUTPUT_DIR = Path("runs") / RUN_FOLDER_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"--> Mode: {HUMAN.upper()} | Pipeline: {PIPELINE_NAME}")
    logging.info("--> Loading banking suite v1.2.2...")
    suite = get_suite("v1.2.2", "banking")

    # Order corrected: instantiate pipeline first, then load_attack with 3 arguments
    pipeline = create_local_amps_pipeline(ollama_model_id=OLLAMA_MODEL_ID)

    logging.info("--> Loading attack 'tool_knowledge'...")
    attack = load_attack("tool_knowledge", suite, pipeline)

    logging.info(f"--> Running attack benchmark in {OUTPUT_DIR} (force_rerun=False)...")

    with OutputLogger(str(OUTPUT_DIR)):
        benchmark_suite_with_injections(
            agent_pipeline=pipeline,
            suite=suite,
            attack=attack,
            logdir=OUTPUT_DIR,
            force_rerun=False,
            user_tasks=["user_task_0", "user_task_2", "user_task_12", "user_task_13"],
            injection_tasks=None,
            benchmark_version="v1.2.2",
        )

    logging.info("--> Attack benchmark execution completed successfully.")