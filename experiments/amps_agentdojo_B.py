import os
import time
import logging
import yaml
from pathlib import Path
from typing import Any
from rich.logging import RichHandler

# =====================================================================
# AMPS IMPORTS (Strict, no try/except)
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
from agentdojo.benchmark import benchmark_suite_without_injections
from agentdojo.logging import OutputLogger

# =====================================================================
# LOGGING CONFIGURATION (Identical to official script)
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# =====================================================================
# INITIAL CONFIGURATION AND LOADING (AMPS)
# =====================================================================
MANIFEST_PATH = "examples/banking.yaml"
POLICY_PATH = "results/banking_policy_B.json"

MANIFEST = load_manifest(MANIFEST_PATH)
POLICY = Policy.load(POLICY_PATH)

# Global session tracking variables
current_engine: PolicyEngine = None
current_task_name: str = "unknown_task"
current_tool_args: dict[str, Any] = {}


# =====================================================================
# HUMAN-IN-THE-LOOP CALLBACK (AMPS Signature)
# =====================================================================
def human_confirmation_callback(agent: str, tool_name: str, label: str) -> bool:
    """Terminal confirmation callback for Mode H."""
    print("\n[MODE H — CONFIRMATION REQUIRED]")
    print(f"-> Agent / Role   : {agent}")
    print(f"-> Current task   : {current_task_name}")
    print(f"-> Proposed tool  : '{tool_name}' (Label: {label})")
    print(f"-> Arguments      : {current_tool_args}")

    choice = input("Allow this action? (yes/no): ").strip().lower()
    allowed = choice in ["o", "oui", "yes", "y"]

    log_path = Path("runs/human_decisions_log.csv")
    if not log_path.exists():
        log_path.write_text("timestamp,task,tool,label,arguments,decision\n")

    with open(log_path, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f'"{timestamp}", "{current_task_name}", "{tool_name}", "{label}", "{str(current_tool_args)}", "{allowed}"\n'
        )

    return allowed


# =====================================================================
# ELEMENT 1: AMPS EPISODE INITIALIZER
# =====================================================================
class AmpsEpisodeInitializer(BasePipelineElement):
    """Instantiates a fresh PolicyEngine for each episode."""

    def __init__(self, prefer_projection: bool = True, enable_human_mode: bool = False):
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
        global current_engine, current_task_name

        current_task_name = extra_args.get("task_name", "user_task_0")

        confirm_func = human_confirmation_callback if self.enable_human_mode else None
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
    """Intercepts and controls tool execution via AMPS."""

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: list[dict[str, Any]],
        extra_args: dict[str, Any],
    ) -> tuple[str, FunctionsRuntime, Env, list[dict[str, Any]], dict[str, Any]]:
        global current_engine, current_tool_args

        last_message = messages[-1] if messages else None
        if not last_message or last_message.get("role") != "assistant" or not last_message.get("tool_calls"):
            return query, runtime, env, messages, extra_args

        tool_calls = last_message["tool_calls"]
        tool_result_messages = []

        for tool_call in tool_calls:
            tool_call_id = getattr(tool_call, "id", None)
            tool_name = getattr(tool_call, "function", "") or ""
            raw_args = getattr(tool_call, "args", {}) or {}

            # Conversion identical to default AgentDojo element:
            # Iterates over each argument and only converts strings representing lists
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

            current_tool_args = tool_args

            # -------------------------------------------------------------
            # AGENTDOJO CHECKS (1st: Empty name, 2nd: Tool existence)
            # -------------------------------------------------------------
            if not tool_name:
                err_msg = "Tool name is empty."
                result_msg = ChatToolResultMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    content=[text_content_block_from_string("")],
                    error=err_msg,
                )
                tool_result_messages.append(result_msg)
                continue

            if not hasattr(runtime, "functions") or tool_name not in runtime.functions:
                err_msg = f"Tool '{tool_name}' not found."
                result_msg = ChatToolResultMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    content=[text_content_block_from_string("")],
                    error=err_msg,
                )
                tool_result_messages.append(result_msg)
                continue

            # -------------------------------------------------------------
            # AMPS CHECK (Executed only for a valid tool)
            # -------------------------------------------------------------
            decision = current_engine.before_tool_call("assistant", tool_name)

            if not decision.allowed:
                error_msg = "Denied by security policy (AMPS Policy Block)."
                logging.info(f"[AMPS BLOCK] Tool '{tool_name}' successfully blocked.")

                result_msg = ChatToolResultMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    content=[text_content_block_from_string(error_msg)],
                    error=error_msg,
                )
                tool_result_messages.append(result_msg)
            else:
                # Actual execution
                raw_output, err_from_runtime = runtime.run_function(env, tool_name, tool_args)

                # Output formatting
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

                # AMPS notification only after actual execution
                current_engine.after_tool_result("assistant", tool_name)

        new_messages = list(messages) + tool_result_messages
        return query, runtime, env, new_messages, extra_args


# =====================================================================
# STEP 6: PIPELINE CONSTRUCTION (local-amps_B)
# =====================================================================
def create_local_amps_pipeline(
    ollama_model_id: str = "qwen3:4b-instruct-2507-q4_K_M",
) -> AgentPipeline:
    """Assembles the exact Run A pipeline with the AMPS interceptor."""
    
    llm = get_llm("local", "local", ollama_model_id, "tool")
    
    # Correct call of load_system_message with None
    default_sys_msg = SystemMessage(load_system_message(None))

    elements = [
        default_sys_msg,
        InitQuery(),
        AmpsEpisodeInitializer(prefer_projection=True, enable_human_mode=False),
        llm,
        ToolsExecutionLoop([AmpsToolExecutionElement(), llm]),
    ]

    pipeline = AgentPipeline(elements)
    pipeline.name = "local-amps_B"
    return pipeline


# =====================================================================
# STEP 7: BENCHMARK EXECUTION ON USER_TASK_0
# =====================================================================
if __name__ == "__main__":
    OLLAMA_MODEL_ID = "qwen3:4b-instruct-2507-q4_K_M"
    RUN_FOLDER_NAME = "qwen3-4b-instruct_B"
    
    OUTPUT_DIR = Path("runs") / RUN_FOLDER_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("--> Loading banking suite v1.2.2...")
    suite = get_suite("v1.2.2", "banking")

    logging.info("--> Instantiating pipeline local-amps_B...")
    pipeline = create_local_amps_pipeline(ollama_model_id=OLLAMA_MODEL_ID)

    logging.info(f"--> Running user_task_0 in {OUTPUT_DIR}...")
    
    with OutputLogger(str(OUTPUT_DIR)):
        benchmark_suite_without_injections(
            agent_pipeline=pipeline,
            suite=suite,
            logdir=OUTPUT_DIR,
            force_rerun=True,
            user_tasks=None,
            benchmark_version="v1.2.2",
        )
    
    logging.info("--> Execution completed successfully.")