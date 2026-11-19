"""Shared setup and helpers for the evaluation scripts.

Import this module before anything from deepeval: it loads .env and sets the
judge's timeouts, which deepeval reads from the environment.
"""

import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# Judge retries and rate limiting live in model.py (PacedGeminiModel): deepeval's
# own Gemini retry policy never activates in 4.3.4, so turn it off explicitly.
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "1")
# Each judge call gets 60s. The per-metric budget must also cover the pacing
# and retry waits done in PacedGeminiModel (up to ~1 minute per 429), so it is
# set generously instead of letting deepeval derive a tight one.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "60")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900")

# The agent code lives in ../agent and uses flat imports (from tools import ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from groq import RateLimitError  # noqa: E402
from deepeval.integrations.langchain import CallbackHandler  # noqa: E402
from deepeval.test_case import ToolCall  # noqa: E402
from deepeval.tracing.context import current_trace_context  # noqa: E402
from deepeval.tracing.types import TraceSpanStatus  # noqa: E402

from agent import agent, agent_tools, final_reply  # noqa: E402

GOLDEN_COOLDOWN_SECONDS = 20  # lets Groq's per-minute limit recover between goldens


def run_agent(user_input: str, *, trace: bool = False, tries: int = 4) -> dict | None:
    """Invoke the agent on a fresh thread, retrying on failures.

    With trace=True a deepeval CallbackHandler is attached so trace-based
    metrics can score the run. Returns None if every attempt fails.
    """
    for attempt in range(1, tries + 1):
        config = {"configurable": {"thread_id": str(uuid4())}}  # fresh thread per attempt
        if trace:
            config["callbacks"] = [CallbackHandler()]
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            if get_answer(response) is not None:
                return response
            # The model occasionally ends its turn with an empty message,
            # leaving nothing to score: try again.
            print(f"Attempt {attempt}/{tries} ended without a reply to the customer")
            time.sleep(2)
        except RateLimitError as e:
            wait = 30 * attempt  # 30s, 60s, 90s, 120s
            print(f"Rate limited (attempt {attempt}/{tries}), waiting {wait}s: {str(e)[:150]}")
            time.sleep(wait)
        except Exception as e:
            print(f"Attempt {attempt}/{tries} failed: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(2)
    print(f"Giving up after {tries} attempts: {user_input}")
    if trace:
        mark_current_trace_failed()
    return None


def mark_current_trace_failed() -> None:
    """Make deepeval skip trace metrics for a run that never finished.

    run_agent catches the agent's exceptions, so the trace of the last failed
    attempt is not marked as errored and deepeval would otherwise spend judge
    calls scoring a broken run. deepeval skips traces whose status is ERRORED.
    """
    trace = current_trace_context.get()
    if trace is not None:
        trace.status = TraceSpanStatus.ERRORED


def get_answer(response: dict) -> str | None:
    """The agent's final reply to the customer, or None if its turn ended without one."""
    return final_reply(response)


TOOL_DESCRIPTIONS = {t.name: t.description.strip() for t in agent_tools}


def extract_tool_calls(response: dict, *, with_arguments_only: bool = False) -> list[ToolCall]:
    """All tool calls the agent made, in order.

    with_arguments_only drops calls to tools that take no arguments (e.g.
    check_current_date). ArgumentCorrectnessMetric's prompt tells the judge to
    fail any call with no input parameters, so those calls would always count
    as wrong even though there is nothing to get wrong.
    """
    tools_called = []
    for message in response["messages"]:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if with_arguments_only and not tool_call["args"]:
                continue
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    description=TOOL_DESCRIPTIONS.get(tool_call["name"]),
                    input_parameters=tool_call["args"],
                )
            )
    return tools_called
