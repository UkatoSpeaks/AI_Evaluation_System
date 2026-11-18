"""Shared setup and helpers for the evaluation scripts.

Import this module before anything from deepeval: it loads .env and sets the
judge's retry policy, which deepeval reads from the environment.
"""

import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# Retry the Gemini judge on 429s with a real backoff. deepeval's defaults
# (2 attempts, 5s cap) are too short for free-tier per-minute limits.
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "6")
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "5")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "60")

# The agent code lives in ../agent and uses flat imports (from tools import ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from groq import RateLimitError  # noqa: E402
from deepeval.integrations.langchain import CallbackHandler  # noqa: E402
from deepeval.test_case import ToolCall  # noqa: E402

from agent import agent  # noqa: E402

STRUCTURED_OUTPUT_TOOL = "final_answer"  # response-format tool, not a business tool
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
            return agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
        except RateLimitError as e:
            wait = 30 * attempt  # 30s, 60s, 90s, 120s
            print(f"Rate limited (attempt {attempt}/{tries}), waiting {wait}s: {str(e)[:150]}")
            time.sleep(wait)
        except Exception as e:
            print(f"Attempt {attempt}/{tries} failed: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(2)
    print(f"Giving up after {tries} attempts: {user_input}")
    return None


def get_answer(response: dict) -> str | None:
    """The agent's final answer, or None if it never produced a structured response."""
    structured = response.get("structured_response")
    return structured.answer if structured is not None else None


def extract_tool_calls(response: dict) -> list[ToolCall]:
    """All business tool calls the agent made, in order."""
    tools_called = []
    for message in response["messages"]:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call["name"] == STRUCTURED_OUTPUT_TOOL:
                continue
            tools_called.append(
                ToolCall(name=tool_call["name"], input_parameters=tool_call["args"])
            )
    return tools_called
