import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from dotenv import load_dotenv

load_dotenv()

from groq import RateLimitError
from agent import agent
from dataset import dataset
from model import eval_model
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, ToolCall, SingleTurnParams

metrics = [
    GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        evaluation_steps=[
            "make sure that the actual output and expected output have the same logical conclusion"
        ],
        threshold=0.8,
        model=eval_model,
        verbose_mode=True
    )
]


def run_agent(user_input: str, tries: int = 4):
    for attempt in range(tries):
        try:
            return agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": str(uuid4())}},  # fresh thread per attempt
            )
        except RateLimitError as e:
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
            print(f"Rate limited (attempt {attempt + 1}/{tries}), waiting {wait}s: {str(e)[:150]}")
            time.sleep(wait)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{tries} failed: {type(e).__name__}")
            time.sleep(2)
    print(f"Giving up after {tries} attempts: {user_input}")
    return None


test_cases = []

for golden in dataset.goldens:
    response = run_agent(golden.input)
    time.sleep(20)  # let Groq's per-minute limit recover before the next golden

    if response is None:
        print(f"SKIPPED (agent never finished): {golden.input}")
        continue  # an empty run has no arguments to judge, so don't score it

    actual_output = response["structured_response"].answer

    tools_called = []
    for message in response["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            if tool_call["name"] == "final_answer":
                continue  # the structured-output tool is not a business tool
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    input_parameters=tool_call["args"],
                )
            )

    test_cases.append(
        LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output
        )
    )

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
)