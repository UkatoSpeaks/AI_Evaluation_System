import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from dotenv import load_dotenv

load_dotenv()

from agent import agent
from dataset import dataset
from model import eval_model
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

metrics = [
    ToolCorrectnessMetric(
        threshold=0.8,
        model=eval_model,
        include_reason=True,
    )
]


def run_agent(user_input: str, tries: int = 3):
    for attempt in range(tries):
        try:
            return agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": str(uuid4())}},  # fresh thread per attempt
            )
        except Exception as e:
            print(f"Attempt {attempt + 1}/{tries} failed: {type(e).__name__}")
            time.sleep(2)
    print(f"Giving up after {tries} attempts: {user_input}")
    return None


test_cases = []

for golden in dataset.goldens:
    response = run_agent(golden.input)

    if response is None:
        # the agent crashed on every attempt: record it as an empty run so it scores as a failure
        actual_output = ""
        tools_called = []
    else:
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
            tools_called=tools_called,
            expected_tools=golden.expected_tools,
        )
    )

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
)