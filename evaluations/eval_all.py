import sys
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from groq import RateLimitError
from agent import agent
from dataset import dataset
from model import eval_model
from deepeval import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall


SUCCESSFUL_TASK_DESCRIPTION = """
If the username is missing, ask the user to provide it.
For a refund request, identify the customer and the requested purchases, and
check that each purchase is no more than 30 days old and is not in the future.
Refund only the items requested by the user. If the customer or requested items
cannot be found, do not issue a refund ticket and explain why.
Once the necessary information is available, tell the user whether a refund
ticket was issued and its amount, or explain why no refund was issued.
For an information-only request, answer the question without issuing a ticket.
"""


def build_metrics():
    """Trace-based metrics (need the callback trace) and single-turn metrics."""
    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
        ),
        StepEfficiencyMetric(threshold=0.8, model=eval_model, include_reason=True),
        PlanQualityMetric(threshold=0.8, model=eval_model, include_reason=True),
        PlanAdherenceMetric(threshold=0.8, model=eval_model, include_reason=True),
    ]

    single_turn_metrics = [
        ArgumentCorrectnessMetric(threshold=0.9, model=eval_model, include_reason=True),
        ToolCorrectnessMetric(threshold=0.9, model=eval_model, include_reason=True),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Check that the actual output and expected output have the same "
                "logical conclusion about the user's request.",
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics


def run_agent(user_input: str, tries: int = 4):
    for attempt in range(tries):
        try:
            return agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={
                    "configurable": {"thread_id": str(uuid4())},  # fresh thread per attempt
                    "callbacks": [CallbackHandler()],
                },
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


def main():
    trace_metrics, single_turn_metrics = build_metrics()
    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    async_config = AsyncConfig(run_async=False)  # one golden at a time
    error_config = ErrorConfig(skip_on_missing_params=True)

    test_cases = []  # local list, so repeated runs don't pile up in dataset.test_cases

    # The callback captures the trace needed by the four trace-based metrics.
    for golden in dataset.evals_iterator(
        metrics=trace_metrics,
        async_config=async_config,
        error_config=error_config,
        cache_config=cache_config,
        display_config=display_config,
    ):
        response = run_agent(golden.input)
        time.sleep(20)  # let Groq's per-minute limit recover before the next golden

        if response is None:
            print(f"SKIPPED (agent never finished): {golden.input}")
            continue

        tools_called = []
        for message in response["messages"]:
            for tool_call in getattr(message, "tool_calls", []) or []:
                if tool_call["name"] == "final_answer":
                    continue  # structured-output tool, not a business tool
                tools_called.append(
                    ToolCall(
                        name=tool_call["name"],
                        input_parameters=tool_call["args"],
                    )
                )

        # Reuse the same agent response for the three single-turn metrics.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=response["structured_response"].answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=single_turn_metrics,
        async_config=async_config,
        cache_config=cache_config,
        display_config=display_config,
    )

    print(f"\nCompleted all 7 metrics for {len(test_cases)} test cases")


if __name__ == "__main__":
    main()