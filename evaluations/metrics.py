"""Single source of truth for every metric's configuration."""

from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import SingleTurnParams

from model import eval_model

SUCCESSFUL_TASK_DESCRIPTION = """
If the username is missing, ask the user to provide it.
For a refund request, identify the customer and the requested purchases, and
check that each purchase was made 30 days ago or less and is not dated in the future.
Refund only the items requested by the user. If the customer or requested items
cannot be found, do not issue a refund ticket and explain why.
Once the necessary information is available, tell the user whether a refund
ticket was issued and its amount, or explain why no refund was issued.
For an information-only request, answer the question without issuing a ticket.
"""

# Need the LangChain callback trace (run through dataset.evals_iterator).
TRACE_METRICS = {
    "task_completion": lambda: TaskCompletionMetric(
        threshold=0.8,
        model=eval_model,
        task=SUCCESSFUL_TASK_DESCRIPTION,
        include_reason=True,
    ),
    "step_efficiency": lambda: StepEfficiencyMetric(
        threshold=0.8, model=eval_model, include_reason=True
    ),
    "plan_quality": lambda: PlanQualityMetric(
        threshold=0.8, model=eval_model, include_reason=True
    ),
    "plan_adherence": lambda: PlanAdherenceMetric(
        threshold=0.8, model=eval_model, include_reason=True
    ),
}

# Scored on LLMTestCase objects built from the agent's response.
SINGLE_TURN_METRICS = {
    "argument_correctness": lambda: ArgumentCorrectnessMetric(
        threshold=0.9, model=eval_model, include_reason=True
    ),
    "tool_correctness": lambda: ToolCorrectnessMetric(
        threshold=0.9, model=eval_model, include_reason=True
    ),
    "answer_correctness": lambda: GEval(
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
    ),
}

ALL_METRICS = [*TRACE_METRICS, *SINGLE_TURN_METRICS]
