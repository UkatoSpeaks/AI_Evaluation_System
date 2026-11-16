import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent import agent
from dataset import dataset
from model import eval_model
from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler
from deepeval.evaluate.configs import AsyncConfig

SUCCESSFUL_TASK_DESCRIPTION = """
The agent is a refund support agent. The task is successful when:

1. The user does not give a username: the agent asks for the username.

2. The user gives a username and asks for a refund. The agent must check
   (it does not need to tell the user about these checks unless they are
   the reason a refund is denied):
   - the purchase was made within 30 days of today's date
   - the refund covers only the items the user asked for
   - the refund covers only items the user actually purchased
   Then the agent tells the user one of these (only if the user was identified):
   a) a refund ticket was issued, with the amount
   b) no refund was issued, and why

3. The user is not asking for a refund (for example, only wants to know what
   they purchased): the task is successful as long as no refund ticket is issued.
"""

metrics = [
    TaskCompletionMetric(
        threshold=0.8,
        model=eval_model,
        task=SUCCESSFUL_TASK_DESCRIPTION,
        include_reason=True,
    )
]


def run_agent(user_input: str, tries: int = 3):
    for attempt in range(tries):
        try:
            return agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={
                    "configurable": {"thread_id": str(uuid4())},  # fresh thread per attempt
                    "callbacks": [CallbackHandler()],
                },
            )
        except Exception as e:
            print(f"Attempt {attempt + 1}/{tries} failed: {type(e).__name__}")
            time.sleep(2)
    print(f"Giving up after {tries} attempts: {user_input}")


for golden in dataset.evals_iterator(
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),  # one golden at a time
):
    run_agent(golden.input)