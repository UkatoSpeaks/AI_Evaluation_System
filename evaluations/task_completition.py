import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent import agent
from dataset import dataset
from model import eval_model
from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler

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

for golden in dataset.evals_iterator(metrics=metrics):
    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": str(uuid4())},
            "callbacks": [CallbackHandler()],
        },
    )