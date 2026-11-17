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
from deepeval.metrics import PlanQualityMetric
from deepeval.integrations.langchain import CallbackHandler
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig

metrics = [
    PlanQualityMetric(
        threshold=0.8,
        model=eval_model,
        include_reason=True,
    )
]


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


for golden in dataset.evals_iterator(
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),                # one golden at a time
    error_config=ErrorConfig(skip_on_missing_params=True),    # don't crash on empty runs
):
    run_agent(golden.input)
    time.sleep(20)  # let Groq's per-minute limit recover before the next golden