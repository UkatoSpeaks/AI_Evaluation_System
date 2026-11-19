"""Run the refund agent over the dataset and score it.

Runs all 7 metrics by default; pass --metrics to run a subset, e.g.
    python evaluations/eval_all.py --metrics tool_correctness answer_correctness
"""

import argparse
import time
from dataclasses import asdict

import common  # noqa: F401  (must come first: loads .env and the judge timeouts)
from agent_trace import steps_from_messages
from common import GOLDEN_COOLDOWN_SECONDS, extract_tool_calls, get_answer, run_agent
from dataset import dataset
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from metrics import ALL_METRICS, SINGLE_TURN_METRICS, TRACE_METRICS
from model import eval_model
from models import model as agent_model
from prompts import agent_system_prompt
from results import run_setup, save_and_compare


def _capture_return(generator, box: dict):
    """Iterate a generator and keep its return value (evals_iterator returns its results)."""
    box["result"] = yield from generator


def parse_args(argv=None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=ALL_METRICS,
        metavar="METRIC",
        help=f"metrics to run (default: all). Choices: {', '.join(ALL_METRICS)}",
    )
    return parser.parse_args(argv).metrics


def main(argv=None):
    selected = parse_args(argv)
    trace_metrics = [TRACE_METRICS[name]() for name in selected if name in TRACE_METRICS]
    single_turn_metrics = {
        name: SINGLE_TURN_METRICS[name]() for name in selected if name in SINGLE_TURN_METRICS
    }
    # Argument Correctness is scored on test cases without zero-argument tool
    # calls (see extract_tool_calls), so it runs as its own evaluate() pass.
    argument_metric = single_turn_metrics.pop("argument_correctness", None)

    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    async_config = AsyncConfig(run_async=False)  # one golden at a time
    # ignore_errors: a judge failure (e.g. Gemini 503) marks that one metric as
    # errored instead of aborting the whole run and losing every score so far.
    error_config = ErrorConfig(ignore_errors=True, skip_on_missing_params=True)

    setup = run_setup(
        agent_model=agent_model.model_name,
        judge_model=eval_model.get_model_name(),
        system_prompt=agent_system_prompt,
        metrics=selected,
    )
    hyperparameters = {k: v for k, v in setup.items() if k != "metrics"}

    # Trace metrics need the callback trace, which evals_iterator collects per golden.
    # Without them, just loop over the goldens and skip the tracing overhead.
    trace_results: dict = {}
    if trace_metrics:
        goldens = _capture_return(
            dataset.evals_iterator(
                metrics=trace_metrics,
                async_config=async_config,
                error_config=error_config,
                cache_config=cache_config,
                display_config=display_config,
                hyperparameters=hyperparameters,
            ),
            trace_results,
        )
    else:
        goldens = iter(dataset.goldens)

    total = len(dataset.goldens)
    test_cases = []  # local lists, so repeated runs don't pile up in dataset.test_cases
    argument_test_cases = []
    agent_runs = {}  # what the agent did per golden, saved with the scores
    skipped = []

    for index, golden in enumerate(goldens, start=1):
        response = run_agent(golden.input, trace=bool(trace_metrics))
        if index < total:
            time.sleep(GOLDEN_COOLDOWN_SECONDS)

        if response is None:
            print(f"SKIPPED (agent never finished): {golden.input}")
            skipped.append(golden.input)
            continue

        answer = get_answer(response)
        if answer is None:
            print(f"SKIPPED (no reply to the customer): {golden.input}")
            skipped.append(golden.input)
            continue

        tools_called = extract_tool_calls(response)
        agent_runs[golden.input] = {
            "answer": answer,
            "expected_output": golden.expected_output,
            "tools_called": [{"name": t.name, "args": t.input_parameters} for t in tools_called],
            # Each tool call with its result, in the format the RefundOps app renders.
            "steps": [asdict(step) for step in steps_from_messages(response["messages"])],
            "expected_tools": [t.name for t in golden.expected_tools or []],
        }

        # Reuse the same agent run for every single-turn metric.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )
        argument_test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=answer,
                tools_called=extract_tool_calls(response, with_arguments_only=True),
            )
        )

    all_results = list(trace_results["result"].test_results) if trace_results.get("result") else []

    passes = [
        (list(single_turn_metrics.values()), test_cases),
        ([argument_metric] if argument_metric else [], argument_test_cases),
    ]
    for metrics, cases in passes:
        if not metrics:
            continue
        if not cases:
            print("No test cases to score: the agent did not finish on any golden.")
            break
        result = evaluate(
            test_cases=cases,
            metrics=metrics,
            async_config=async_config,
            error_config=error_config,
            cache_config=cache_config,
            display_config=display_config,
            hyperparameters=hyperparameters,
        )
        all_results.extend(result.test_results)

    for user_input in skipped:
        print(f"  skipped: {user_input}")
    save_and_compare(setup, all_results, agent_runs, skipped, total)


if __name__ == "__main__":
    main()
