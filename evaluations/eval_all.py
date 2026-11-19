"""Run the refund agent over the dataset and score it.

Runs all 7 metrics by default; pass --metrics to run a subset, e.g.
    python evaluations/eval_all.py --metrics tool_correctness answer_correctness
"""

import argparse
import time

import common  # noqa: F401  (must come first: loads .env and the judge retry policy)
from common import GOLDEN_COOLDOWN_SECONDS, extract_tool_calls, get_answer, run_agent
from dataset import dataset
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from metrics import ALL_METRICS, SINGLE_TURN_METRICS, TRACE_METRICS


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

    # Trace metrics need the callback trace, which evals_iterator collects per golden.
    # Without them, just loop over the goldens and skip the tracing overhead.
    if trace_metrics:
        goldens = dataset.evals_iterator(
            metrics=trace_metrics,
            async_config=async_config,
            error_config=error_config,
            cache_config=cache_config,
            display_config=display_config,
        )
    else:
        goldens = iter(dataset.goldens)

    total = len(dataset.goldens)
    test_cases = []  # local lists, so repeated runs don't pile up in dataset.test_cases
    argument_test_cases = []
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

        # Reuse the same agent run for every single-turn metric.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=answer,
                expected_output=golden.expected_output,
                tools_called=extract_tool_calls(response),
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
        evaluate(
            test_cases=cases,
            metrics=metrics,
            async_config=async_config,
            error_config=error_config,
            cache_config=cache_config,
            display_config=display_config,
        )

    print(
        f"\nRan {len(selected)} metric(s) ({', '.join(selected)}) on "
        f"{total - len(skipped)}/{total} goldens."
    )
    for user_input in skipped:
        print(f"  skipped: {user_input}")


if __name__ == "__main__":
    main()
