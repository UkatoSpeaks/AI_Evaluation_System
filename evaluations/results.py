"""Save each evaluation run and compare it with the previous one.

Every run writes:
  results/runs/<timestamp>.json  full detail: per golden answer, tools, metric scores and reasons
  results/history.jsonl          one summary line per run (append-only), used for comparisons

Each run records what produced it (agent model, judge model, prompt hash,
git commit) so a change in scores can be traced back to a change in setup.
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HISTORY_FILE = RESULTS_DIR / "history.jsonl"


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_setup(agent_model: str, judge_model: str, system_prompt: str, metrics: list[str]) -> dict:
    """What produced this run. Also passed to deepeval as hyperparameters."""
    return {
        "agent_model": agent_model,
        "judge_model": judge_model,
        "prompt_sha": hashlib.sha256(system_prompt.encode()).hexdigest()[:8],
        "git_commit": _git_commit(),
        "metrics": metrics,
    }


def summarize_metrics(test_results) -> dict:
    """Per-metric average score and pass rate over the cases the judge actually scored."""
    by_metric: dict[str, list] = {}
    for result in test_results:
        for md in result.metrics_data or []:
            by_metric.setdefault(md.name, []).append(md)

    summary = {}
    for name, items in sorted(by_metric.items()):
        scored = [md for md in items if md.error is None and md.score is not None]
        summary[name] = {
            "scored": len(scored),
            "errored": len(items) - len(scored),
            "avg_score": round(sum(md.score for md in scored) / len(scored), 3) if scored else None,
            "pass_rate": round(sum(bool(md.success) for md in scored) / len(scored), 3)
            if scored
            else None,
        }
    return summary


def per_golden_detail(test_results, agent_runs: dict) -> list[dict]:
    """Merge every metric's verdict for a golden with what the agent actually did."""
    details = {
        user_input: {"input": user_input, **run, "metrics": {}}
        for user_input, run in agent_runs.items()
    }
    for result in test_results:
        entry = details.setdefault(result.input, {"input": result.input, "metrics": {}})
        for md in result.metrics_data or []:
            entry["metrics"][md.name] = {
                "score": md.score,
                "success": md.success,
                "reason": md.reason,
                "error": md.error,
            }
    return list(details.values())


def _previous_run() -> dict | None:
    if not HISTORY_FILE.exists():
        return None
    lines = [line for line in HISTORY_FILE.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def save_and_compare(setup: dict, test_results, agent_runs: dict, skipped: list[str], total: int):
    """Write this run to disk and print how it compares with the previous run."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    summary = {
        "timestamp": timestamp,
        **setup,
        "goldens_total": total,
        "goldens_scored": total - len(skipped),
        "metric_summary": summarize_metrics(test_results),
    }
    previous = _previous_run()

    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    run_file = RESULTS_DIR / "runs" / f"{timestamp}.json"
    suffix = 2
    while run_file.exists():  # never overwrite an earlier run
        run_file = RESULTS_DIR / "runs" / f"{timestamp}-{suffix}.json"
        suffix += 1
    run_file.write_text(
        json.dumps(
            {**summary, "skipped": skipped, "goldens": per_golden_detail(test_results, agent_runs)},
            indent=2,
            default=str,
        )
    )
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    _print_report(summary, previous)
    print(f"\nSaved: {run_file.relative_to(RESULTS_DIR.parent)} (history: {HISTORY_FILE.name})")


def _fmt(value) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def _print_report(current: dict, previous: dict | None) -> None:
    print("\n" + "=" * 78)
    print(
        f"Run summary  agent={current['agent_model']}  prompt={current['prompt_sha']}  "
        f"commit={current['git_commit']}"
    )
    print(f"Goldens scored: {current['goldens_scored']}/{current['goldens_total']}")
    if previous:
        changed = [
            f"{key}: {previous.get(key)} -> {current[key]}"
            for key in ("agent_model", "judge_model", "prompt_sha", "git_commit")
            if previous.get(key) != current[key]
        ]
        print(f"Compared with run {previous['timestamp']}")
        print("  changed since then: " + ("; ".join(changed) if changed else "nothing (same setup)"))
    print("-" * 78)
    print(f"{'Metric':<34}{'avg':>7}{'prev':>7}{'delta':>8}{'pass':>7}{'errored':>9}")
    prev_metrics = (previous or {}).get("metric_summary", {})
    for name, m in current["metric_summary"].items():
        prev_avg = prev_metrics.get(name, {}).get("avg_score")
        delta = (
            f"{m['avg_score'] - prev_avg:+.2f}"
            if m["avg_score"] is not None and prev_avg is not None
            else "  -"
        )
        errored = f"{m['errored']}" + (" !" if m["errored"] else "")
        print(
            f"{name:<34}{_fmt(m['avg_score']):>7}{_fmt(prev_avg):>7}{delta:>8}"
            f"{_fmt(m['pass_rate']):>7}{errored:>9}"
        )
    if any(m["errored"] for m in current["metric_summary"].values()):
        print("  ! errored = judge failures (e.g. rate limits), excluded from avg and pass rate")
    print("=" * 78)
