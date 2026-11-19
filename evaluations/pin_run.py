"""Pin a saved eval run as the baseline or current run shown in the RefundOps app.

The app compares two pinned runs instead of rerunning the evals, which would
spend the Groq and Gemini quotas. Run files under results/runs/ are not
committed; pinned copies under results/pinned/ are, so the deployed app has them.

    python evaluations/pin_run.py baseline results/runs/2026-11-19T10-00-00.json --label "Before the date-check fix"
    python evaluations/pin_run.py current                # newest run, default label
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PINNED_DIR = RESULTS_DIR / "pinned"
SLOTS = {"baseline": "Baseline", "current": "After fix"}


def newest_run() -> Path:
    runs = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"No runs in {RUNS_DIR}. Run evaluations/eval_all.py first.")
    return runs[-1]


def resolve(run: str | None) -> Path:
    if run is None:
        return newest_run()
    path = Path(run)
    for candidate in (path, RESULTS_DIR / path, RUNS_DIR / path, RUNS_DIR / f"{run}.json"):
        if candidate.is_file():
            return candidate
    sys.exit(f"Run file not found: {run}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slot", choices=SLOTS, help="which side of the comparison to pin")
    parser.add_argument("run", nargs="?", help="run file or timestamp (default: the newest run)")
    parser.add_argument("--label", help="name shown in the app (default: Baseline / After fix)")
    args = parser.parse_args(argv)

    source = resolve(args.run)
    run = json.loads(source.read_text())
    if "goldens" not in run:
        sys.exit(f"{source} has no per-golden results; is it a file written by eval_all.py?")
    run["label"] = args.label or SLOTS[args.slot]
    run["source_file"] = source.name

    PINNED_DIR.mkdir(parents=True, exist_ok=True)
    target = PINNED_DIR / f"{args.slot}.json"
    target.write_text(json.dumps(run, indent=2, default=str) + "\n")

    print(f"Pinned {source.name} as {args.slot} ({run['label']!r}) -> {target.relative_to(RESULTS_DIR.parent)}")
    print(
        f"  {run.get('goldens_scored')}/{run.get('goldens_total')} goldens scored, "
        f"metrics: {', '.join(run.get('metrics', []))}"
    )
    print("  Commit this file so the deployed app shows it.")


if __name__ == "__main__":
    main()
