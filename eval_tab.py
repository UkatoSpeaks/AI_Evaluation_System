"""The "Eval Results" tab: two pinned eval runs, compared.

Reads evaluations/results/pinned/{baseline,current}.json, written by
evaluations/pin_run.py. Nothing here calls a model, so viewing the tab costs
no quota.
"""

import json
from pathlib import Path

import streamlit as st

from agent_trace import Step

PINNED_DIR = Path(__file__).resolve().parent / "evaluations" / "results" / "pinned"

# Column order for the cards and the table; deepeval's names as saved in the run files.
METRIC_ORDER = [
    "Task Completion",
    "Step Efficiency",
    "Plan Quality",
    "Plan Adherence",
    "Tool Correctness",
    "Argument Correctness",
    "Answer Correctness [GEval]",
]
CELL = {"pass": "✅", "fail": "❌", "error": "⚠️", "missing": "—"}


def short_metric(name: str) -> str:
    return name.replace(" [GEval]", "")


def load_pinned(pinned_dir: str) -> dict:
    """Read on every rerun (two small files) so re-pinning shows up without a restart."""
    runs = {}
    for slot in ("baseline", "current"):
        path = Path(pinned_dir) / f"{slot}.json"
        if path.is_file():
            runs[slot] = json.loads(path.read_text())
    return runs


def verdict_of(golden: dict | None, metric: str) -> str:
    """pass, fail, error (the judge failed) or missing (not run, or golden skipped)."""
    data = (golden or {}).get("metrics", {}).get(metric)
    if not data:
        return "missing"
    if data.get("error"):
        return "error"
    return "pass" if data.get("success") else "fail"


def metric_names(*runs) -> list[str]:
    seen = {m for run in runs if run for g in run["goldens"] for m in g.get("metrics", {})}
    ordered = [m for m in METRIC_ORDER if m in seen]
    return ordered + sorted(seen - set(ordered))


def counts(run: dict, metric: str) -> dict:
    tally = {"pass": 0, "fail": 0, "error": 0, "missing": 0}
    for golden in run["goldens"]:
        tally[verdict_of(golden, metric)] += 1
    tally["scored"] = tally["pass"] + tally["fail"]
    return tally


def by_input(run: dict | None) -> dict:
    return {g["input"]: g for g in run["goldens"]} if run else {}


def golden_inputs(current: dict, baseline: dict | None) -> list[str]:
    """Every golden in either run, in run order, skipped ones included."""
    order = []
    for run in (current, baseline):
        if not run:
            continue
        for text in [g["input"] for g in run["goldens"]] + list(run.get("skipped", [])):
            if text not in order:
                order.append(text)
    return order


def _short(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_header(current: dict, baseline: dict | None):
    def describe(run):
        return (
            f"**{run.get('label', 'Run')}** · {run.get('timestamp', '?')}  \n"
            f"agent `{run.get('agent_model', '?')}` · prompt `{run.get('prompt_sha', '?')}` · "
            f"commit `{run.get('git_commit', '?')}` · "
            f"{run.get('goldens_scored', '?')}/{run.get('goldens_total', '?')} goldens scored"
        )

    left, right = st.columns(2)
    if baseline:
        left.markdown(describe(baseline))
    right.markdown(describe(current))
    if baseline:
        changed = [
            f"{key.replace('_', ' ')}: `{baseline.get(key)}` → `{current.get(key)}`"
            for key in ("agent_model", "judge_model", "prompt_sha", "git_commit")
            if baseline.get(key) != current.get(key)
        ]
        st.caption("Changed between the runs: " + ("; ".join(changed) if changed else "nothing in the setup"))


def render_cards(metrics: list[str], current: dict, baseline: dict | None):
    per_row = 4
    for start in range(0, len(metrics), per_row):
        cols = st.columns(per_row)
        for col, metric in zip(cols, metrics[start : start + per_row]):
            now = counts(current, metric)
            value = f"{now['pass']}/{now['scored']}" if now["scored"] else "—"
            delta, note, help_lines = None, None, []
            if baseline:
                before = counts(baseline, metric)
                if before["scored"] and now["scored"]:
                    was = f"{before['pass']}/{before['scored']}"
                    points = round(100 * (now["pass"] / now["scored"] - before["pass"] / before["scored"]))
                    if points:
                        delta = f"{points:+d} pts · was {was}"
                    else:
                        note = "unchanged" if was == value else f"same rate · was {was}"
            if now["error"]:
                help_lines.append(f"{now['error']} not scored: the judge failed (e.g. rate limit).")
            if now["missing"]:
                help_lines.append(f"{now['missing']} not run or skipped.")
            col.metric(
                short_metric(metric),
                value,
                delta=delta,
                help=" ".join(help_lines) or "Goldens passed / goldens the judge scored.",
                border=True,
            )
            if note:
                col.caption(note)


def render_table(metrics: list[str], current: dict, baseline: dict | None):
    now, before = by_input(current), by_input(baseline)
    skipped = set(current.get("skipped", []))
    rows = []
    for text in golden_inputs(current, baseline):
        row = {"Golden": _short(text)}
        for metric in metrics:
            if text in skipped:
                row[short_metric(metric)] = "skipped"
                continue
            cell = CELL[verdict_of(now.get(text), metric)]
            if baseline and text in before:
                old = CELL[verdict_of(before[text], metric)]
                if old != cell and "—" not in (old, cell):
                    cell = f"{old}→{cell}"
            row[short_metric(metric)] = cell
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")
    legend = "✅ pass · ❌ fail · ⚠️ judge error, not scored · — not run"
    if baseline:
        legend += " · ❌→✅ changed since the baseline"
    st.caption(legend)


def _steps(golden: dict) -> list[Step]:
    return [Step(**s) for s in golden.get("steps", [])]


def render_failures(metrics: list[str], run: dict, step_html):
    failures = [
        (golden, metric, verdict_of(golden, metric))
        for golden in run["goldens"]
        for metric in metrics
        if verdict_of(golden, metric) in ("fail", "error")
    ]
    if not failures:
        st.success("No failures in this run.")
    for golden, metric, outcome in failures:
        data = golden["metrics"][metric]
        icon = CELL[outcome]
        score = data.get("score")
        score_text = f" · score {score:.2f}" if isinstance(score, (int, float)) else ""
        with st.expander(f"{icon} {short_metric(metric)}{score_text} · {_short(golden['input'], 80)}"):
            if outcome == "error":
                st.warning(f"The judge failed, so this was not scored: {data.get('error')}")
            else:
                st.markdown("**Judge's reason**")
                st.write(data.get("reason") or "_No reason recorded._")

            if "answer" in golden:
                left, right = st.columns(2)
                left.markdown("**Agent's answer**")
                left.write(golden["answer"])
                right.markdown("**Expected**")
                right.write(golden.get("expected_output") or "_none_")

            steps = _steps(golden)
            if steps:
                st.markdown("**What the agent did**")
                for step in steps:
                    st.markdown(step_html(step), unsafe_allow_html=True)
            elif golden.get("tools_called") is not None:
                called = " → ".join(t["name"] for t in golden["tools_called"]) or "no tools"
                st.markdown(f"**Tools called:** `{called}`")
            if golden.get("expected_tools"):
                st.markdown(f"**Expected tools:** `{' → '.join(golden['expected_tools'])}`")

    skipped = run.get("skipped", [])
    if skipped:
        st.caption("Skipped (the agent never finished, so nothing was scored): " + "; ".join(skipped))


def render(step_html, pinned_dir: Path = PINNED_DIR):
    runs = load_pinned(str(pinned_dir))
    current = runs.get("current") or runs.get("baseline")
    baseline = runs.get("baseline") if runs.get("current") else None

    if not current:
        st.info(
            "No eval runs pinned yet. Run the evals, then pin a run for each side of the comparison:\n\n"
            "```bash\n"
            "uv run python evaluations/eval_all.py\n"
            "uv run python evaluations/pin_run.py baseline --label \"Before the fix\"\n"
            "# ...make the fix, run the evals again, then:\n"
            "uv run python evaluations/pin_run.py current --label \"After the fix\"\n"
            "```\n"
            "Commit `evaluations/results/pinned/` so the deployed app shows them."
        )
        return

    st.caption(
        "Scored offline by a Gemini judge and loaded from saved runs, so this page never reruns the evals. "
        f"{len(current['goldens'])} goldens, so each golden is worth about "
        f"{100 / max(len(current['goldens']), 1):.0f} points of pass rate."
    )
    render_header(current, baseline)
    metrics = metric_names(current, baseline)

    st.markdown("##### Pass rate per metric")
    render_cards(metrics, current, baseline)

    st.markdown("##### Every golden × every metric")
    render_table(metrics, current, baseline)

    st.markdown("##### Why it failed")
    shown = current
    if baseline:
        pair = [current, baseline]
        choice = st.radio(
            "Run",
            options=[0, 1],
            format_func=lambda i: pair[i].get("label", ["After fix", "Baseline"][i]),
            horizontal=True,
            label_visibility="collapsed",
        )
        shown = pair[choice]
    render_failures(metrics, shown, step_html)
