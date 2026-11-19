"""RefundOps: chat with the refund agent and watch how it decides.

Run locally:   uv run streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file `streamlit_app.py`,
               with GROQ_API_KEY set in the app's secrets.
"""

import html
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Streamlit Cloud exposes secrets through st.secrets; the agent reads env vars.
# Copy the key across before the agent module (and its Groq client) is imported.
try:
    if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:  # no secrets.toml locally: .env is used instead
    pass

# The agent code lives in ./agent and uses flat imports (from tools import ...).
sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))

from groq import APIStatusError  # noqa: E402
from langchain.messages import AIMessage, HumanMessage  # noqa: E402

import eval_tab  # noqa: E402
from agent_trace import (  # noqa: E402
    TraceBuilder,
    policy_checks,
    reply_conflict,
    steps_from_messages,
    verdict,
)

MAX_TURNS_PER_SESSION = 15  # the app is public and every turn spends the Groq quota
AGENT_TRIES = 4

# One click sends a golden input from evaluations/dataset.py.
SCENARIOS = {
    "Eligible": "I am user123, I would like a refund for the book and the pen",
    "Too old": "I am user126, I would like a refund for the water_bottle.",
    "Partly eligible": "I am user126, I would like a refund for the water_bottle and the backpack.",
    "Unknown user": "I am user999, I would like a refund for the book.",
    "Never bought": "I am user123, I would like a refund for the sneakers.",
    "Info only": "I am user123, I do not want a refund, I just want to know what I purchased.",
}

st.set_page_config(page_title="RefundOps", page_icon="🎫", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 6rem; }
      .ro-title { font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
      .ro-title span { color: #2dd4bf; }
      .ro-sub { color: #8b98a5; margin: 0.15rem 0 1rem; }
      .ro-badge {
        position: fixed; right: 1rem; bottom: 0.6rem; z-index: 1000;
        font-family: monospace; font-size: 0.75rem; color: #8b98a5;
        background: #131c26; border: 1px solid #1f2b38;
        border-radius: 999px; padding: 0.25rem 0.75rem;
      }

      /* trace steps */
      .ro-step { border: 1px solid #1f2b38; border-radius: 0.6rem; padding: 0.55rem 0.7rem;
                 margin: 0 0 0.5rem; background: #0f1720; }
      .ro-step.bad { border-color: #7f1d1d; }
      .ro-head { display: flex; align-items: center; gap: 0.5rem; font-family: monospace; font-size: 0.85rem; }
      .ro-num { flex: none; width: 1.35rem; height: 1.35rem; border-radius: 999px; display: grid;
                place-items: center; background: #134e4a; color: #5eead4; font-size: 0.72rem; font-weight: 700; }
      .ro-name { color: #e6edf3; font-weight: 600; overflow-wrap: anywhere; }
      .ro-icon { margin-left: auto; }
      .ro-args { font-family: monospace; font-size: 0.78rem; color: #8b98a5; margin: 0.25rem 0 0 1.85rem;
                 overflow-wrap: anywhere; }
      .ro-result { font-family: monospace; font-size: 0.76rem; color: #c9d4df; background: #0b1117;
                   border-radius: 0.4rem; padding: 0.4rem 0.55rem; margin: 0.35rem 0 0 1.85rem;
                   white-space: pre-wrap; overflow-wrap: anywhere; max-height: 9rem; overflow-y: auto; }
      .ro-thought { font-size: 0.78rem; color: #8b98a5; font-style: italic; margin: 0 0 0.3rem; }
      .ro-retry { font-size: 0.8rem; color: #fbbf24; margin: 0.2rem 0 0.5rem; }

      /* policy checks */
      .ro-checks { margin-top: 0.4rem; }
      .ro-check { display: flex; gap: 0.5rem; font-size: 0.86rem; padding: 0.22rem 0; }
      .ro-check small { color: #8b98a5; }

      /* verdict card */
      .ro-verdict { border-radius: 0.6rem; padding: 0.6rem 0.8rem; margin-top: 0.6rem; border: 1px solid; }
      .ro-verdict b { font-size: 0.98rem; }
      .ro-verdict ul { margin: 0.3rem 0 0; padding-left: 1.1rem; font-size: 0.82rem; }
      .ro-verdict.issued { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.45); color: #86efac; }
      .ro-verdict.denied { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.45); color: #fca5a5; }
      .ro-verdict.warn   { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.45); color: #fcd34d; }
      .ro-verdict.info   { background: rgba(148,163,184,0.08); border-color: rgba(148,163,184,0.35); color: #cbd5e1; }
      .ro-conflict { font-size: 0.82rem; color: #fcd34d; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the agent...")
def load_agent():
    """Build the agent once per server process; every session shares it.

    Sessions stay separate through their thread_id in the shared checkpointer.
    """
    from agent import agent, final_reply
    from data import users_data
    from tools import TODAY

    return agent, final_reply, users_data, TODAY


agent, final_reply, users_data, store_today = load_agent()


def init_session():
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("turns", [])  # [{"user", "reply", "messages", "error", "seconds"}]
    st.session_state.setdefault("pending", None)


def new_chat():
    st.session_state.thread_id = str(uuid4())
    st.session_state.turns = []
    st.session_state.pending = None
    st.session_state.reseed = False


def history_as_messages() -> list:
    """Earlier turns as plain text, to rebuild a thread after a failed call."""
    messages = []
    for turn in st.session_state.turns:
        if turn["reply"]:
            messages += [HumanMessage(content=turn["user"]), AIMessage(content=turn["reply"])]
    return messages


def _groq_error(exc: APIStatusError) -> str:
    if exc.status_code == 429:
        return "Groq rate limit (429)"
    if exc.status_code == 400:
        return "Groq rejected a malformed tool call (400)"
    return f"Groq error ({exc.status_code})"


def _retry_wait(exc: APIStatusError, attempt: int) -> float:
    """Seconds to wait before retrying: Groq's retry-after on a 429, capped at 30s."""
    if exc.status_code != 429:
        return 1
    try:
        return min(float(exc.response.headers.get("retry-after", "")), 30)
    except ValueError:
        return 5 * attempt


def _quota_wait(exc: APIStatusError) -> str | None:
    """'4 min' when a 429 asks for a wait longer than 30s (the daily token cap), else None."""
    if exc.status_code != 429:
        return None
    try:
        seconds = float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None
    if seconds <= 30:
        return None
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def run_turn(user_input: str, on_event) -> dict:
    """Stream one turn of the agent, reporting each tool step through on_event.

    on_event receives ("calling", [names]), ("step", Step) or ("retry", reason).

    When Groq fails mid-turn (Qwen sometimes garbles a tool name, e.g.
    "issue_refund ticket", which Groq rejects with a 400), the thread is
    resumed from its last checkpoint: only the failed model call is retried
    and the tool steps already done are kept. If the agent instead ends its
    turn without a reply, the turn restarts on a fresh thread seeded with the
    conversation so far as plain text.
    """
    started = time.monotonic()
    earlier = history_as_messages() if st.session_state.pop("reseed", False) else []
    graph_input = {"messages": earlier + [HumanMessage(content=user_input)]}
    builder = TraceBuilder()
    turn_messages = []  # only this turn's messages: the trace is built from these
    error = None

    for attempt in range(1, AGENT_TRIES + 1):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            for chunk in agent.stream(graph_input, config=config, stream_mode="updates"):
                for update in chunk.values():
                    new = update.get("messages", []) if isinstance(update, dict) else []
                    for message in new if isinstance(new, list) else [new]:
                        turn_messages.append(message)
                        for event in builder.feed(message):
                            on_event(event)
        except APIStatusError as exc:  # 429 rate limit, 400 garbled tool name, 5xx
            error = _groq_error(exc)
            long_wait = _quota_wait(exc)
            if long_wait:  # daily token quota used up: retrying now cannot succeed
                error = f"the demo's daily Groq quota is used up, please try again in about {long_wait}"
                break
            if attempt < AGENT_TRIES:
                on_event(("retry", f"{error}. Retrying that step (attempt {attempt + 1} of {AGENT_TRIES})."))
                time.sleep(_retry_wait(exc, attempt))
            graph_input = None  # resume from the last checkpoint
            continue

        reply = final_reply({"messages": turn_messages}) if turn_messages else None
        if reply:
            return {
                "user": user_input,
                "reply": reply,
                "messages": turn_messages,
                "error": None,
                "seconds": time.monotonic() - started,
            }
        error = "the agent ended its turn without a reply"
        if attempt < AGENT_TRIES:
            on_event(("retry", f"{error}. Starting the turn again (attempt {attempt + 1} of {AGENT_TRIES})."))
        st.session_state.thread_id = str(uuid4())
        graph_input = {"messages": history_as_messages() + [HumanMessage(content=user_input)]}
        builder, turn_messages = TraceBuilder(), []

    # Leave the failed thread behind (it may end in a half-finished tool call);
    # the next turn seeds the new thread with the conversation so far.
    st.session_state.thread_id = str(uuid4())
    st.session_state.reseed = True
    return {"user": user_input, "reply": None, "messages": [], "error": error, "seconds": time.monotonic() - started}


# ---------------------------------------------------------------- rendering helpers

def _short_json(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=1, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def step_html(step) -> str:
    args = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in step.args.items())
    if not step.ok:
        icon = "❌"
    elif step.name == "issue_refund_ticket":
        icon = "🎫"
    else:
        icon = "✅"
    thought = (
        f'<div class="ro-thought">💭 {html.escape(_short_json(step.thought, 240))}</div>' if step.thought else ""
    )
    return (
        f'<div class="ro-step{"" if step.ok else " bad"}">{thought}'
        f'<div class="ro-head"><span class="ro-num">{step.index}</span>'
        f'<span class="ro-name">{html.escape(step.name)}</span><span class="ro-icon">{icon}</span></div>'
        f'<div class="ro-args">({html.escape(_short_json(args, 300))})</div>'
        f'<div class="ro-result">→ {html.escape(_short_json(step.result, 600))}</div></div>'
    )


CHECK_ICONS = {True: "✅", False: "❌", None: "🚫"}  # None: policy blocked an item, as it should


def checks_html(checks) -> str:
    rows = "".join(
        f'<div class="ro-check"><span>{CHECK_ICONS[c["ok"]]}</span>'
        f'<span>{html.escape(c["label"])}'
        f'{f" <small>· {html.escape(c["detail"])}</small>" if c["detail"] else ""}</span></div>'
        for c in checks
    )
    return f'<div class="ro-checks">{rows}</div>'


def verdict_html(result, conflict: str | None) -> str:
    if not result:
        return f'<div class="ro-conflict">⚠ {html.escape(conflict)}</div>' if conflict else ""
    details = "".join(f"<li>{html.escape(d)}</li>" for d in result["details"])
    conflict_html = f'<div class="ro-conflict">⚠ Reply disagrees with tools: {html.escape(conflict)}</div>' if conflict else ""
    return (
        f'<div class="ro-verdict {result["kind"]}"><b>{html.escape(result["title"])}</b>'
        f'{f"<ul>{details}</ul>" if details else ""}{conflict_html}</div>'
    )


def render_reply(turn):
    """The assistant bubble's contents: the reply plus a verdict card built from the tools."""
    if not turn["reply"]:
        st.error(f"No reply: {turn['error']}.")
        return
    st.write(turn["reply"])
    steps = steps_from_messages(turn["messages"])
    card = verdict_html(verdict(steps), reply_conflict(turn["reply"], steps))
    if card:
        st.markdown(card, unsafe_allow_html=True)


def render_trace(turn):
    steps = steps_from_messages(turn["messages"])
    with st.container(border=True):
        n = len(steps)
        st.caption(f"{n} tool call{'s' if n != 1 else ''} · {turn.get('seconds', 0):.1f}s")
        if not steps:
            st.caption("No tools were called in this turn.")
        for step in steps:
            st.markdown(step_html(step), unsafe_allow_html=True)
    checks = policy_checks(steps)
    if checks:
        st.markdown("**Policy check**")
        st.markdown(checks_html(checks), unsafe_allow_html=True)
        st.caption("✅ passed · 🚫 blocked by policy, as it should be · ❌ agent mistake")


def customers_table() -> list[dict]:
    return [
        {"user": user, "item": item, "price": f"${p['price']}", "date": p["date"]}
        for user, purchases in users_data.items()
        for item, p in purchases.items()
    ]


init_session()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Demo customers")
    st.caption(f"The store's date is frozen at **{store_today.isoformat()}**, so the 30-day window is reproducible.")
    st.dataframe(customers_table(), hide_index=True, height=280, width="stretch")

    st.markdown("### Try a scenario")
    for label, text in SCENARIOS.items():
        if st.button(label, help=text, width="stretch"):
            st.session_state.pending = text

    st.divider()
    st.button("New chat", on_click=new_chat, type="primary", width="stretch")
    turns_left = MAX_TURNS_PER_SESSION - len(st.session_state.turns)
    st.caption(f"{turns_left} messages left in this session.")

# ---------------------------------------------------------------- header
st.markdown('<p class="ro-title">Refund<span>Ops</span></p>', unsafe_allow_html=True)
st.markdown(
    '<p class="ro-sub">A refund agent you can watch think, and the evals that keep it honest.</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="ro-badge">Agent: Qwen on Groq · Judge: Gemini</div>', unsafe_allow_html=True)

agent_tab, results_tab = st.tabs(["Agent", "Eval Results"])

with agent_tab:
    chat_col, trace_col = st.columns([3, 2], gap="large")

    typed = st.chat_input(
        "Try: I'm user126, refund the backpack",
        disabled=turns_left <= 0,
    )
    user_input = typed or st.session_state.pending
    st.session_state.pending = None
    running = bool(user_input) and turns_left > 0

    with chat_col:
        chat_box = st.container(height=560, border=False)
        with chat_box:
            if not st.session_state.turns and not running:
                st.info("Pick a scenario on the left, or type a message. Usernames and items are in the table.")
            for turn in st.session_state.turns:
                st.chat_message("user").write(turn["user"])
                with st.chat_message("assistant"):
                    render_reply(turn)
            if running:
                st.chat_message("user").write(user_input)
                reply_slot = st.chat_message("assistant").empty()
                reply_slot.caption("Thinking…")

    with trace_col:
        st.markdown("#### Agent trace")
        if running:
            status = st.status("Agent working...", expanded=True)
        elif st.session_state.turns:
            turns = st.session_state.turns
            shown = st.selectbox(
                "Turn",
                options=range(len(turns)),
                index=len(turns) - 1,
                format_func=lambda i: f"{i + 1}. {turns[i]['user'][:60]}",
                label_visibility="collapsed",
            )
            render_trace(turns[shown])
        else:
            st.caption("Each tool call the agent makes shows up here, with its arguments and result.")

    if running:
        def show_event(event):
            kind, payload = event
            if kind == "calling":
                status.update(label=f"Calling {', '.join(payload)}…")
            elif kind == "step":
                status.markdown(step_html(payload), unsafe_allow_html=True)
            elif kind == "retry":
                status.markdown(f'<div class="ro-retry">↻ {html.escape(payload)}</div>', unsafe_allow_html=True)

        turn = run_turn(user_input, show_event)
        if turn["reply"]:
            status.update(label="Done", state="complete")
        else:
            status.update(label=f"Failed: {turn['error']}", state="error")
        with reply_slot.container():
            render_reply(turn)
        st.session_state.turns.append(turn)
        time.sleep(0.8)  # let the finished trace sit for a moment before the redraw
        st.rerun()  # redraw from state: the full trace with policy checks, and the sidebar counter

with results_tab:
    eval_tab.render(step_html)
