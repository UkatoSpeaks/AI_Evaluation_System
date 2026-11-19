"""Turn the agent's messages into trace steps, policy checks and a verdict.

Everything here is read from the tool results, not from the model's reply, so
the verdict shows what the agent actually did. If the reply says something
else, reply_conflict() flags it.
"""

import ast
import json
import re
from dataclasses import dataclass

REFUND_WINDOW_DAYS = 30


@dataclass
class Step:
    index: int  # 1-based position in the turn
    name: str
    args: dict
    result: object  # parsed dict/list when the tool returned JSON, else the raw string
    ok: bool
    thought: str = ""  # text the model wrote alongside the tool call, if any


def parse_result(content):
    """Tool results arrive as strings; dict results are JSON-encoded."""
    if not isinstance(content, str):
        return content
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (ValueError, SyntaxError, TypeError):
            pass
    return content


def _succeeded(result, status: str | None) -> bool:
    if status == "error":
        return False
    if isinstance(result, dict) and "error" in result:
        return False
    if isinstance(result, str) and result.lower().startswith("error"):
        return False
    return True


class TraceBuilder:
    """Feed messages in order (live, as they stream) and get trace events back.

    feed() returns a list of events:
      ("calling", [tool names])  the model asked for these tools
      ("step", Step)             a tool finished
    """

    def __init__(self):
        self.steps: list[Step] = []
        self._pending: dict[str, tuple[str, dict, str]] = {}

    def feed(self, message) -> list[tuple]:
        events = []
        tool_calls = getattr(message, "tool_calls", None)
        if message.type == "ai" and tool_calls:
            thought = (message.text or "").strip()
            for call in tool_calls:
                self._pending[call["id"]] = (call["name"], call.get("args") or {}, thought)
                thought = ""  # attach the model's text to the first call only
            events.append(("calling", [call["name"] for call in tool_calls]))
        elif message.type == "tool":
            name, args, thought = self._pending.pop(
                message.tool_call_id, (getattr(message, "name", None) or "unknown_tool", {}, "")
            )
            result = parse_result(message.content)
            step = Step(
                index=len(self.steps) + 1,
                name=name,
                args=args,
                result=result,
                ok=_succeeded(result, getattr(message, "status", None)),
                thought=thought,
            )
            self.steps.append(step)
            events.append(("step", step))
        return events


def steps_from_messages(messages) -> list[Step]:
    builder = TraceBuilder()
    for message in messages:
        builder.feed(message)
    return builder.steps


def money(amount) -> str:
    amount = float(amount)
    return f"${amount:,.0f}" if amount.is_integer() else f"${amount:,.2f}"


def _calls(steps, name) -> list[Step]:
    return [s for s in steps if s.name == name]


def _last_calculation(steps) -> dict | None:
    calcs = [s.result for s in _calls(steps, "calculate_right_refund") if isinstance(s.result, dict)]
    return calcs[-1] if calcs else None


def _tickets(steps) -> list[Step]:
    return [s for s in _calls(steps, "issue_refund_ticket") if s.ok]


def _ticket_total(steps) -> float:
    return sum(float(s.args.get("refund_amount", 0) or 0) for s in _tickets(steps))


def _short_reason(reason: str) -> str:
    if "outside" in reason:
        return f"outside {REFUND_WINDOW_DAYS} days"
    if "future" in reason:
        return "purchase date in the future"
    return reason


def _not_refunded(calc: dict) -> list[str]:
    """One line per requested item that was not refunded, with why."""
    lines = [f"{item}: {reason}" for item, reason in (calc.get("ineligible_items") or {}).items()]
    lines += [f"{item}: not purchased by this customer" for item in calc.get("items_not_purchased") or []]
    return lines


def verdict(steps: list[Step]) -> dict | None:
    """The outcome of the turn as {kind, title, details}, or None if no refund work happened.

    kind is one of: issued, denied, warn, info.
    """
    if not steps:
        return None
    calc = _last_calculation(steps)
    history_errors = [s for s in _calls(steps, "get_user_purchase_history") if not s.ok]

    if _tickets(steps):
        details = _not_refunded(calc) if calc else []
        return {
            "kind": "issued",
            "title": f"Refund issued, {money(_ticket_total(steps))}",
            "details": [f"Not refunded, {line}" for line in details],
        }
    if calc and "error" in calc:
        return {"kind": "denied", "title": "Refund denied: unknown customer", "details": [calc["error"]]}
    if calc and float(calc.get("refund_amount", 0) or 0) > 0:
        return {
            "kind": "warn",
            "title": f"{money(calc['refund_amount'])} is refundable, but no ticket was issued",
            "details": _not_refunded(calc),
        }
    if calc:
        ineligible = calc.get("ineligible_items") or {}
        if ineligible:
            headline = _short_reason(next(iter(ineligible.values())))
        elif calc.get("items_not_purchased"):
            headline = "item not purchased"
        else:
            headline = "nothing eligible to refund"
        return {"kind": "denied", "title": f"Refund denied: {headline}", "details": _not_refunded(calc)}
    if history_errors:
        result = history_errors[0].result
        detail = result.get("error") if isinstance(result, dict) else str(result)
        return {"kind": "denied", "title": "Refund denied: unknown customer", "details": [detail]}
    if _calls(steps, "get_user_purchase_history"):
        return {"kind": "info", "title": "No refund requested", "details": ["Looked up purchases only."]}
    return None


def policy_checks(steps: list[Step]) -> list[dict]:
    """The checks that apply to this turn, as {label, ok, detail}.

    ok is True (passed), False (the agent got it wrong) or None (the policy
    blocked an item, which is the correct outcome).
    """
    checks = []
    history = _calls(steps, "get_user_purchase_history")
    calc_steps = _calls(steps, "calculate_right_refund")
    calc = _last_calculation(steps)
    tickets = _tickets(steps)

    if history:
        found = all(s.ok for s in history)
        checks.append({"label": "Customer found", "ok": found, "detail": "" if found else "unknown username"})

    if calc_steps:
        first_calc = calc_steps[0].index
        date_steps = [s.index for s in _calls(steps, "check_current_date")]
        dated = any(i < first_calc for i in date_steps)
        checks.append({
            "label": "Date checked before calculating",
            "ok": dated,
            "detail": "" if dated else "check_current_date was not called first",
        })

    if calc and "error" not in calc:
        # ok=None: the policy blocked an item. That is the policy working, not
        # the agent failing; whether the agent respected it shows in the ticket check.
        missing = calc.get("items_not_purchased") or []
        checks.append({
            "label": "Items purchased",
            "ok": None if missing else True,
            "detail": f"blocked, never bought: {', '.join(missing)}" if missing else "",
        })
        ineligible = calc.get("ineligible_items") or {}
        checks.append({
            "label": f"{REFUND_WINDOW_DAYS}-day window",
            "ok": None if ineligible else True,
            "detail": "blocked " + "; ".join(f"{item} ({_short_reason(r)})" for item, r in ineligible.items())
            if ineligible
            else "",
        })

    if tickets or calc_steps:
        owed = float(calc.get("refund_amount", 0) or 0) if calc and "error" not in calc else 0.0
        if tickets and not calc_steps:
            ok, detail = False, "ticket issued without calculating the refund"
        elif tickets:
            paid = _ticket_total(steps)
            ok = abs(paid - owed) < 0.01
            detail = "" if ok else f"ticket {money(paid)} but policy allows {money(owed)}"
        else:
            ok = owed == 0
            detail = "correctly no ticket" if ok else f"{money(owed)} owed but no ticket"
        checks.append({"label": "Ticket matches calculation", "ok": ok, "detail": detail})

    return checks


_NEGATION = re.compile(r"\b(not|no|unable|cannot|can't|couldn't|won't|isn't|wasn't)\b", re.I)
_ISSUED = re.compile(r"\b(has been|have|was|been|I've|I have)\s+(issued|processed|approved)\b", re.I)


def _mentions_amount(reply: str, amount: float) -> bool:
    number = f"{amount:g}"
    if "." in number:
        pattern = rf"(?<![\d.]){re.escape(number)}0*(?!\d)"
    else:
        pattern = rf"(?<![\d.]){number}(\.0+)?(?!\d)"
    return re.search(pattern, reply.replace(",", "")) is not None


def reply_conflict(reply: str | None, steps: list[Step]) -> str | None:
    """A warning when the reply to the customer disagrees with the tool results."""
    if not reply:
        return None
    if _tickets(steps):
        total = _ticket_total(steps)
        if not _mentions_amount(reply, total):
            return f"The reply doesn't state the ticketed amount ({money(total)})."
        return None
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        if _ISSUED.search(sentence) and re.search(r"refund|ticket", sentence, re.I) and not _NEGATION.search(sentence):
            return "The reply says a refund was issued, but no ticket was created."
    return None
