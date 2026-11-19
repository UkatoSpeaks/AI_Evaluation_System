import asyncio
import os
import re
import threading
import time

from deepeval.models import GeminiModel
from dotenv import load_dotenv
from google.genai import errors

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")

# Gemini free tier allows 15 requests/minute for this model. Stay just under it.
JUDGE_REQUESTS_PER_MINUTE = float(os.environ.get("JUDGE_REQUESTS_PER_MINUTE", "14"))
JUDGE_MAX_ATTEMPTS = 6
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_slot_lock = threading.Lock()
_next_slot = 0.0


def _reserve_slot() -> float:
    """Reserve the next request slot; returns how long to wait before using it."""
    global _next_slot
    with _slot_lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + 60.0 / JUDGE_REQUESTS_PER_MINUTE
        return start - now


def _retry_delay(error: errors.APIError, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if the error is not worth retrying."""
    if error.code not in RETRYABLE_STATUS_CODES:
        return None
    # Gemini 429s say exactly how long to wait: "Please retry in 53.67s".
    hint = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error))
    if hint:
        return float(hint.group(1)) + 1
    return min(60.0, 5.0 * 2 ** (attempt - 1))


class PacedGeminiModel(GeminiModel):
    """Gemini judge that stays under the free-tier rate limit and retries 429/5xx.

    deepeval 4.3.4's built-in Gemini retry policy never activates (it looks up
    `google.genai.gerrors`, which does not exist), so without this every rate
    limit or overload error turns straight into an errored metric.
    """

    def generate(self, *args, **kwargs):
        for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
            time.sleep(_reserve_slot())
            try:
                return super().generate(*args, **kwargs)
            except errors.APIError as e:
                delay = _retry_delay(e, attempt)
                if delay is None or attempt == JUDGE_MAX_ATTEMPTS:
                    raise
                print(f"Judge {e.code}, retrying in {delay:.0f}s ({attempt}/{JUDGE_MAX_ATTEMPTS})")
                time.sleep(delay)

    async def a_generate(self, *args, **kwargs):
        for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
            await asyncio.sleep(_reserve_slot())
            try:
                return await super().a_generate(*args, **kwargs)
            except errors.APIError as e:
                delay = _retry_delay(e, attempt)
                if delay is None or attempt == JUDGE_MAX_ATTEMPTS:
                    raise
                print(f"Judge {e.code}, retrying in {delay:.0f}s ({attempt}/{JUDGE_MAX_ATTEMPTS})")
                await asyncio.sleep(delay)


eval_model = PacedGeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)
