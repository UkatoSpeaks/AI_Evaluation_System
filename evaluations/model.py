import os

from deepeval.models import GeminiModel
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")

eval_model = GeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)
