import os
from dotenv import load_dotenv
from deepeval.models import GeminiModel

load_dotenv()

eval_model = GeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=os.environ["GOOGLE_API_KEY"],
)