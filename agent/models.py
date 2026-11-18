from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    # Tested against openai/gpt-oss-120b, which often ended its turn with an
    # empty message after a tool result. qwen occasionally garbles a tool name,
    # which Groq rejects with a 400; the eval's run_agent retries those.
    model="qwen/qwen3.8-27b",
    model_provider="groq",
    temperature=0,
    timeout=60,  # a stuck Groq request fails instead of hanging the run
    max_retries=2,
)
