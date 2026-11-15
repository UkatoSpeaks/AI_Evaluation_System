from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from groq import Groq, AsyncGroq

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        self.client = Groq()
        self.aclient = AsyncGroq()

    def load_model(self):
        return self.client

    def get_model_name(self):
        return f"groq/{self.model_name}"

    def _params(self, prompt, schema):
        params = dict(
            model=self.model_name,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        if schema:
            params["response_format"] = {"type": "json_object"}
        return params

    def generate(self, prompt: str, schema=None):
        out = self.client.chat.completions.create(**self._params(prompt, schema))
        text = out.choices[0].message.content
        return schema.model_validate_json(text) if schema else text

    async def a_generate(self, prompt: str, schema=None):
        out = await self.aclient.chat.completions.create(**self._params(prompt, schema))
        text = out.choices[0].message.content
        return schema.model_validate_json(text) if schema else text


eval_model = GroqJudge()