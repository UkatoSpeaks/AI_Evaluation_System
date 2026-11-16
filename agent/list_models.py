from dotenv import load_dotenv
from groq import Groq

load_dotenv()

for m in sorted(Groq().models.list().data, key=lambda m: m.id):
    print(m.id)