from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

def ask_llm(prompt):
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [ {
            "role":"user",
            "content": prompt
        }]
    )
    return response.choices[0].message.content

prompt = """
    You are a motorcycle expert.

    Short Review the Yamaha MT-15.

    Include:
    - Engine
    - Design
    - Riding Position
    - Pros
    - Cons

    Return in Markdown.
"""

result = ask_llm(prompt)

print(result)