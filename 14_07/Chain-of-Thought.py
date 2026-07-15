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
    A rider travels:

    40 km in city
    80 km on highway

    The motorcycle fuel consumption is:

    45 km/l

    Fuel tank:

    10 liters

    Think step by step.

    Can the rider finish the trip without refueling?
"""

result = ask_llm(prompt)

print(result)