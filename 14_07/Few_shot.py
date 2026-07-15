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
    Example 1

    Bike: Yamaha MT-15

    Category: Naked Bike

    Suitable For:
    City Riding

    ----------------

    Example 2

    Bike: Yamaha R15 V4

    Category: Sport Bike

    Suitable For:
    Sport Riding

    ----------------

    Bike: Yamaha 900GT+

    Category:
    
"""

result = ask_llm(prompt)

print(result)