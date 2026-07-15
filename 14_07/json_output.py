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
    Generate motorcycle information.

    Return JSON only.

    Bike:
    Yamaha MT-15
    
    {
    "name":"",
    "engine":"",
    "horsepower":"",
    "torque":"",
    "fuel_tank":"",
    "weight":""
    }
"""

result = ask_llm(prompt)

print(result)