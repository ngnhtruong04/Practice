import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)

MODEL_NAME = os.getenv("MODEL_NAME")


def call_llm(user_input:str):
    response = client.chat.completions.create(
        model = MODEL_NAME,
        
        messages =[
            {
                "role":"system",
                "content":__import__("prompt").SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":user_input
            }
        ],
        temperature = 0
    )
    
    return response.choices[0].message.content