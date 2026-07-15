from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)

def generate_answer(
    system_prompt,
    user_prompt,
    temperature,
    top_p
):
    response = client.chat.completions.create(
        
        model = "llama-3.3-70b-versatile",
        
        messages = [
            {
                "role":"system",
                "content": system_prompt,
            },
            {
                "role":"user",
                "content": user_prompt,
            },
        ],
        
        temperature = temperature,
        
        top_p = top_p,
        
    )
    
    return response.choices[0].message.content