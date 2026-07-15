from openai import OpenAI

from logger import logger

from config import (
    BASE_URL,
    GROQ_API_KEY,
    MODEL_NAME,
)

from prompt import SYSTEM_PROMPT

client = OpenAI(
    api_key = GROQ_API_KEY,
    base_url = BASE_URL
)

def generate_answer(
    user_prompt: str, 
    temperature: float = 0.2,
    top_p: float = 1.0
) -> str:
    logger.info(
        "Sending request to Groq"
        
    )
    
    
    response = client.chat.completions.create(
        model = MODEL_NAME, 
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role":"user",
                "content":user_prompt,
            },
        ],
        temperature = temperature, 
        top_p = top_p,
    )
    logger.info(
        "Response received successfully"    )
    
    return response.choices[0].message.content