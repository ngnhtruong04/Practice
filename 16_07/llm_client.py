import os
from openai import OpenAI
from dotenv import load_dotenv
from logger import logger

load_dotenv() 

# Khởi tạo client kết nối tới Groq API
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1" 
)
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile") 
def call_llm(prompt: str, user_input: str) -> str:
    """Gửi yêu cầu tới Groq API và nhận kết quả văn bản (JSON String)"""
    logger.info(f"Sending request to Groq model: {MODEL_NAME}") 
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": prompt}, 
                {"role": "user", "content": user_input} 
            ],
            temperature=0  # Giảm tính ngẫu nhiên, giúp output ổn định [cite: 573, 574]
        )
        logger.info("Received response successfully from Groq API") 
        return response.choices[0].message.content 
    except Exception as e:
        logger.error(f"API Call failed: {str(e)}") 
        raise e