from openai import OpenAI
from openai import (
    AuthenticationError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)
from logger import logger
from config import (
    GROQ_API_KEY,
    BASE_URL,
    MODEL_NAME
)

from prompt import SYSTEM_PROMPT

client = OpenAI(
    api_key = GROQ_API_KEY,
    base_url = BASE_URL
)

def extract_tasks(
     meeting_note: str
):
    try:
        response = client.chat.completions.create(
            model = MODEL_NAME,
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": meeting_note
                }
            ],
            temperature = 0,
            response_format = {
                "type": "json_object"
            }
        )
        
        return response.choices[0].message.content
    except AuthenticationError:
        logger.error("Invalid API Key")
        raise RuntimeError("Invalid API Key")
    
    except RateLimitError:
        logger.error("Rate limit exceeded")
        raise RuntimeError("Rate limit exceeded. Please try again later.")
    
    except APITimeoutError:
        logger.error("API request timed out")
        raise RuntimeError("API request timed out. Please try again later.")
    
    except APIConnectionError:
        logger.error("Failed to connect to the API")
        raise RuntimeError("Failed to connect to the API. Please check your internet connection.")
    
    except APIStatusError as e:
        logger.error(
            f"API returned an error: {e.status_code} - {e.message}"
        )
        
        raise RuntimeError(
            f"API Error {e.status_code}"
        )
    
    except Exception as e:
        logger.exception(e)
        
        raise RuntimeError(
            "Unexpected Error."
        )