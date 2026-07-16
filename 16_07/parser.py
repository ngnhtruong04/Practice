import json
from logger import logger

def parse_json(response: str) -> dict:
    """Parse chuỗi JSON sang Dict, bắt lỗi nếu LLM trả về JSON sai định dạng"""
    try:
        data = json.loads(response) 
        logger.info("JSON parsed successfully") 
        return data
    except json.JSONDecodeError as e: 
        logger.error(f"Invalid JSON format received: {e}")
        raise ValueError(f"Invalid JSON: {e}") 