from pydantic import ValidationError
from models import MeetingTasks
from logger import logger

def validate_meeting_tasks(data: dict) -> MeetingTasks:
    """Kiểm tra Dict có đúng Schema của MeetingTasks hay không"""
    try:
        validated_data = MeetingTasks.model_validate(data) 
        logger.info("Pydantic validation passed")
        return validated_data
    except ValidationError as e: 
        logger.error(f"Validation failed: {e}")
        raise ValueError(f"Validation failed: {e}")