from prompt import SYSTEM_PROMPT 

from llm_client import call_llm 

from parser import parse_json 
from validator import validate_meeting_tasks
from logger import logger

def main():
    # 1. Dữ liệu đầu vào (Biên bản cuộc họp) [cite: 907, 908, 948]
    meeting_note = """
    Team discussed the new payment system today. 
    John will update the backend API before Friday with high priority.
    Anna will prepare the UI design next Monday, this is medium priority.
    The manager wants high priority for security testing, deadline not specified yet, assigned to Kevin.
    """
    
    print("=" * 60)
    print("MEETING NOTE INPUT:")
    print(meeting_note.strip())
    print("=" * 60)

    try:
        # 2. Gọi LLM API 
        raw_response = call_llm(SYSTEM_PROMPT, meeting_note) 
        logger.debug(f"Raw LLM Output:\n{raw_response}")
        
        # 3. Parse JSON 
        parsed_data = parse_json(raw_response) 
        
        # 4. Validate dữ liệu bằng Pydantic 
        meeting_tasks = validate_meeting_tasks(parsed_data)
        
        # 5. Business Logic / Hiển thị kết quả 
        print("\n" + "=" * 60)
        print("EXTRACTION RESULT (VALIDATED OBJECTS)")
        print("=" * 60)
        for i, task in enumerate(meeting_tasks.tasks, 1): 
            print(f"Task #{i}: {task.task_name}") 
            print(f"  ├── Assignee : {task.assignee}") 
            print(f"  ├── Priority : {task.priority}") 
            print(f"  └── Deadline : {task.deadline}") 
            print("-" * 40) 

        # 6. Có thể xuất ngược lại JSON sạch để gửi đi API khác hoặc lưu DB [cite: 446, 486, 893]
        clean_json = meeting_tasks.model_dump_json(indent=2)
        print("\nFinal Clean JSON Output:")
        print(clean_json)

    except ValueError as e: 
        print(f"\n[ERROR] Pipeline stopped due to error: {e}") 
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Unexpected system error: {e}")

if __name__ == "__main__":
    main()