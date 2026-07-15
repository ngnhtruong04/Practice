#Sử dụng Pydantic để validate dữ liệu.
from pydantic import BaseModel

class Task(BaseModel):
    task_name: str
    
    assignee: str
    
    priority: str
    
    deadline: str