from pydantic import BaseModel
from typing import List

class Task(BaseModel):
    task_name:str
    assignee: str
    priority: str
    deadline: str
    
class MeetingTasks(BaseModel):
    tasks:List[Task]