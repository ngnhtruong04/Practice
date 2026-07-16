from typing import List
from pydantic import BaseModel, Field

class Task(BaseModel):
    task_name: str = Field(description="Tên công việc hoặc hành động cần thực hiện")
    assignee: str = Field(description="Người chịu trách nhiệm thực hiện")
    priority: str = Field(description="Mức độ ưu tiên (Low, Medium, High)")
    deadline: str = Field(description="Thời hạn hoàn thành công việc")

class MeetingTasks(BaseModel):
    tasks: List[Task] = Field(description="Danh sách các công việc được trích xuất") 