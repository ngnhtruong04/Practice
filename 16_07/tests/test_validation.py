import pytest
from models import Task, MeetingTasks
from validator import validate_meeting_tasks

def test_valid_task(): [cite: 1031]
    task = Task(
        task_name="Deploy API", [cite: 1031]
        assignee="John", [cite: 1031]
        priority="High", [cite: 1031]
        deadline="Friday" [cite: 1031]
    )
    assert task.assignee == "John" [cite: 1031]

def test_missing_field(): [cite: 1031]
    # Thiếu các trường bắt buộc như assignee, priority, deadline
    data = {"tasks": [{"task_name": "Deploy API"}]}
    with pytest.raises(ValueError): [cite: 1031]
        validate_meeting_tasks(data)

def test_empty_array_is_valid(): [cite: 1007]
    # Cuộc họp không có task nào vẫn phải hợp lệ theo Pydantic List[Task] [cite: 1007, 1009]
    data = {"tasks": []} [cite: 1006]
    result = validate_meeting_tasks(data)
    assert len(result.tasks) == 0