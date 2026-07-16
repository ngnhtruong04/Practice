import pytest
from parser import parse_json

def test_valid_json(): [cite: 1028]
    data = '{"tasks": [{"task_name": "Test", "assignee": "John", "priority": "High", "deadline": "Today"}]}'
    result = parse_json(data) [cite: 1028]
    assert "tasks" in result
    assert result["tasks"][0]["assignee"] == "John" [cite: 1028]

def test_invalid_json(): [cite: 1028]
    # Thiếu dấu ngoặc nhọn đóng hoặc sai cú pháp
    data = '{"tasks": [{task_name: "Test"}]}' [cite: 1028]
    with pytest.raises(ValueError): [cite: 1028]
        parse_json(data) [cite: 1028]