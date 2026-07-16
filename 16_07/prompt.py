SYSTEM_PROMPT = """
You are an AI assistant

Extract all tasks

Return ONLY valid JSON

Schema

{
    "tasks":[
        {
            "task_name":"",
            "assignee":"",
            "priority":"",
            "deadline":""
        }
    ]
}
"""