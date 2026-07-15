import json

from models import Task


def parse_task(response_text: str):

    try:

        data = json.loads(
            response_text
        )


        validated_tasks = []


        for item in data:


            # kiểm tra item có phải dict không

            if not isinstance(
                item,
                dict
            ):
                continue


            task = Task(
                **item
            )


            validated_tasks.append(
                task.model_dump()
            )


        return validated_tasks


    except json.JSONDecodeError:

        return []


    except Exception as e:

        print(
            f"Parser error: {e}"
        )

        return []