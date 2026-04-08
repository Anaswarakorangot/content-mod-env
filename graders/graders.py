from typing import Any
from environment import TASKS


def grade_task(task_id: int, action_history: list[dict[str, Any]]) -> dict[str, Any]:
    task = TASKS[task_id]
    task_history = [item for item in action_history if item.get("task_id") == task_id]
    expected_steps = len(task["posts"])
    graded_steps = len(task_history)
    total_score = round(
        sum(float(item.get("score", 0.0)) for item in task_history), 3
    )
    accuracy = (
        round(
            sum(1 for item in task_history if item.get("correct")) / graded_steps, 3
        )
        if graded_steps
        else 0.0
    )
    normalized_score = round(total_score / expected_steps, 3) if expected_steps else 0.0
    return {
        "task_id": task_id,
        "task_name": task["name"],
        "score": normalized_score,
        "accuracy": accuracy,
        "graded_steps": graded_steps,
        "expected_steps": expected_steps,
        "passed": graded_steps == expected_steps,
    }


GRADERS = {task_id: grade_task for task_id in TASKS}
