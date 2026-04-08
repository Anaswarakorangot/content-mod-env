from fastapi import APIRouter
from openenv.core import create_app

from environment import (
    TASKS,
    ContentAction,
    ContentModerationEnv,
    ContentObservation,
)
from graders.graders import GRADERS

_shared_env = ContentModerationEnv()


def env_factory():
    return _shared_env


app = create_app(
    env=env_factory,
    action_cls=ContentAction,
    observation_cls=ContentObservation,
    env_name="content-moderation-env",
)

ui_router = APIRouter()


@ui_router.get("/")
def root():
    return {
        "status": "ok",
        "env": "content-moderation-env",
        "message": "Use /reset, /step, /state, and /tasks to interact with the benchmark.",
    }


@ui_router.get("/health")
def health():
    return {"status": "healthy"}


@ui_router.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                "name": task["name"],
                "difficulty": task.get("difficulty", "unknown"),
                "description": task["description"],
                "objective": task["description"],
                "max_steps": len(task["posts"]),
                "num_items": len(task["posts"]),
                "grader": task_id in GRADERS,
                "grader_name": "graders.graders:grade_task" if task_id in GRADERS else None,
            }
            for task_id, task in TASKS.items()
        ]
    }


@ui_router.get("/validate")
def validate():
    checks = {
        "openenv_yaml": True,
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
        "min_3_tasks": len(TASKS) >= 3,
        "all_tasks_have_graders": all(task_id in GRADERS for task_id in TASKS),
        "grader_count": len(GRADERS),
    }
    return {
        "valid": checks["min_3_tasks"] and checks["all_tasks_have_graders"],
        "checks": checks,
        "env_name": "content-moderation-env",
    }


@ui_router.post("/reset")
def reset(request: dict = None):
    task_id = (request or {}).get("task_id", 1)
    obs = _shared_env.reset(task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done
    }


@ui_router.post("/step")
def step(request: dict):
    action_data = request.get("action", {})
    action_obj = ContentAction(**action_data)
    obs = _shared_env.step(action_obj)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done
    }


@ui_router.get("/grade/{task_id}")
def grade(task_id: int):
    if task_id not in GRADERS:
        return {"error": f"No grader registered for task {task_id}"}
    return GRADERS[task_id](task_id, _shared_env.action_history)


app.router.routes = ui_router.routes + app.router.routes


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
