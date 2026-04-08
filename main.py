from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import ContentModerationEnv, Action, Observation, Reward

app = FastAPI(title="Content Moderation Environment")

# Global environment instance
env = ContentModerationEnv()

class ResetRequest(BaseModel):
    task_id: Optional[int] = 1

class StepRequest(BaseModel):
    decision: str
    confidence: float
    reason: str

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool

@app.get("/")
def root():
    return {"status": "ok", "env": "Content Moderation Environment"}

@app.post("/reset")
def reset(request: ResetRequest):
    obs = env.reset(task_id=request.task_id)
    return {"observation": obs, "done": False}

@app.post("/step")
def step(request: StepRequest):
    action = Action(
        decision=request.decision,
        confidence=request.confidence,
        reason=request.reason
    )
    obs, reward, done = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done)

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"task_id": 1, "name": "obvious_moderation", "difficulty": "easy"},
            {"task_id": 2, "name": "subtle_harassment", "difficulty": "medium"},
            {"task_id": 3, "name": "borderline_content", "difficulty": "hard"},
        ]
    }