"""
Individual grader functions for each task.
Each grader is a standalone callable that survives parameterless reflection.
NO loops — each grader is explicitly defined.
"""
from typing import Any


def _compute_score(task_id: int = None, trajectory: dict = None) -> float:
    """Shared grading logic. Returns float in (0.0, 1.0)."""
    if trajectory is None:
        return 0.5  # Safe default for reflection checks
    actions = trajectory.get("actions", [])
    if task_id is not None:
        actions = [a for a in actions if a.get("task_id") == task_id]
    if not actions:
        return 0.5
    correct = sum(1 for a in actions if a.get("correct", False))
    total = len(actions)
    return min(max(round(correct / total, 3), 0.01), 0.99)


# ── Individual Grader Functions (Trap 3: default params) ────

def grade_task_1(trajectory: dict = None) -> float:
    """Grader for task 1: direct_toxicity_detection."""
    return _compute_score(1, trajectory)


def grade_task_2(trajectory: dict = None) -> float:
    """Grader for task 2: social_bias_and_nuance."""
    return _compute_score(2, trajectory)


def grade_task_3(trajectory: dict = None) -> float:
    """Grader for task 3: adversarial_expression."""
    return _compute_score(3, trajectory)


def grade_task_4(trajectory: dict = None) -> float:
    """Grader for task 4: multi_turn_contextual_harassment."""
    return _compute_score(4, trajectory)


# ── Explicit mapping — NO loop (Trap xtemper) ───────────────
GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
    4: grade_task_4,
}
