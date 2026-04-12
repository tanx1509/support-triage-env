# Copyright (c) 2026 tanx1509
# MIT License
"""
Support Triage Environment Implementation.

A real-world customer support ticket triage environment. The agent is
shown a stream of tickets and must classify each one along three
dimensions: category, priority, and team.

Three difficulty tasks (easy / medium / hard), 15 tickets each.
"""

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Dual-import pattern: works both as installed package and as standalone server.
try:
    from ..models import TriageAction, TriageObservation
    from ..tickets import CATEGORIES, PRIORITIES, TASKS, TEAMS
    from ..grader import clamp_task_score, grade_action
except ImportError:
    from models import TriageAction, TriageObservation
    from tickets import CATEGORIES, PRIORITIES, TASKS, TEAMS
    from grader import clamp_task_score, grade_action


DEFAULT_TASK = "easy"


class TriageEnvironment(Environment):
    """
    Support Triage environment.

    Each episode plays through one of three task difficulties:
    - easy   : 15 tickets, surface-signal-aligned
    - medium : 15 tickets, ambiguous wording, tier-aware urgency
    - hard   : 15 tickets, adversarial (cry-wolf, sarcasm, mismatch)

    Per-ticket reward is in [0.05, 0.95] (strictly inside (0, 1)).
    Task-level mean reward is therefore also strictly inside (0, 1),
    satisfying the OpenEnv hackathon validator constraint.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: str = DEFAULT_TASK
        self._tickets: list = []
        self._idx: int = 0
        self._cum_reward: float = 0.0
        self._rewards: list = []
        self._last_feedback: str = ""
        self._done: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, task: Optional[str] = None) -> TriageObservation:  # type: ignore[override]
        """
        Reset the environment to the start of a task.

        Args:
            task: One of "easy", "medium", "hard". Defaults to "easy".

        Returns:
            TriageObservation showing the first ticket.
        """
        chosen = (task or DEFAULT_TASK).strip().lower()
        if chosen not in TASKS:
            chosen = DEFAULT_TASK

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = chosen
        self._tickets = TASKS[chosen]
        self._idx = 0
        self._cum_reward = 0.0
        self._rewards = []
        self._last_feedback = ""
        self._done = False

        return self._observe(reward=0.0)

    def step(self, action: TriageAction) -> TriageObservation:  # type: ignore[override]
        """
        Submit an action for the current ticket and advance.

        Args:
            action: TriageAction with category, priority, team.

        Returns:
            TriageObservation with reward, feedback, and the next ticket
            (or done=True if the episode is finished).
        """
        if self._done:
            # Defensive: if the agent calls step on a done episode, return
            # a terminal observation rather than crashing.
            return self._observe(reward=0.0)

        current = self._tickets[self._idx]
        gold = current["gold"]
        tier = current.get("customer_tier", "standard")

        action_dict = {
            "category": action.category,
            "priority": action.priority,
            "team": action.team,
        }
        reward, feedback = grade_action(action_dict, gold, tier)

        self._cum_reward += reward
        self._rewards.append(reward)
        self._last_feedback = feedback
        self._state.step_count += 1

        self._idx += 1
        if self._idx >= len(self._tickets):
            self._done = True

        return self._observe(reward=reward)

    @property
    def state(self) -> State:
        """Return the current State (episode_id, step_count)."""
        return self._state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _observe(self, reward: float) -> TriageObservation:
        """Build a TriageObservation reflecting the current environment state."""
        ticket_id: Optional[str] = None
        subject: Optional[str] = None
        body: Optional[str] = None
        customer_tier: Optional[str] = None

        if not self._done and self._idx < len(self._tickets):
            t = self._tickets[self._idx]
            ticket_id = t["id"]
            subject = t["subject"]
            body = t["body"]
            customer_tier = t.get("customer_tier", "standard")

        mean_reward = (
            clamp_task_score(sum(self._rewards) / len(self._rewards)) if self._rewards else 0.05
        )

        return TriageObservation(
            done=self._done,
            reward=reward,
            ticket_id=ticket_id,
            subject=subject,
            body=body,
            customer_tier=customer_tier,
            ticket_index=self._idx,
            total_tickets=len(self._tickets),
            task=self._task,
            allowed_categories=CATEGORIES,
            allowed_priorities=PRIORITIES,
            allowed_teams=TEAMS,
            last_feedback=self._last_feedback,
            cumulative_reward=round(self._cum_reward, 4),
            mean_reward=round(mean_reward, 4),
            metadata={
                "task": self._task,
                "ticket_id": ticket_id,
                "step_count": self._state.step_count,
            },
        )

    def task_score(self) -> float:
        """Return the clamped mean reward for the current episode."""
        if not self._rewards:
            return 0.0
        mean = sum(self._rewards) / len(self._rewards)
        return clamp_task_score(mean)
