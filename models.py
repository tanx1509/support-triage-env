# Copyright (c) 2026 tanx1509
# MIT License
"""
Data models for the Support Triage OpenEnv environment.

Defines:
- TriageAction: what the agent submits per ticket (category + priority + team)
- TriageObservation: what the environment returns (ticket view + reward + feedback)

Both inherit from openenv.core.env_server.types base classes which provide
standard fields like `done`, `reward`, and `metadata`.
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TriageAction(Action):
    """
    Action submitted by the agent for the current ticket.

    The agent must classify the ticket along three dimensions:
    - category: what the ticket is about
    - priority: how urgent it is
    - team: which internal team should handle it
    """

    category: str = Field(
        ...,
        description="Ticket category (e.g. billing, technical_issue, account_access)",
    )
    priority: str = Field(
        ...,
        description="Priority level: low, medium, high, or urgent",
    )
    team: str = Field(
        ...,
        description="Team to route the ticket to",
    )


class TriageObservation(Observation):
    """
    Observation returned after reset / step.

    Inherits `done`, `reward`, and `metadata` from the OpenEnv Observation base.
    Adds task-specific fields describing the current ticket and the action space.
    """

    # Current ticket fields (None when episode is done)
    ticket_id: Optional[str] = Field(default=None, description="Current ticket ID")
    subject: Optional[str] = Field(default=None, description="Ticket subject line")
    body: Optional[str] = Field(default=None, description="Ticket body text")
    customer_tier: Optional[str] = Field(
        default=None, description="Customer tier: standard, premium, or enterprise"
    )

    # Episode progress
    ticket_index: int = Field(default=0, description="Index of current ticket in task")
    total_tickets: int = Field(default=0, description="Total tickets in current task")
    task: str = Field(default="", description="Current task name (easy, medium, hard)")

    # Action space (echoed every step so the agent can build prompts)
    allowed_categories: List[str] = Field(
        default_factory=list, description="Valid category values"
    )
    allowed_priorities: List[str] = Field(
        default_factory=list, description="Valid priority values"
    )
    allowed_teams: List[str] = Field(
        default_factory=list, description="Valid team values"
    )

    # Last action feedback
    last_feedback: str = Field(
        default="", description="Human-readable explanation of the last reward"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Sum of rewards for the episode so far"
    )
    mean_reward: float = Field(
        default=0.0, description="Mean reward across graded tickets so far"
    )
