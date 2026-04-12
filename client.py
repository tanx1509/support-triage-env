# Copyright (c) 2026 tanx1509
# MIT License
"""
Client for the Support Triage OpenEnv environment.

Provides TriageEnv, an EnvClient subclass that wraps WebSocket
communication with a running TriageEnvironment server.
"""

from openenv.core.client.env_client import EnvClient

try:
    from .models import TriageAction, TriageObservation
except ImportError:
    from models import TriageAction, TriageObservation


class TriageEnv(EnvClient[TriageAction, TriageObservation]):
    """
    Client for the Support Triage environment.

    Example (async):
        >>> async with TriageEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     result = await env.step(TriageAction(
        ...         category="billing",
        ...         priority="high",
        ...         team="billing_team",
        ...     ))

    Example (sync):
        >>> with TriageEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     result = env.step(TriageAction(
        ...         category="billing",
        ...         priority="high",
        ...         team="billing_team",
        ...     ))
    """

    action_type = TriageAction
    observation_type = TriageObservation
