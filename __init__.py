# Copyright (c) 2026 tanx1509
# MIT License
"""
Support Triage OpenEnv environment.

Public exports:
    TriageAction       - action submitted by the agent
    TriageObservation  - observation returned by the environment
    TriageEnv          - client class (from client.py)
"""

from .models import TriageAction, TriageObservation

try:
    from .client import TriageEnv  # noqa: F401
except Exception:
    # client.py may import optional dependencies that are not present in
    # all install modes. Failing to import the client should not break
    # server-side imports.
    pass

__all__ = ["TriageAction", "TriageObservation"]
