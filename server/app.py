# Copyright (c) 2026 tanx1509
# MIT License
"""
FastAPI application for the Support Triage Environment.

Uses openenv.core.env_server.create_app() with a SHARED singleton
TriageEnvironment instance so that HTTP /reset and /step requests
maintain state across calls (the same episode persists between calls
on the same server, just like the Phase 1 evaluator expects).
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv-core is required. Install with: uv sync") from e

# Dual-import pattern: works both as installed package and as standalone server.
try:
    from ..models import TriageAction, TriageObservation
    from .triage_environment import TriageEnvironment
except (ImportError, ModuleNotFoundError):
    from models import TriageAction, TriageObservation
    from server.triage_environment import TriageEnvironment


# Module-level singleton: one shared environment instance for all HTTP
# requests. This makes /reset and /step stateful across HTTP calls,
# which is the behavior the Phase 1 evaluator expects when it scores
# tasks via plain HTTP (no WebSocket session).
_shared_env = TriageEnvironment()


def env_factory() -> TriageEnvironment:
    """Return the shared environment instance.

    create_app() accepts a callable; we return the same singleton on
    every call so that HTTP-mode evaluators see persistent state.
    For WebSocket sessions, OpenEnv's session manager will still wrap
    this with its own per-session bookkeeping.
    """
    return _shared_env


app = create_app(
    env_factory,
    TriageAction,
    TriageObservation,
    env_name="triage",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point referenced by [project.scripts] in pyproject.toml."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
