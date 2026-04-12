# Copyright (c) 2026 tanx1509
# MIT License
"""
FastAPI application for the Support Triage Environment.

Uses openenv.core.env_server.create_app() to construct an HTTP + WebSocket
server that exposes TriageEnvironment with the standard /reset, /step,
/state, /schema, and /ws endpoints.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with: uv sync"
    ) from e

# Dual-import pattern: works both as installed package and as standalone server.
try:
    from ..models import TriageAction, TriageObservation
    from .triage_environment import TriageEnvironment
except (ImportError, ModuleNotFoundError):
    from models import TriageAction, TriageObservation
    from server.triage_environment import TriageEnvironment


# Build the FastAPI app via the official factory.
# Pass the CLASS (not an instance) so the server can create per-session
# environment instances when concurrent WebSocket sessions are enabled.
app = create_app(
    TriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="triage",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point referenced by [project.scripts] in pyproject.toml.

    Run via:
        uv run --project . server
        python -m triage.server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
