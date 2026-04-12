# Copyright (c) 2026 tanx1509
# MIT License
"""
Baseline inference script for the Support Triage OpenEnv environment.

Runs all three tasks (easy / medium / hard) against a running environment
server using an OpenAI-compatible LLM client. Emits structured stdout logs
in the [START] / [STEP] / [END] / [SUMMARY] format required by the
OpenEnv hackathon evaluator.

Required env vars:
    API_BASE_URL   OpenAI-compatible base URL  (default: Groq)
    MODEL_NAME     Model identifier            (default: llama-3.1-8b-instant)
    HF_TOKEN       API key for the provider    (no default - REQUIRED)

Optional:
    LOCAL_IMAGE_NAME  Used if running env from a local Docker image
    ENV_BASE_URL      URL of the running env server (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests
from openai import OpenAI

# Required env vars (with sensible defaults except HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - used if running environment from a local Docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert customer support triage assistant.

For each ticket you receive, decide:
  1. category - one of the allowed categories
  2. priority - one of the allowed priorities (low, medium, high, urgent)
  3. team     - one of the allowed teams to route the ticket to

Important guidelines:
- Read the ticket BODY carefully, not just the subject. Subjects can mislead.
- A "URGENT" subject does not always mean urgent priority - judge by content.
- Customer tier matters: enterprise customers get higher priority for the same issue.
- Multi-issue tickets: focus on the PRIMARY (most impactful) issue.
- Sarcastic or polite phrasing can hide real urgency - read intent.

Respond ONLY with a compact JSON object:
{"category": "...", "priority": "...", "team": "..."}
No extra text, no markdown, no code fences.
"""


def log_start(task: str, total: int) -> None:
    print(f"[START] task={task} total_tickets={total}", flush=True)


def log_step(
    task: str,
    idx: int,
    ticket_id: str,
    action: Dict[str, Any],
    reward: float,
    cumulative: float,
    feedback: str,
) -> None:
    print(
        f"[STEP] task={task} index={idx} ticket_id={ticket_id} "
        f'action={json.dumps(action)} reward={reward:.4f} '
        f'cumulative={cumulative:.4f} feedback="{feedback}"',
        flush=True,
    )


def log_end(task: str, mean_reward: float, total: int) -> None:
    print(
        f"[END] task={task} mean_reward={mean_reward:.4f} total_tickets={total}",
        flush=True,
    )


def wait_for_env(url: str, timeout: int = 60) -> None:
    start = time.time()
    while time.time() - start < timeout:
        for path in ("/", "/health"):
            try:
                r = requests.get(f"{url}{path}", timeout=3)
                if r.status_code == 200:
                    return
            except Exception:
                pass
        time.sleep(1)
    raise RuntimeError(f"Environment at {url} not ready in {timeout}s")


def parse_action(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction with safe fallback."""
    fallback = {
        "category": "general_inquiry",
        "priority": "medium",
        "team": "customer_success",
    }
    if not raw:
        return fallback
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return fallback
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return fallback
    return {
        "category": str(obj.get("category", fallback["category"])).strip().lower(),
        "priority": str(obj.get("priority", fallback["priority"])).strip().lower(),
        "team": str(obj.get("team", fallback["team"])).strip().lower(),
    }


def unwrap(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the {observation: {...}, reward: x, done: y} envelope.

    OpenEnv's create_app wraps responses in an outer envelope; the
    inference loop wants a single flat dict with both observation
    fields AND reward/done at the top level.
    """
    if not isinstance(resp_json, dict):
        return {}
    inner = resp_json.get("observation") or {}
    flat = dict(inner)
    if "reward" in resp_json:
        flat["reward"] = resp_json["reward"]
    if "done" in resp_json:
        flat["done"] = resp_json["done"]
    return flat


def call_llm(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    user_msg = (
        f"Allowed categories: {obs.get('allowed_categories')}\n"
        f"Allowed priorities: {obs.get('allowed_priorities')}\n"
        f"Allowed teams: {obs.get('allowed_teams')}\n\n"
        f"Ticket ID: {obs.get('ticket_id')}\n"
        f"Customer tier: {obs.get('customer_tier')}\n"
        f"Subject: {obs.get('subject')}\n"
        f"Body: {obs.get('body')}\n\n"
        f"Return only the JSON object."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", flush=True)
        raw = ""
    return parse_action(raw)


def run_task(client: OpenAI, task: str) -> float:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task},
        timeout=15,
    )
    r.raise_for_status()
    obs = unwrap(r.json())

    total = obs.get("total_tickets", 0)
    log_start(task, total)

    idx = 0
    while not obs.get("done", False):
        ticket_id = obs.get("ticket_id", "?") or "?"
        action = call_llm(client, obs)

        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action},
            timeout=30,
        )
        r.raise_for_status()
        obs = unwrap(r.json())

        log_step(
            task=task,
            idx=idx,
            ticket_id=ticket_id,
            action=action,
            reward=float(obs.get("reward", 0.0)),
            cumulative=float(obs.get("cumulative_reward", 0.0)),
            feedback=obs.get("last_feedback", ""),
        )
        idx += 1

    mean_reward = float(obs.get("mean_reward", 0.0))
    log_end(task, mean_reward, total)
    return mean_reward


def main() -> int:
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN env var not set", flush=True)
        return 1

    print(f"[INFO] Waiting for environment at {ENV_BASE_URL} ...", flush=True)
    wait_for_env(ENV_BASE_URL)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores: Dict[str, float] = {}
    for task in TASKS:
        try:
            scores[task] = run_task(client, task)
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", flush=True)
            scores[task] = 0.05  # floor, never 0.0

    overall = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
    print(
        f"[SUMMARY] scores={json.dumps(scores)} overall={overall:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
