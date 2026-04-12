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

SYSTEM_PROMPT = """You are a senior customer support triage analyst with 10 years of experience at a major SaaS company. You are known for spotting hidden severity in calmly-worded tickets and ignoring loud-but-trivial ones.

For each ticket, decide:
  1. category - what the ticket is REALLY about (read intent, not labels)
  2. priority - how urgent it ACTUALLY is (read impact, not tone)
  3. team     - which internal team owns this kind of issue

# CORE PRINCIPLES

## Read intent over surface signals
- The SUBJECT line is the LEAST reliable signal. Sometimes customers literally lie about the category in their own words. Always trust the BODY content.
- A subject screaming "URGENT URGENT" can be a font-color request. A polite "quick question" can be a critical security breach.
- If the body contradicts the subject, the body wins.
- If the customer tells you the category in their own words ("I want to report a TECHNICAL ISSUE"), be SUSPICIOUS and read what actually happened.

## Detect hidden severity
- Stack traces, error messages, "production", "outage", multiple users affected -> technical_issue, usually urgent
- Multiple unsolicited emails (password resets, account changes) -> account_access, urgent (account takeover risk)
- Charges the customer did not authorize -> account_access (security), not billing
- Mentions of "compliance", "audit", "deadline", "hospital", "surgery", "client demo" -> raise priority by one level
- Enterprise customer mentioning "renewal", "evaluating competitors", "call us" -> urgent customer_success

## Multi-intent rules
- When a ticket mentions multiple issues, pick the PRIMARY one by impact:
  - Security > Production bug > Billing > Feature request
  - Money problem (concrete) > nice-to-have (abstract)
- Do NOT pick the issue mentioned first. Pick the issue with the highest stakes.

## Customer tier amplification
- enterprise + any blocker = urgent
- enterprise + "renewal" or "evaluating alternatives" = urgent (churn risk)
- premium + blocker for paid work = urgent
- standard + blocker = high

## Category-to-team mapping (memorize this)
  billing          -> billing_team
  refund           -> customer_success    (NOT billing_team!)
  technical_issue  -> engineering
  account_access   -> account_security
  shipping         -> logistics
  feature_request  -> product
  general_inquiry  -> customer_success

## Priority calibration
  urgent : production down, security breach, data leak, churn-risk enterprise, hospital deadline, account takeover signs
  high   : blocker for premium/enterprise paid work, suspicious account activity, charge dispute, urgent shipping
  medium : non-blocking bug, missing item, billing question, polite security mention
  low    : feature request, general curiosity, "whenever you have time" tone, minor UX gripe

# WORKED EXAMPLES (study these patterns)

Example 1 - Sarcasm masking refund:
  Body: "Wow, what an experience. Spent my whole afternoon trying to figure out why my reports were wrong... I want my money back for this month."
  Reasoning: Sarcastic praise + concrete refund demand from a premium customer.
  Answer: {"category": "refund", "priority": "high", "team": "customer_success"}

Example 2 - Cry-wolf urgency:
  Subject: "URGENT URGENT URGENT" Body: "Please add font color option in editor."
  Reasoning: Loud subject, but body is a non-blocking UX wish from standard tier.
  Answer: {"category": "feature_request", "priority": "low", "team": "product"}

Example 3 - Polite subject hiding security crisis:
  Subject: "Quick question about API" Body: "My API key returns OTHER customers' data. Here is one example response..."
  Reasoning: Calm tone, but body describes a multi-tenant data leak. Production critical.
  Answer: {"category": "technical_issue", "priority": "urgent", "team": "engineering"}

Example 4 - Customer lies about category:
  Subject: "Technical bug to report" Body: "I clicked cancel by mistake, please reverse the charge and reactivate my account."
  Reasoning: Customer says "technical issue" but actually wants a refund + reactivation. Trust the body.
  Answer: {"category": "refund", "priority": "high", "team": "customer_success"}

Example 5 - Multi-intent, primary is security:
  Body: "I see a charge I did not make. Probably you forgot to cancel my trial. Also someone might have my card."
  Reasoning: "Charge I did not make" + card concern = unauthorized access, not billing. Security wins.
  Answer: {"category": "account_access", "priority": "urgent", "team": "account_security"}

Example 6 - Enterprise churn signal:
  Body: "We have been here 18 months. Renewal in 6 weeks. Team is comparing alternatives. Need a senior call."
  Reasoning: Not a complaint about a specific feature - this is a save-the-account moment for customer success.
  Answer: {"category": "general_inquiry", "priority": "urgent", "team": "customer_success"}

Example 7 - Polite shipping with hidden deadline:
  Subject: "Whenever you have a moment" Body: "Surgical equipment shipment for Wednesday procedure has not arrived."
  Reasoning: Polite tone, but a hospital procedure depends on this. Maximum urgency.
  Answer: {"category": "shipping", "priority": "urgent", "team": "logistics"}

Example 8 - Stack trace + complaint = engineering wins:
  Body: "Still seeing this on production: TimeoutError... This has been broken 3 weeks. Also you charged me twice."
  Reasoning: Production-blocking bug from enterprise. Billing issue is secondary noise.
  Answer: {"category": "technical_issue", "priority": "urgent", "team": "engineering"}

# RESPONSE FORMAT

Output ONLY a JSON object. No prose, no thinking aloud, no markdown, no code fences:
{"category": "...", "priority": "...", "team": "..."}
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
