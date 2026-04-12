---
title: Support Triage OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Support Triage — OpenEnv Environment

A real-world **customer support ticket triage** environment built to the
OpenEnv specification. Agents are shown incoming support tickets and must
decide, for each one:

1. **Category** — what the ticket is about
2. **Priority** — how urgent it is
3. **Team** — which internal team should handle it

## Why this environment matters

Triage is the first step of every customer-support workflow at every SaaS,
e-commerce, and fintech company. Misrouted tickets and wrong priorities
directly translate to slower resolution times, unhappy customers, and churn.
This environment models that task with **45 hand-crafted tickets** across
three difficulty tasks, including **15 adversarial cases** that even strong
frontier models struggle with.

## Task suite

| Task   | Difficulty | Tickets | What makes it hard |
|--------|-----------|---------|--------------------|
| easy   | 🟢        | 15      | Mostly aligned signals, with a few confusable subjects |
| medium | 🟡        | 15      | Vague subjects, multi-intent bodies, tier-aware urgency |
| hard   | 🔴        | 15      | Cry-wolf urgency, sarcasm, subject-vs-body mismatch, multi-language fragments, code snippets |

## Reward design — strictly in (0, 1)

Per-ticket reward components:

| Component  | Weight | Notes                                                        |
|------------|--------|--------------------------------------------------------------|
| Floor bonus| 0.05   | Any non-empty action gets a baseline (no zero rewards)       |
| Category   | 0.40   | Exact match, with cluster partial credit (billing↔refund)    |
| Priority   | 0.30   | Off-by-one partial credit, tier-weighted under-prioritization|
| Team       | 0.25   | Exact match, with small credit for using the action space    |

Per-ticket reward is **clamped to [0.05, 0.95]**, so:
- No reward is ever exactly **0.0** (garbage actions still earn the floor)
- No reward is ever exactly **1.0** (perfect actions cap at 0.95)

This guarantees task-level mean rewards stay **strictly inside (0, 1)**,
satisfying the OpenEnv hackathon validator's task-score constraint.

## Action space

```json
{
  "category": "billing | technical_issue | account_access | feature_request | shipping | refund | general_inquiry",
  "priority": "low | medium | high | urgent",
  "team":     "billing_team | engineering | account_security | product | logistics | customer_success"
}
```

## Running locally

```bash
# Install
uv sync
# or: pip install -e .

# Start the server
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_groq_key"
python inference.py
```

## OpenEnv API

Built with `openenv.core.env_server.create_app()`, exposing:

- `POST /reset` — start a new episode (`{"task": "easy|medium|hard"}`)
- `POST /step` — submit an action, get the next observation
- `GET /state` — current episode state
- `GET /schema` — typed action / observation schemas
- `WS /ws` — persistent WebSocket session (preferred for training loops)

## Project structure
triage/
├── init.py
├── README.md
├── client.py              # TriageEnv client (EnvClient subclass)
├── grader.py              # Reward function with strict (0,1) bounds
├── inference.py           # Baseline inference script (root)
├── models.py              # TriageAction, TriageObservation
├── openenv.yaml           # Environment manifest
├── pyproject.toml         # Package config + scripts entry point
├── tickets.py             # 45-ticket dataset
├── uv.lock
└── server/
├── init.py
├── app.py             # create_app() factory
├── triage_environment.py  # Environment subclass
├── Dockerfile
└── requirements.txt
## Design choices

- **Pure text in, structured JSON out.** Runs on 2 vCPU / 8 GB.
- **Deterministic graders.** Same action always yields the same reward.
- **Strict (0, 1) bounds.** Hard floor at 0.05, hard ceiling at 0.95.
- **Tier-aware reward shaping.** Enterprise customers get harsher under-prioritization penalties.
- **Adversarial test cases.** Hard task is calibrated so even strong frontier models cannot achieve a perfect score.

## License

MIT
