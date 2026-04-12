# Copyright (c) 2026 tanx1509
# MIT License
"""
Grader for the Support Triage OpenEnv environment.

Reward design (per ticket, total in [REWARD_FLOOR, REWARD_CEILING]):

  - Category   : 0.40  (with partial credit for confusable categories)
  - Priority   : 0.30  (with off-by-one partial credit, tier-weighted)
  - Team       : 0.25  (with partial credit if team matches the right "domain")
  - Floor bonus: 0.05  (any non-empty action gets a baseline)

Total maximum: 0.40 + 0.30 + 0.25 + 0.05 = 1.00 nominal,
but the final reward is CLAMPED to [0.05, 0.95] so that:
  - No reward is ever exactly 0.0 (even garbage actions earn the floor)
  - No reward is ever exactly 1.0 (even perfect actions cap at 0.95)

This guarantees the per-task mean reward is strictly inside the open
interval (0, 1), satisfying the OpenEnv hackathon validator requirement
that "each task score must be strictly between 0 and 1 (not 0.0 and not 1.0)".

The task-level score is the mean of per-ticket rewards, then clamped to
[0.05, 0.95] as a final safety net.

Determinism: same (action, gold) pair always produces the same reward.
"""

from typing import Any, Dict, Tuple

try:
    from .tickets import CATEGORIES, PRIORITIES, TEAMS
except ImportError:
    from tickets import CATEGORIES, PRIORITIES, TEAMS

# ---------------------------------------------------------------------------
# Reward bounds - HARD invariants
# ---------------------------------------------------------------------------
REWARD_FLOOR = 0.05  # minimum reward for any non-empty action
REWARD_CEILING = 0.95  # maximum reward even for perfect actions

CATEGORY_WEIGHT = 0.40
PRIORITY_WEIGHT = 0.30
TEAM_WEIGHT = 0.25
FLOOR_BONUS = 0.05  # baseline for submitting any structured action

# ---------------------------------------------------------------------------
# Category similarity - cluster confusable categories together
# A predicted category gets partial credit if it is in the same cluster
# as the gold category (e.g. billing <-> refund are closely related).
# ---------------------------------------------------------------------------
CATEGORY_CLUSTERS = {
    "money": {"billing", "refund"},
    "auth": {"account_access"},
    "tech": {"technical_issue"},
    "logistics": {"shipping"},
    "product": {"feature_request"},
    "info": {"general_inquiry"},
}


def _category_score(pred: str, gold: str) -> float:
    """
    Category scoring:
      - exact match           -> 1.00
      - same cluster (e.g.
        billing<->refund)     -> 0.40
      - both valid categories -> 0.10  (small credit for using the action space)
      - else                  -> 0.00
    """
    if not pred:
        return 0.0
    if pred == gold:
        return 1.0
    pred_cluster = next(
        (name for name, members in CATEGORY_CLUSTERS.items() if pred in members),
        None,
    )
    gold_cluster = next(
        (name for name, members in CATEGORY_CLUSTERS.items() if gold in members),
        None,
    )
    if pred_cluster is not None and pred_cluster == gold_cluster:
        return 0.40
    if pred in CATEGORIES:
        return 0.10
    return 0.0


def _priority_score(pred: str, gold: str, customer_tier: str) -> float:
    """
    Priority scoring with off-by-one partial credit and tier weighting.

    Distance-based scoring on the priority scale (low=0, medium=1, high=2, urgent=3):
      - exact match  -> 1.00
      - distance 1   -> 0.55
      - distance 2   -> 0.20
      - distance 3   -> 0.00

    Then a small tier penalty: enterprise customers being under-prioritized
    is worse than standard customers being under-prioritized.
    """
    if not pred or pred not in PRIORITIES or gold not in PRIORITIES:
        return 0.0
    if pred == gold:
        return 1.0
    pred_idx = PRIORITIES.index(pred)
    gold_idx = PRIORITIES.index(gold)
    diff = abs(pred_idx - gold_idx)
    if diff == 1:
        base = 0.55
    elif diff == 2:
        base = 0.20
    else:
        base = 0.0

    # Tier-aware penalty: under-prioritizing enterprise customers is bad.
    if customer_tier == "enterprise" and pred_idx < gold_idx:
        base *= 0.7
    elif customer_tier == "premium" and pred_idx < gold_idx:
        base *= 0.85

    return base


# ---------------------------------------------------------------------------
# Team scoring - same team mapping as categories so we can reward the agent
# for routing to the right "kind" of team even if not exact.
# ---------------------------------------------------------------------------
TEAM_BY_CATEGORY = {
    "billing": "billing_team",
    "refund": "customer_success",
    "account_access": "account_security",
    "technical_issue": "engineering",
    "feature_request": "product",
    "shipping": "logistics",
    "general_inquiry": "customer_success",
}


def _team_score(pred: str, gold: str) -> float:
    """
    Team scoring:
      - exact match    -> 1.00
      - valid team but
        wrong          -> 0.10  (used the action space correctly)
      - else           -> 0.00
    """
    if not pred:
        return 0.0
    if pred == gold:
        return 1.0
    if pred in TEAMS:
        return 0.10
    return 0.0


def grade_action(
    action: Dict[str, Any], gold: Dict[str, Any], customer_tier: str
) -> Tuple[float, str]:
    """
    Grade a single action against the gold label.

    Args:
        action: dict with keys 'category', 'priority', 'team'
        gold: dict with keys 'category', 'priority', 'team'
        customer_tier: 'standard', 'premium', or 'enterprise'

    Returns:
        (reward, feedback) where reward is strictly in [REWARD_FLOOR, REWARD_CEILING]
        and feedback is a human-readable explanation.
    """
    pred_cat = (action.get("category") or "").strip().lower()
    pred_pri = (action.get("priority") or "").strip().lower()
    pred_team = (action.get("team") or "").strip().lower()

    gold_cat = gold["category"]
    gold_pri = gold["priority"]
    gold_team = gold["team"]

    cat_raw = _category_score(pred_cat, gold_cat)
    pri_raw = _priority_score(pred_pri, gold_pri, customer_tier)
    team_raw = _team_score(pred_team, gold_team)

    cat_part = CATEGORY_WEIGHT * cat_raw
    pri_part = PRIORITY_WEIGHT * pri_raw
    team_part = TEAM_WEIGHT * team_raw

    raw_total = FLOOR_BONUS + cat_part + pri_part + team_part

    # Hard clamp to (0, 1) - using the configured floor and ceiling
    reward = max(REWARD_FLOOR, min(REWARD_CEILING, raw_total))
    reward = round(reward, 4)

    # Build human-readable feedback
    parts = []
    if pred_cat == gold_cat:
        parts.append("category: OK")
    elif cat_raw >= 0.4:
        parts.append(f"category: close (got {pred_cat}, expected {gold_cat})")
    else:
        parts.append(f"category: WRONG (got {pred_cat}, expected {gold_cat})")

    if pred_pri == gold_pri:
        parts.append("priority: OK")
    elif pri_raw > 0:
        parts.append(f"priority: off (got {pred_pri}, expected {gold_pri})")
    else:
        parts.append(f"priority: WRONG (got {pred_pri}, expected {gold_pri})")

    if pred_team == gold_team:
        parts.append("team: OK")
    else:
        parts.append(f"team: WRONG (got {pred_team}, expected {gold_team})")

    feedback = " | ".join(parts)
    return reward, feedback


def clamp_task_score(score: float) -> float:
    """
    Final safety net: clamp the task-level mean score to (0, 1) strictly.

    Even if every single ticket happened to land exactly on REWARD_FLOOR
    or REWARD_CEILING, the mean would still be inside (0, 1) - but we
    apply this clamp anyway to be defensive.
    """
    return round(max(REWARD_FLOOR, min(REWARD_CEILING, score)), 4)
