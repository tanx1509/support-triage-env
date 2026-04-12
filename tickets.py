# Copyright (c) 2026 tanx1509
# MIT License
"""
Ticket dataset for the Support Triage OpenEnv environment.

Three tasks of progressive difficulty, 15 tickets each (45 total).
Each ticket has a gold-standard label for category, priority, and team.

Design philosophy:
- Easy: surface signals (subject keywords) usually point to the right answer,
  but a few tickets have a confusable element so a perfect 15/15 is non-trivial.
- Medium: ambiguous subjects, multi-intent bodies, customer-tier-aware urgency.
- Hard: adversarial - cry-wolf urgency, subject-vs-body mismatch, sarcasm,
  passive-aggressive language, code snippets and stack traces, deceptive priority.

The hard task is intentionally calibrated so that even a strong frontier model
will not score above ~0.85 on average, ensuring scores stay strictly inside (0, 1).
"""

from typing import Any, Dict, List

CATEGORIES = [
    "billing",
    "technical_issue",
    "account_access",
    "feature_request",
    "shipping",
    "refund",
    "general_inquiry",
]

PRIORITIES = ["low", "medium", "high", "urgent"]

TEAMS = [
    "billing_team",
    "engineering",
    "account_security",
    "product",
    "logistics",
    "customer_success",
]


# ---------------------------------------------------------------------------
# EASY TASK: 15 tickets
# Surface signals usually align with the gold label, but a few tickets
# have one confusable element (wrong-sounding subject, mixed signals, etc.)
# ---------------------------------------------------------------------------
EASY_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "E001",
        "subject": "Wrong amount charged on my credit card",
        "body": "Hi, I was charged $99 instead of $49 for my monthly plan. Please refund the difference.",
        "customer_tier": "standard",
        "gold": {"category": "billing", "priority": "high", "team": "billing_team"},
    },
    {
        "id": "E002",
        "subject": "Cannot log into my account",
        "body": "I forgot my password and the reset email is not arriving. Please help.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "high", "team": "account_security"},
    },
    {
        "id": "E003",
        "subject": "Where is my order?",
        "body": "I placed order #45821 five days ago and tracking has not updated. When will it arrive?",
        "customer_tier": "standard",
        "gold": {"category": "shipping", "priority": "medium", "team": "logistics"},
    },
    {
        "id": "E004",
        "subject": "Feature suggestion: dark mode",
        "body": "It would be great if the app had a dark mode for nighttime use. Thanks!",
        "customer_tier": "standard",
        "gold": {"category": "feature_request", "priority": "low", "team": "product"},
    },
    {
        "id": "E005",
        "subject": "App crashes on startup",
        "body": "Since the latest update, the app crashes immediately when I open it on my iPhone 14.",
        "customer_tier": "standard",
        "gold": {"category": "technical_issue", "priority": "high", "team": "engineering"},
    },
    {
        "id": "E006",
        "subject": "Refund for cancelled order",
        "body": "I cancelled order #92344 yesterday but I have not received the refund yet. It has been 24 hours.",
        "customer_tier": "standard",
        "gold": {"category": "refund", "priority": "medium", "team": "customer_success"},
    },
    {
        "id": "E007",
        "subject": "How do I export my data?",
        "body": "Quick question - is there a way to export my transaction history as CSV? Could not find it in the menu.",
        "customer_tier": "standard",
        "gold": {"category": "general_inquiry", "priority": "low", "team": "customer_success"},
    },
    {
        "id": "E008",
        "subject": "Two-factor authentication not working",
        "body": "I am not receiving the 2FA codes on my phone. I have checked spam folder and signal strength is fine.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "high", "team": "account_security"},
    },
    {
        "id": "E009",
        "subject": "Damaged package received",
        "body": "The box arrived crushed and one item inside is broken. Photos attached. Need a replacement.",
        "customer_tier": "standard",
        "gold": {"category": "shipping", "priority": "medium", "team": "logistics"},
    },
    {
        "id": "E010",
        "subject": "Charged twice for same subscription",
        "body": "I see two charges of $29.99 on the same date for my monthly plan. Please refund one.",
        "customer_tier": "standard",
        "gold": {"category": "billing", "priority": "high", "team": "billing_team"},
    },
    # Confusable: subject says "feature" but the body is actually a bug report.
    {
        "id": "E011",
        "subject": "Feature is broken",
        "body": "The CSV export feature you added last month is generating empty files. Worked fine until last week.",
        "customer_tier": "standard",
        "gold": {"category": "technical_issue", "priority": "medium", "team": "engineering"},
    },
    {
        "id": "E012",
        "subject": "Please add Spanish language support",
        "body": "Hola, would love to see the dashboard available in Spanish. Many of my users are in LATAM.",
        "customer_tier": "standard",
        "gold": {"category": "feature_request", "priority": "low", "team": "product"},
    },
    {
        "id": "E013",
        "subject": "Account locked after too many login attempts",
        "body": "I tried logging in a few times with old password and now my account is locked. How do I unlock it?",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "medium", "team": "account_security"},
    },
    {
        "id": "E014",
        "subject": "Refund request for unused subscription",
        "body": "I signed up by mistake yesterday and never used the service. Can I please get a refund?",
        "customer_tier": "standard",
        "gold": {"category": "refund", "priority": "low", "team": "customer_success"},
    },
    {
        "id": "E015",
        "subject": "Question about pricing tiers",
        "body": "Curious what is included in the Pro plan vs Business plan. The pricing page is a bit unclear.",
        "customer_tier": "standard",
        "gold": {"category": "general_inquiry", "priority": "low", "team": "customer_success"},
    },
]


# ---------------------------------------------------------------------------
# MEDIUM TASK: 15 tickets
# Vague subjects, multi-intent bodies, customer-tier-aware urgency.
# ---------------------------------------------------------------------------
MEDIUM_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "M001",
        "subject": "Subscription issue",
        "body": "I upgraded to premium last week but the premium features are still locked. I already paid. Getting frustrated as I need this for a client demo tomorrow.",
        "customer_tier": "premium",
        "gold": {"category": "billing", "priority": "urgent", "team": "billing_team"},
    },
    {
        "id": "M002",
        "subject": "Strange behavior",
        "body": "My dashboard shows different numbers than the export CSV. Not sure if it is a bug or I am doing something wrong. Happens every time I filter by last 30 days.",
        "customer_tier": "standard",
        "gold": {"category": "technical_issue", "priority": "medium", "team": "engineering"},
    },
    {
        "id": "M003",
        "subject": "Account",
        "body": "Hi team, my colleague left the company and I need to transfer her admin access to me. She is no longer reachable. This is blocking our monthly reporting.",
        "customer_tier": "premium",
        "gold": {"category": "account_access", "priority": "high", "team": "account_security"},
    },
    {
        "id": "M004",
        "subject": "Not happy",
        "body": "I have been a customer for 3 years and the recent price hike is ridiculous. I want to cancel and get a refund for the remaining months on my annual plan.",
        "customer_tier": "premium",
        "gold": {"category": "refund", "priority": "high", "team": "customer_success"},
    },
    {
        "id": "M005",
        "subject": "Delivery",
        "body": "Order arrived but one item is missing from the package. Invoice shows 3 items, box had 2. Need the missing item or a refund.",
        "customer_tier": "standard",
        "gold": {"category": "shipping", "priority": "medium", "team": "logistics"},
    },
    {
        "id": "M006",
        "subject": "Need help asap",
        "body": "Our team of 50 cannot access the dashboard since this morning. Tried different browsers, different networks. We are an enterprise customer and this is causing major workflow delays.",
        "customer_tier": "enterprise",
        "gold": {"category": "technical_issue", "priority": "urgent", "team": "engineering"},
    },
    {
        "id": "M007",
        "subject": "About my invoice",
        "body": "Hey, I noticed my invoice this month is much higher than usual. Did the pricing change? If not, can someone explain the extra line items?",
        "customer_tier": "standard",
        "gold": {"category": "billing", "priority": "medium", "team": "billing_team"},
    },
    {
        "id": "M008",
        "subject": "Suggestion",
        "body": "Have you considered adding integration with Slack? Would be a game changer for our notifications. Not urgent but would love to know if it is on the roadmap.",
        "customer_tier": "premium",
        "gold": {"category": "feature_request", "priority": "low", "team": "product"},
    },
    {
        "id": "M009",
        "subject": "Issue with my profile",
        "body": "I cannot update my email address. The form just spins forever after I click save. Tried logging out and back in, no luck.",
        "customer_tier": "standard",
        "gold": {"category": "technical_issue", "priority": "medium", "team": "engineering"},
    },
    {
        "id": "M010",
        "subject": "Order status",
        "body": "Order #88421 was supposed to arrive 2 days ago. Tracking still says in transit. I need this for an event this weekend, can you escalate with the carrier?",
        "customer_tier": "premium",
        "gold": {"category": "shipping", "priority": "high", "team": "logistics"},
    },
    # Multi-intent: complains about billing AND asks about a feature.
    # Gold answer focuses on the billing complaint, which is the primary issue.
    {
        "id": "M011",
        "subject": "Few things",
        "body": "First, I was charged for a feature I never enabled - please refund $40. Also, totally unrelated, when will you add 2FA via authenticator app? Thanks.",
        "customer_tier": "premium",
        "gold": {"category": "billing", "priority": "high", "team": "billing_team"},
    },
    {
        "id": "M012",
        "subject": "Confused customer",
        "body": "I am not sure if this is the right channel but my last 3 emails to support have gone unanswered. Just wanted someone to actually read this and respond. The original issue was about my dashboard not loading.",
        "customer_tier": "standard",
        "gold": {"category": "technical_issue", "priority": "high", "team": "engineering"},
    },
    {
        "id": "M013",
        "subject": "Refund question",
        "body": "Hi, my trial just ended and I was auto-charged for the annual plan. I had set a reminder to cancel but forgot. Can you make an exception and refund? I will not use the service.",
        "customer_tier": "standard",
        "gold": {"category": "refund", "priority": "medium", "team": "customer_success"},
    },
    {
        "id": "M014",
        "subject": "Help understanding",
        "body": "Can someone explain how the API rate limits work? I am hitting 429 errors on production and the docs are not super clear. We process about 10k requests per minute on average.",
        "customer_tier": "premium",
        "gold": {"category": "general_inquiry", "priority": "high", "team": "customer_success"},
    },
    {
        "id": "M015",
        "subject": "Login is weird",
        "body": "Sometimes I am logged in, sometimes I am redirected to the login page randomly. Happens across devices. Annoying but not blocking work for now.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "medium", "team": "account_security"},
    },
]


# ---------------------------------------------------------------------------
# HARD TASK: 15 tickets
# Adversarial - cry-wolf urgency, sarcasm, subject-vs-body mismatch,
# code snippets, deceptive cues. Designed so even strong frontier models
# struggle to exceed ~0.80 average reward.
# ---------------------------------------------------------------------------
HARD_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "H001",
        "subject": "URGENT: production down + billing question",
        "body": "Our entire team cannot access the API since 2am. We are on the enterprise plan. Also, while I have you, can you confirm our next invoice date? This outage is costing us money every minute.",
        "customer_tier": "enterprise",
        "gold": {"category": "technical_issue", "priority": "urgent", "team": "engineering"},
    },
    # Sarcastic: appears positive but is actually a complaint demanding a refund.
    {
        "id": "H002",
        "subject": "Just wanted to say thanks",
        "body": "Wow, what an experience. Spent my whole afternoon trying to figure out why my reports were wrong, only to discover it was a 'feature' you shipped silently last week. Truly amazing. I want my money back for this month, this is unacceptable.",
        "customer_tier": "premium",
        "gold": {"category": "refund", "priority": "high", "team": "customer_success"},
    },
    {
        "id": "H003",
        "subject": "Suspicious activity on my account",
        "body": "I got an email saying my password was changed but I did not change it. I can still log in for now. My payment method also shows a charge I do not recognize from 3 days ago.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "urgent", "team": "account_security"},
    },
    # Cry-wolf: subject says "URGENT" but content is a low-priority feature request.
    {
        "id": "H004",
        "subject": "URGENT URGENT URGENT",
        "body": "Please please please add a way to change the font color in the editor. I have been waiting forever for this. Truly the most important thing right now for my workflow.",
        "customer_tier": "standard",
        "gold": {"category": "feature_request", "priority": "low", "team": "product"},
    },
    # Subject hides severity, body reveals critical security issue.
    {
        "id": "H005",
        "subject": "Quick question about API",
        "body": "Hey, noticed my API key seems to be returning data for OTHER customers' accounts when I query certain endpoints. Probably my mistake but wanted to flag it. Here is one example response: {user_id: 8821, email: 'someone-else@gmail.com'}.",
        "customer_tier": "premium",
        "gold": {"category": "technical_issue", "priority": "urgent", "team": "engineering"},
    },
    # Stack trace embedded; technical but also has billing complaint.
    # Gold focuses on the production-blocking bug.
    {
        "id": "H006",
        "subject": "Re: Re: Re: bug not fixed",
        "body": "Still seeing this on production:\n\nTraceback (most recent call last):\n  File 'app.py', line 42, in process\n    result = sdk.compute(payload)\n  TimeoutError: deadline exceeded after 30s\n\nThis has been broken for 3 weeks. Also you charged me twice this month but the bug is more important right now.",
        "customer_tier": "enterprise",
        "gold": {"category": "technical_issue", "priority": "urgent", "team": "engineering"},
    },
    # Passive-aggressive but the actual ask is a feature request.
    {
        "id": "H007",
        "subject": "I guess this is fine",
        "body": "So apparently every other tool in this category has bulk operations except yours. Cool. Anyway, just wanted to formally request bulk delete since asking nicely in the forum did nothing for 6 months.",
        "customer_tier": "premium",
        "gold": {"category": "feature_request", "priority": "medium", "team": "product"},
    },
    # Customer-tier-aware: enterprise customer mentioning churn = high priority.
    {
        "id": "H008",
        "subject": "Renewal coming up",
        "body": "Our 3-year contract is up next month. The new pricing you sent is 40% higher with no new features. We are evaluating competitors. Need someone from your team to call us this week or we are gone.",
        "customer_tier": "enterprise",
        "gold": {"category": "general_inquiry", "priority": "urgent", "team": "customer_success"},
    },
    # Ambiguous: looks like shipping but actually billing dispute.
    {
        "id": "H009",
        "subject": "Order #99821",
        "body": "I see this order in my account but I never placed it. It is showing as shipped to an address that is not mine. Card was charged $340. I need this reversed immediately and an investigation.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "urgent", "team": "account_security"},
    },
    # Looks low priority due to tone, actually a security issue.
    {
        "id": "H010",
        "subject": "Minor thing maybe",
        "body": "Probably nothing but I just got a Slack notification for a workspace I left 2 years ago. Your system still has me as a member apparently. Felt weird, thought I would mention it.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "medium", "team": "account_security"},
    },
    # Multi-language fragment, real intent is billing.
    {
        "id": "H011",
        "subject": "Problema con la cuenta",
        "body": "Hola, me cobraron dos veces este mes. I see two charges on my statement of $59 each. Por favor refund one of them. Gracias.",
        "customer_tier": "standard",
        "gold": {"category": "billing", "priority": "high", "team": "billing_team"},
    },
    # Looks like feature request but is actually about a broken feature.
    {
        "id": "H012",
        "subject": "Idea for improvement",
        "body": "How about making the search bar actually return results? Currently it returns 0 results for queries that definitely have matches. Started after the 5.2 release. Idea: fix it.",
        "customer_tier": "premium",
        "gold": {"category": "technical_issue", "priority": "high", "team": "engineering"},
    },
    # Polite framing, actually a critical shipping escalation.
    {
        "id": "H013",
        "subject": "Whenever you have a moment",
        "body": "No rush at all but my surgical equipment shipment was supposed to arrive Monday for a procedure scheduled Wednesday. It is now Tuesday evening. The hospital needs this item or we have to reschedule.",
        "customer_tier": "enterprise",
        "gold": {"category": "shipping", "priority": "urgent", "team": "logistics"},
    },
    # General inquiry that looks like complaint.
    {
        "id": "H014",
        "subject": "Why is this so hard",
        "body": "Honestly just trying to figure out where to find the option to change my notification frequency. Spent 20 minutes clicking around. Help?",
        "customer_tier": "standard",
        "gold": {"category": "general_inquiry", "priority": "low", "team": "customer_success"},
    },
    # Refund disguised as technical issue.
    {
        "id": "H015",
        "subject": "Bug in checkout flow",
        "body": "I tried to cancel my subscription 5 times and each time the page errored. Now I am still being charged. I want a refund for the last 2 months and my subscription cancelled effective immediately.",
        "customer_tier": "premium",
        "gold": {"category": "refund", "priority": "high", "team": "customer_success"},
    },
]


TASKS = {
    "easy": EASY_TICKETS,
    "medium": MEDIUM_TICKETS,
    "hard": HARD_TICKETS,
}
