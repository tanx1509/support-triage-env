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
    # NUCLEAR: false-flag urgency - sounds critical but is actually a polite
    # nudge from a happy customer asking about an old, already-resolved bug.
    {
        "id": "H001",
        "subject": "Re: incident from last month",
        "body": "Hey team, hope you are well! Just following up on that database hiccup from 3 weeks ago - your engineer Sara already fixed it and sent us a postmortem, which was great. I am writing because our compliance officer needs the postmortem PDF in a specific format for our SOC2 audit. Could someone resend it as PDF instead of the Notion link? No rush, just for our records. Thanks!",
        "customer_tier": "enterprise",
        "gold": {"category": "general_inquiry", "priority": "low", "team": "customer_success"},
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
    # Brutal: looks like a polite product feedback, actually a paying customer
    # whose 2FA is broken (security + billing impact). Tone hides the severity.
    {
        "id": "H004",
        "subject": "Some thoughts on your auth flow",
        "body": "Hey team, love the product overall. Wanted to share some constructive feedback - the 2FA implementation seems to silently fail when you use authenticator apps with time drift. I have been locked out of my admin account for 2 days now, missed our compliance audit deadline. Anyway just wanted to flag in case it helps. We are on the enterprise plan if that matters.",
        "customer_tier": "enterprise",
        "gold": {"category": "account_access", "priority": "urgent", "team": "account_security"},
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
    # NUCLEAR: looks like simple feedback. Body buried with GDPR right-to-erasure
    # request which is legally time-bound (30 days, fines if missed). Security/legal urgency.
    {
        "id": "H006",
        "subject": "Some quick feedback on your onboarding",
        "body": "Hi! Loved trying your product over the last 2 weeks. The onboarding could be smoother (took me 4 attempts to verify my email). Anyway, I have decided not to continue, so under GDPR Article 17 I am formally requesting deletion of all my personal data and any derived data within the 30-day statutory window. My account email is on file. Please confirm completion in writing for my records. Thanks!",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "high", "team": "account_security"},
    },
    # Brutal: ticket TELLS you the wrong category in the body to test if
    # the agent thinks for itself. Actual issue is a refund (canceled by mistake).
    {
        "id": "H007",
        "subject": "Technical bug to report",
        "body": "Hi support team, I want to report this as a TECHNICAL ISSUE so it gets routed to engineering quickly. Here is what happened: I clicked the cancel button on my subscription thinking it was the pause button. Now I cannot get back in and I have already been charged for next month. Please reverse the charge and reactivate my account. The bug is that the cancel button looks like the pause button.",
        "customer_tier": "premium",
        "gold": {"category": "refund", "priority": "high", "team": "customer_success"},
    },
    # Customer-tier-aware: enterprise customer mentioning churn = high priority.
    {
        "id": "H008",
        "subject": "Renewal coming up",
        "body": "Our 3-year contract is up next month. The new pricing you sent is 40% higher with no new features. We are evaluating competitors. Need someone from your team to call us this week or we are gone.",
        "customer_tier": "enterprise",
        "gold": {"category": "general_inquiry", "priority": "urgent", "team": "customer_success"},
    },
    {
        "id": "H009",
        "subject": "Order #99821",
        "body": "I see this order in my account but I never placed it. It is showing as shipped to an address that is not mine. Card was charged $340. I need this reversed immediately and an investigation.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "urgent", "team": "account_security"},
    },
    # Brutal: customer asks "is this a bug" but actually describes account
    # takeover. Tone is curious, not alarmed. Contains a stack trace red herring.
    {
        "id": "H010",
        "subject": "Is this a bug?",
        "body": "Quick one - I just got 14 password reset emails in the last 5 minutes. None of them were initiated by me. Login still works fine. Saw this in browser console:\n\nTypeError: Cannot read property uid of undefined\n  at handleAuth (auth.js:142)\n\nProbably unrelated. Should I be worried?",
        "customer_tier": "premium",
        "gold": {"category": "account_access", "priority": "urgent", "team": "account_security"},
    },
    # Brutal: appears to be a billing complaint but is actually a phishing
    # report. The customer never made the charge - someone else did. Security.
    {
        "id": "H011",
        "subject": "Charge I do not recognize",
        "body": "Hi, I see a charge of $299 on my card from your company. I have an account with you but I am on the FREE plan and never upgraded. I checked my account and it still shows free tier. The charge happened while I was on a flight with my phone in airplane mode. Please investigate and refund. Also worried someone has my card details.",
        "customer_tier": "standard",
        "gold": {"category": "account_access", "priority": "urgent", "team": "account_security"},
    },
    # Looks like feature request but is actually about a broken feature.
    {
        "id": "H012",
        "subject": "Idea for improvement",
        "body": "How about making the search bar actually return results? Currently it returns 0 results for queries that definitely have matches. Started after the 5.2 release. Idea: fix it.",
        "customer_tier": "premium",
        "gold": {"category": "technical_issue", "priority": "high", "team": "engineering"},
    },
    # NUCLEAR: looks like a casual team-internal note. Body buried with the fact
    # that another company's data is appearing in this customer's exports - massive
    # multi-tenant data leak. Reads like routine support but is a P0 security incident.
    {
        "id": "H013",
        "subject": "Small thing - export filename typo",
        "body": "Heads up team, when I export my dashboard data the filename has an underscore where I think a hyphen would look nicer (export_2026.csv vs export-2026.csv). Minor nit. Oh also unrelated, the export this morning had ~200 rows from what looks like a different company at the bottom (saw user emails with @acme.com domain, we are @ourcompany.com). Probably just a UI bug? Not blocking, file looks fine otherwise.",
        "customer_tier": "premium",
        "gold": {"category": "technical_issue", "priority": "urgent", "team": "engineering"},
    },
    # Brutal: looks like rage about features but is actually a churn-risk
    # enterprise customer asking for a sales/CSM call disguised as feature complaint.
    {
        "id": "H014",
        "subject": "This is not working for us",
        "body": "We have been using your platform for 18 months. Honestly the last 3 months have been increasingly frustrating - features we asked for never ship, our usage is hitting limits we did not know existed, and our team is openly comparing alternatives. Our renewal is in 6 weeks. I do not want to write a long ticket explaining everything. Can someone senior actually call us this week? We are an enterprise account and we need a real conversation, not a support ticket.",
        "customer_tier": "enterprise",
        "gold": {"category": "general_inquiry", "priority": "urgent", "team": "customer_success"},
    },
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
