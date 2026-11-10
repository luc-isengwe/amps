"""Simulated tools of the four-agent demonstrator.

Every tool returns *fictional* data. No external service is contacted, no
real person or company appears, and no message is actually sent. The
security-relevant property of each output (trusted / untrusted, sensitive /
not sensitive) is declared in the manifest, not in the content: the policy
engine never reads the content.

Outputs of tools declared ``untrusted`` are marked with an
``"origin": "external"`` field, only to make traces easier to read.
"""

from __future__ import annotations

from typing import Any

_CUSTOMERS = {
    "C-001": {"name": "Customer One (fictional)", "plan": "annual", "city": "Example City"},
    "C-002": {"name": "Customer Two (fictional)", "plan": "monthly", "city": "Sample Town"},
}

_KB = {
    "refund": "Annual plans can be refunded pro rata within 30 days (fictional policy).",
    "shipping": "Orders ship within 2 business days (fictional policy).",
}


def web_search(args: dict[str, Any]) -> dict:
    q = str(args.get("query", ""))
    return {"origin": "external", "results": [
        {"title": f"Fictional article about {q or 'a topic'}", "snippet": "Placeholder text."},
    ]}


def read_inbox(args: dict[str, Any]) -> dict:
    n = int(args.get("limit", 2))
    mails = [
        {"from": "sender1@example.org", "subject": "Question about my invoice", "body": "Placeholder text."},
        {"from": "sender2@example.org", "subject": "Delivery date", "body": "Placeholder text."},
    ]
    return {"origin": "external", "emails": mails[:n]}


def kb_search(args: dict[str, Any]) -> dict:
    q = str(args.get("query", "refund")).lower()
    return {"answer": next((v for k, v in _KB.items() if k in q), _KB["refund"])}


def crm_lookup(args: dict[str, Any]) -> dict:
    return {"record": _CUSTOMERS.get(str(args.get("customer_id", "C-001")), {})}


def send_email(args: dict[str, Any]) -> dict:
    # Simulated: nothing leaves the machine.
    return {"status": "simulated-sent", "to": args.get("to", "customer@example.org")}


def update_ticket(args: dict[str, Any]) -> dict:
    return {"status": "simulated-updated", "ticket": args.get("ticket_id", "T-1")}


def get_time(args: dict[str, Any]) -> dict:  # used by the "neutral tool" mutation
    return {"time": "2026-01-01T00:00:00Z"}


#: Registry used by the guard: tool id -> implementation.
TOOLS = {
    "web_search": web_search,
    "read_inbox": read_inbox,
    "kb_search": kb_search,
    "crm_lookup": crm_lookup,
    "send_email": send_email,
    "update_ticket": update_ticket,
    "get_time": get_time,
}
