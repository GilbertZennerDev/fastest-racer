"""Stripe Checkout + webhook wiring for the Pro subscription.

Requires these environment variables in production (unset -> billing
endpoints return a clear "not configured" error instead of crashing, so the
app still runs for local dev / demoing the free tier without a Stripe
account):
  STRIPE_SECRET_KEY       - sk_live_... / sk_test_...
  STRIPE_PRICE_ID_PRO     - price_... for the Pro monthly plan
  STRIPE_WEBHOOK_SECRET   - whsec_... from the webhook endpoint's settings
  PUBLIC_BASE_URL         - e.g. https://fastest-racer.workflowsolved.com
"""
import os

import stripe

from . import auth

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID_PRO = os.environ.get("STRIPE_PRICE_ID_PRO")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def is_configured() -> bool:
    return bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID_PRO)


def create_checkout_session(user: dict) -> str:
    """Creates a Stripe Checkout session for the Pro plan and returns its URL."""
    if not is_configured():
        raise RuntimeError("Stripe is not configured on this server (missing STRIPE_SECRET_KEY / STRIPE_PRICE_ID_PRO).")

    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        customer = stripe.Customer.create(email=user["email"])
        customer_id = customer["id"]
        auth.set_stripe_customer(user["id"], customer_id)

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID_PRO, "quantity": 1}],
        success_url=f"{PUBLIC_BASE_URL}/app/?upgraded=1",
        cancel_url=f"{PUBLIC_BASE_URL}/#pricing",
    )
    return session["url"]


def create_portal_session(user: dict) -> str:
    """Stripe's hosted billing portal, so users can cancel/update payment
    methods without us building that UI ourselves."""
    if not is_configured():
        raise RuntimeError("Stripe is not configured on this server.")
    if not user.get("stripe_customer_id"):
        raise RuntimeError("This account has no billing history yet.")
    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{PUBLIC_BASE_URL}/app/",
    )
    return portal["url"]


def handle_webhook(payload: bytes, sig_header: str):
    """Verifies and processes a Stripe webhook event, updating the user's
    tier when a subscription is created, updated, or cancelled."""
    if not STRIPE_WEBHOOK_SECRET:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET is not configured.")

    event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    obj = event["data"]["object"]
    event_type = event["type"]

    if event_type in ("customer.subscription.created", "customer.subscription.updated"):
        customer_id = obj["customer"]
        user = auth.get_user_by_stripe_customer(customer_id)
        if user:
            status = obj["status"]
            tier = "pro" if status in ("active", "trialing") else "free"
            auth.set_subscription(user["id"], tier, obj["id"])

    elif event_type == "customer.subscription.deleted":
        customer_id = obj["customer"]
        user = auth.get_user_by_stripe_customer(customer_id)
        if user:
            auth.set_subscription(user["id"], "free", None)

    return event_type
