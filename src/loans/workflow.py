"""The bank's workflow: the steps around the agents.

The workflow decides when an agent runs and does the bank's own steps, which
need no judgment and so no LLM. It reacts to events from outside (an
application arrives, the customer replies, time passes); in a real bank these
come from the website, the staff's screens and timers, and the same functions
would be called on each event. In this project the events come from the
scripted customer in loans.evals.simulate, which is only for testing.

Here the workflow is plain code. In production it would run on a durable
workflow engine (for example Temporal) or a BPM engine, which add recovery
after a crash, real timers and retries; these steps would become the steps
that engine calls.

What arrives from the customer (through the bank's website) is recorded
here, in both phases:

- receive_application: intake. The application, its documents and the
  customer's first message are stored and logged.
- receive_reply: a reply from the customer, with any message, uploads and
  form corrections, recorded together (they arrive together). During the
  offer phase a reply is usually just a message.

The agents and the bank's steps:

- open_case_for_agent: grants an agent access to one case. The tools
  themselves live on the MCP server (their code, rules and database work),
  and the agent's loop only uses them; neither decides which case an agent
  may work on. That is the workflow's job, because it is the trusted side,
  holding the database password: like a manager who assigns a case to an
  employee and gives them a badge that opens only that file. Through the MCP
  server (as deployed) it opens a session for the case, connects the client
  with the session's token and hands the agent the resulting tool
  connections, and closes the session at the end, also if the run fails.
  Directly (no server) the tools run in this program on the store.
- after_application: once the application agent has completed a case, the
  credit check (loans.services.credit_bureau) and, if it passes, the offer
  (loans.offers), which is sent at once. Cancelled cases and cases sent to a
  human go no further.

The offer phase (the bank's steps; the offer agent handles the customer's
messages in between):

- send_offer: sends the case's latest offer with its notice (the numbers
  truth-in-lending rules require, written by code). The offer can be signed
  for policy.OFFER_VALID_DAYS days from when it was sent.
- sign_offer: the customer's e-signature, from the bank's website; only the
  latest offer sent can be signed, and only while it is valid. It closes the
  offer phase as accepted.
- expire_if_due: a daily timer's job; once the latest offer's validity has
  passed without a signature, it closes the offer phase as expired.
- decide_approval: a loan officer's decision on the offer agent's pending
  request, from the bank's staff screen. Approved with an amount and a term,
  which code checks against the product's limits and prices from policy.py
  (the officer never sets a rate); the revised offer is created and sent,
  and replaces the earlier one. Or declined. Either way the decision is kept
  with the officer's ID and note.

A step that would change a closed offer phase (a signature or decision that
arrives too late) is refused: the portal or screen shows the error, and a
decision that arrives late is logged as blocked.
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from langchain_core.tools import BaseTool

from loans.agent.tools import make_tools
from loans import policy
from loans.mcp.client import load_tools
from loans.mcp.sessions import SessionClock, close_session, open_session
from loans.offers import create_offer, offer_for, offer_notice
from loans.schemas.application import (
    ALLOWED_TERMS,
    MAX_LOAN_AMOUNT,
    MIN_LOAN_AMOUNT,
    RawApplication,
    RawFile,
)
from loans.schemas.documents import UploadedDocument, UploadSlot
from loans.services.credit_bureau import check_credit
from loans.store import (
    ApplicationStore,
    CustomerReply,
    OfferClosing,
    OfferSent,
    Signature,
    StoredApplication,
)

Clock = Callable[[], datetime]


def receive_application(
    store: ApplicationStore,
    raw: RawApplication,
    documents: dict[str, UploadedDocument],
    now: datetime,
    application_id: str | None = None,
    first_message: str | None = None,
) -> str:
    """Intake: store a new application with its documents and the customer's
    first message, and return its ID."""
    return store.submit(raw, documents, now, application_id=application_id, first_message=first_message)


def receive_reply(
    store: ApplicationStore,
    application_id: str,
    now: datetime,
    message: str | None = None,
    uploads: dict[UploadSlot, tuple[UploadedDocument, RawFile]] | None = None,
    changes: dict | None = None,
) -> None:
    """A reply from the customer: the message, uploads (each slot's document
    and file) and form corrections, recorded together at the same time."""
    uploads = uploads or {}
    changes = changes or {}
    store.record_reply(
        application_id,
        CustomerReply(
            received_at=now, message=message, uploads=list(uploads), corrected_fields=list(changes)
        ),
    )
    for slot, (document, file) in uploads.items():
        store.upload(application_id, slot, document, file, now=now)
    if changes:
        store.correct_fields(application_id, changes, now=now)


@asynccontextmanager
async def open_case_for_agent(
    store: ApplicationStore,
    application_id: str,
    start: datetime,
    mcp_url: str | None,
    direct_clock: Clock,
) -> AsyncGenerator[tuple[list[BaseTool], Clock]]:
    """Access to one case for an agent, while the context is open: the tools
    it may use on that case, and the case's clock. Through the MCP server if
    mcp_url is given (the clock is then the session's, shared with the
    server), else directly with direct_clock."""
    if mcp_url is None:
        yield make_tools(store, application_id, direct_clock), direct_clock
        return
    conn = store.connection  # a PostgresStore: the server works on the same database
    token = open_session(conn, application_id, start, store.agent)
    try:
        yield await load_tools(mcp_url, token), SessionClock(conn, token)
    finally:
        close_session(conn, token)


def after_application(store: ApplicationStore, application_id: str, clock: Clock) -> None:
    """The bank's steps after the application agent ended the case: for a
    complete case, the credit check and, if it passes, the offer."""
    stored = store.get(application_id)
    if stored.ending is None or stored.ending.outcome != "complete":
        return
    result = check_credit(stored.metadata, clock())
    store.record_credit_check(application_id, result)
    if result.passed:
        store.record_offer(application_id, offer_for(stored.raw), clock())
        send_offer(store, application_id, clock())


class OfferPhaseError(ValueError):
    """A step the offer phase does not allow, for example signing an offer
    that was not the latest sent, or after the phase closed."""


def _latest_sent(stored: StoredApplication) -> OfferSent | None:
    return stored.offers_sent[-1] if stored.offers_sent else None


def valid_until(sent: OfferSent) -> datetime:
    """When the offer stops being valid."""
    return sent.sent_at + timedelta(days=policy.OFFER_VALID_DAYS)


def _check_open(stored: StoredApplication) -> None:
    if stored.offer_closing is not None:
        raise OfferPhaseError(f"the offer phase has closed ({stored.offer_closing.outcome})")


def send_offer(store: ApplicationStore, application_id: str, now: datetime) -> None:
    """Send the case's latest offer, with its notice."""
    stored = store.get(application_id)
    _check_open(stored)
    index = len(stored.offers) - 1
    until = (now + timedelta(days=policy.OFFER_VALID_DAYS)).date()
    notice = offer_notice(stored.offers[index], until)
    store.record_offer_sent(application_id, OfferSent(offer_index=index, sent_at=now, notice=notice))


def sign_offer(store: ApplicationStore, application_id: str, offer_index: int, now: datetime) -> None:
    """The customer signs an offer: only the latest sent, while valid. Closes
    the offer phase as accepted."""
    stored = store.get(application_id)
    _check_open(stored)
    latest = _latest_sent(stored)
    if latest is None or offer_index != latest.offer_index:
        raise OfferPhaseError("only the latest offer sent can be signed")
    if now >= valid_until(latest):
        raise OfferPhaseError("the offer is no longer valid")
    store.record_signature(application_id, Signature(offer_index=offer_index, signed_at=now))


def expire_if_due(store: ApplicationStore, application_id: str, now: datetime) -> bool:
    """Close the offer phase as expired if the latest offer's validity has
    passed without a signature. True if it closed."""
    stored = store.get(application_id)
    latest = _latest_sent(stored)
    if stored.offer_closing is not None or latest is None or now < valid_until(latest):
        return False
    store.record_offer_closing(application_id, OfferClosing(outcome="expired", closed_at=now))
    return True


def decide_approval(
    store: ApplicationStore,
    application_id: str,
    officer_id: str,
    decision: Literal["approve", "decline"],
    note: str,
    now: datetime,
    amount: Decimal | None = None,
    term_months: int | None = None,
) -> None:
    """A loan officer decides the pending request. Approving needs an amount
    and a term within the product's limits; the revised offer is priced by
    code from policy.py, created and sent."""
    stored = store.get(application_id)
    if stored.offer_closing is not None:
        store.log_event(
            application_id, now, "guardrail", "action_blocked",
            f"approval decision: the offer phase has closed ({stored.offer_closing.outcome})",
        )  # fmt: skip
        raise OfferPhaseError(f"the offer phase has closed ({stored.offer_closing.outcome})")
    request = stored.pending_approval()
    if request is None:
        raise OfferPhaseError("there is no request waiting for a decision")
    if decision == "decline":
        store.record_approval_decision(application_id, request.request_id, "declined", officer_id, note, now)
        return
    if amount is None or term_months is None:
        raise OfferPhaseError("an approval needs an amount and a term")
    if not MIN_LOAN_AMOUNT <= amount <= MAX_LOAN_AMOUNT:
        raise OfferPhaseError(f"the amount must be between ${MIN_LOAN_AMOUNT:,} and ${MAX_LOAN_AMOUNT:,}")
    if term_months not in ALLOWED_TERMS:
        raise OfferPhaseError(f"the term must be one of {sorted(ALLOWED_TERMS)} months")
    store.record_approval_decision(application_id, request.request_id, "approved", officer_id, note, now)
    store.record_offer(application_id, create_offer(amount, term_months), now)
    send_offer(store, application_id, now)
