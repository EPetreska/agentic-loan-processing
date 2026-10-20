"""The six tools, as the LLM sees them.

make_tools wraps the plain tools in tools/ for LangGraph, for one case. The
store, the application ID and the clock are filled in here, and the call ID
comes from the LLM's tool call (InjectedToolCallId), so none of them is part
of the parameters the LLM sees: the LLM chooses only the slot, the text, the
reason, the outcome and the explanation. The LLM is never given the
application ID, so text in a prompt cannot point a tool at another case.

The descriptions here are what the LLM reads about each tool; the plain tools
keep their own docstrings for developers.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from loans.schemas.documents import IncomeSlot
from loans.schemas.outcome import OutcomeKind, Reason
from loans.store import ApplicationStore
from loans.tools.check_application import check_application as _check_application
from loans.tools.finish import finish as _finish
from loans.tools.mark_income_verified import mark_income_verified as _mark_income_verified
from loans.tools.read_document import read_document as _read_document
from loans.tools.send_message import MAX_MESSAGE_LENGTH
from loans.tools.send_message import send_message as _send_message
from loans.tools.verify_identity import verify_identity as _verify_identity

CallId = Annotated[str, InjectedToolCallId]


def make_tools(
    store: ApplicationStore, application_id: str, clock: Callable[[], datetime]
) -> list[BaseTool]:
    """The tools for one case. clock gives the current (possibly simulated)
    time at each call."""

    @tool
    def check_application() -> str:
        """Check every field of the application form. Returns one line per
        field: ok, error or warning. For fields such as the name, employer,
        income and loan, it also shows the customer's value: the checked
        value when ok (free text in quotes), and exactly what was typed, in
        quotes, when there is an error. Sensitive fields such as the SSN and
        the date of birth show only the result."""
        return _check_application(store, application_id, clock())

    @tool
    def read_document(slot: IncomeSlot) -> str:
        """Read an uploaded income document. Returns two parts. First,
        "extraction findings:", what the extraction service read, one
        "name: value" line each for: what the file really is (document_type)
        and whether it matches the slot it was uploaded in (matches_slot);
        whether it could be read (readable); the fields on the document, with
        text from the document in quotes; values calculated in code, marked
        "(calculated)", such as the yearly income and the pay stub's age; and
        any other text found on the document, in quotes. Then the bank's own
        check: "bank income rules: met", or "not met" with one line per rule
        that is not met."""
        return _read_document(store, application_id, slot, clock())

    @tool
    def verify_identity() -> str:
        """Compare the uploaded ID with the form. Returns one line per
        finding: yes, no, or "cannot check" with the reason. The findings are
        facts, not a decision. A "no" never says how the values differ,
        because personal data is never shown. After a new ID upload or a form
        correction, verify again: earlier findings are then out of date."""
        return _verify_identity(store, application_id, clock())

    @tool
    def mark_income_verified(
        slot: IncomeSlot,
        reason: Annotated[str, "Why this document verifies the claimed income."],
        call_id: CallId,
    ) -> str:
        """Record that the document in this slot verifies the claimed income.
        Refused if a bank income rule is not met. After a new upload of this
        document or a form correction, mark it again: the earlier
        verification is then out of date."""
        return _mark_income_verified(
            store, application_id, clock(), call_id, slot, reason
        )

    # The description is built from MAX_MESSAGE_LENGTH, so it cannot drift
    # from the real limit; a docstring cannot hold a variable.
    @tool(
        description=(
            "Send a message to the customer. Refused if the customer has not "
            "given e-sign consent, if the text is empty or longer than "
            f"{MAX_MESSAGE_LENGTH} characters, or without a reason."
        )
    )
    def send_message(
        text: Annotated[str, "The message to the customer."],
        reason: Annotated[str, "Why you are sending it."],
        call_id: CallId,
    ) -> str:
        return _send_message(
            store, application_id, clock(), call_id, text, reason
        )

    @tool
    def finish(
        outcome: OutcomeKind,
        reason: Reason,
        explanation: Annotated[str, "A short explanation for whoever takes the case next."],
        call_id: CallId,
    ) -> str:
        """End the case with an outcome, a reason that fits it, and an
        explanation. Refused if the bank's rules do not allow this ending,
        listing every problem. A case ends only once."""
        return _finish(
            store, application_id, clock(), call_id, outcome, reason, explanation
        )

    return [
        check_application,
        read_document,
        verify_identity,
        mark_income_verified,
        send_message,
        finish,
    ]

