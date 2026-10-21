"""The six tools, as the LLM sees them.

make_tools wraps the plain tools in tools/application/ for LangGraph, for one case. The
store, the application ID and the clock are filled in here, and the call ID
comes from the LLM's tool call (InjectedToolCallId), so none of them is part
of the parameters the LLM sees: the LLM chooses only the slot, the text, the
reason, the outcome and the explanation. The LLM is never given the
application ID, so text in a prompt cannot point a tool at another case.

What the LLM reads about each tool comes from loans.tools.application.descriptions, shared
with the MCP server (loans.mcp.server), which offers the same tools across the
network.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from loans.schemas.documents import IncomeSlot
from loans.tools.application import descriptions
from loans.schemas.outcome import OutcomeKind, Reason
from loans.store import ApplicationStore
from loans.tools.application.check_application import check_application as _check_application
from loans.tools.application.finish import finish as _finish
from loans.tools.application.mark_income_verified import mark_income_verified as _mark_income_verified
from loans.tools.application.read_document import read_document as _read_document
from loans.tools.application.send_message import send_message as _send_message
from loans.tools.application.verify_identity import verify_identity as _verify_identity

CallId = Annotated[str, InjectedToolCallId]


def make_tools(
    store: ApplicationStore, application_id: str, clock: Callable[[], datetime]
) -> list[BaseTool]:
    """The tools for one case. clock gives the current (possibly simulated)
    time at each call."""

    @tool(description=descriptions.CHECK_APPLICATION)
    def check_application() -> str:
        return _check_application(store, application_id, clock())

    @tool(description=descriptions.READ_DOCUMENT)
    def read_document(slot: IncomeSlot) -> str:
        return _read_document(store, application_id, slot, clock())

    @tool(description=descriptions.VERIFY_IDENTITY)
    def verify_identity() -> str:
        return _verify_identity(store, application_id, clock())

    @tool(description=descriptions.MARK_INCOME_VERIFIED)
    def mark_income_verified(
        slot: IncomeSlot,
        reason: Annotated[str, descriptions.INCOME_REASON],
        call_id: CallId,
    ) -> str:
        return _mark_income_verified(
            store, application_id, clock(), call_id, slot, reason
        )

    @tool(description=descriptions.SEND_MESSAGE)
    def send_message(
        text: Annotated[str, descriptions.MESSAGE_TEXT],
        reason: Annotated[str, descriptions.MESSAGE_REASON],
        call_id: CallId,
    ) -> str:
        return _send_message(
            store, application_id, clock(), call_id, text, reason
        )

    @tool(description=descriptions.FINISH)
    def finish(
        outcome: OutcomeKind,
        reason: Reason,
        explanation: Annotated[str, descriptions.EXPLANATION],
        call_id: CallId,
    ) -> str:
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

