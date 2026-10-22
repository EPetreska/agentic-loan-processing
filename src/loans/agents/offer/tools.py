"""The offer agent's three tools, as the LLM sees them.

make_offer_tools wraps the plain tools in tools/offer/ for LangGraph, for one
case, as loans.agents.application.tools does for the application agent. The
store, the application ID, the clock and the search are filled in here, and
the call ID comes from the LLM's tool call (InjectedToolCallId), so none of
them is part of the parameters the LLM sees: it chooses only the query, the
request and the message. The LLM is never given the application ID, so text
in a prompt cannot point a tool at another case.

The search is handed in ready to use (a function from a query to the closest
sections), built by whoever sets the agent up from a database connection:
the workflow in a direct run, the MCP server on each call. The tool itself
never opens a connection, and tests can pass a stand-in.

What the LLM reads about each tool comes from loans.tools.offer.descriptions.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from loans.rag.search import SearchResult
from loans.store import ApplicationStore
from loans.tools.offer import descriptions
from loans.tools.offer.request_approval import request_approval as _request_approval
from loans.tools.offer.retrieve_documents import retrieve_documents as _retrieve_documents
from loans.tools.offer.send_message import send_message as _send_message

CallId = Annotated[str, InjectedToolCallId]


def make_offer_tools(
    store: ApplicationStore,
    application_id: str,
    clock: Callable[[], datetime],
    search: Callable[[str], list[SearchResult]],
) -> list[BaseTool]:
    """The offer agent's tools for one case. clock gives the current (possibly
    simulated) time at each call; search finds the closest sections."""

    @tool(description=descriptions.RETRIEVE_DOCUMENTS)
    def retrieve_documents(query: Annotated[str, descriptions.QUERY], call_id: CallId) -> str:
        return _retrieve_documents(store, application_id, clock(), call_id, query, search)

    @tool(description=descriptions.REQUEST_APPROVAL)
    def request_approval(
        request: Annotated[str, descriptions.REQUEST],
        reason: Annotated[str, descriptions.REQUEST_REASON],
        call_id: CallId,
    ) -> str:
        return _request_approval(store, application_id, clock(), call_id, request, reason)

    @tool(description=descriptions.SEND_MESSAGE)
    def send_message(
        text: Annotated[str, descriptions.MESSAGE_TEXT],
        reason: Annotated[str, descriptions.MESSAGE_REASON],
        call_id: CallId,
    ) -> str:
        return _send_message(store, application_id, clock(), call_id, text, reason)

    return [retrieve_documents, request_approval, send_message]
