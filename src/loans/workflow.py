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
  (loans.offers). Cancelled cases and cases sent to a human go no further.
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime

from langchain_core.tools import BaseTool

from loans.agent.tools import make_tools
from loans.mcp.client import load_tools
from loans.mcp.sessions import SessionClock, close_session, open_session
from loans.offers import offer_for
from loans.services.credit_bureau import check_credit
from loans.store import ApplicationStore

Clock = Callable[[], datetime]


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
