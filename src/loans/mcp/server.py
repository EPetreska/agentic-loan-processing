"""The MCP server: the bank's application system, offered to AI agents.

One server for every case, with one endpoint per agent: the application
agent's six tools at /mcp/application and the offer agent's three at
/mcp/offer, over HTTP (MCP's streamable HTTP transport), run on the cases in
Postgres. Start it with:

    uv run python -m loans.mcp.server

It listens on 127.0.0.1:8000, reachable from this laptop only.

Each agent sees and can call only its own tools (least privilege): the
endpoint offers nothing else, and a session's token works only at the
endpoint of the agent it was opened for (loans.mcp.sessions). Separate
endpoints rather than one list filtered per caller, because the two agents
do different jobs, each endpoint then enforces its tools by construction,
and both have a send_message with different rules.

Each call is checked before any tool runs:

- The token. Every call carries a session token in the standard
  "Authorization: Bearer <token>" header (see loans.mcp.sessions). A missing,
  wrong or closed token, or one opened for the other agent, is refused with
  HTTP 401 and nothing is stored: storing what an untrusted caller sends
  would let an attacker fill the database. The case comes from the token,
  never from a tool argument, so no tool can be pointed at another case.
- The call ID. The tools that change the case need the ID of the LLM's tool
  call, so a retried call is not done twice. It travels in the request's
  _meta field (under CALL_ID_KEY), MCP's place for extra information about
  one call, so it is never a tool argument and the LLM never sees it.

The tools then run the same code as the direct tools (loans.tools.application
and loans.tools.offer), with the same descriptions, on a PostgresStore that
adds the session's model and prompt version to the audit log. The time comes
from the session's clock. Each call borrows its own database connection from
a pool, so calls at the same time cannot mix their transactions. The offer
agent's document search (loans.rag.search) runs here too, on the call's
connection; it needs Ollama for the query's embedding.

The SDK's token check asks for an "issuer": the OAuth server that hands out
tokens. Here the bank's workflow hands them out (it opens sessions in the
database), so the issuer is this server's own address; real OAuth login
comes with the loan officer's assistant.
"""

from collections.abc import Callable, Generator
from contextlib import AsyncExitStack, contextmanager
from typing import Annotated, NamedTuple

import anyio
import uvicorn
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
import psycopg
from psycopg_pool import ConnectionPool
from pydantic import Field
from starlette.applications import Starlette

from loans.db.postgres_store import PostgresStore, create_tables, database_conninfo
from loans.mcp import CALL_ID_KEY
from loans.mcp.sessions import AgentName, InvalidToken, SessionClock, find_session
from loans.rag.search import SearchResult
from loans.rag.search import search as rag_search
from loans.schemas.documents import IncomeSlot
from loans.schemas.outcome import OutcomeKind, Reason
from loans.tools.application import descriptions
from loans.tools.offer import descriptions as offer_descriptions
from loans.tools.application.check_application import check_application as _check_application
from loans.tools.application.finish import finish as _finish
from loans.tools.application.mark_income_verified import mark_income_verified as _mark_income_verified
from loans.tools.application.read_document import read_document as _read_document
from loans.tools.application.send_message import send_message as _send_message
from loans.tools.application.verify_identity import verify_identity as _verify_identity
from loans.tools.offer.request_approval import request_approval as _request_approval
from loans.tools.offer.retrieve_documents import retrieve_documents as _retrieve_documents
from loans.tools.offer.send_message import send_message as _offer_send_message

HOST = "127.0.0.1"
PORT = 8000
# The endpoints are at /mcp/<agent>.
MCP_PATH = "/mcp"

# The search the offer agent's retrieve_documents uses on the server: on the
# call's database connection. Tests pass a stand-in.
ServerSearch = Callable[[psycopg.Connection, str], list[SearchResult]]

NOT_ALLOWED = "not allowed: unknown or closed session"


class SessionTokens:
    """Checks each call's token against the open sessions of one agent (the
    SDK's TokenVerifier): a token opened for another agent is refused."""

    def __init__(self, pool: ConnectionPool, agent_name: AgentName) -> None:
        self._pool = pool
        self._agent_name = agent_name

    def _find(self, token: str):
        with self._pool.connection() as conn:
            return find_session(conn, token)

    async def verify_token(self, token: str) -> AccessToken | None:
        # The database call blocks, so it runs off the server's event loop.
        session = await anyio.to_thread.run_sync(self._find, token)
        if session is None or session.agent_name != self._agent_name:
            return None
        return AccessToken(token=token, client_id="case-session", scopes=[])


class Case(NamedTuple):
    """What a tool needs for the call's case."""

    store: PostgresStore
    application_id: str
    clock: SessionClock


def _new_server(pool: ConnectionPool, url: str, agent_name: AgentName, instructions: str) -> MCPServer:
    """An empty server for one agent's endpoint, with its token check."""
    return MCPServer(
        name=f"loans-{agent_name}",
        instructions=instructions,
        token_verifier=SessionTokens(pool, agent_name),
        auth=AuthSettings(
            issuer_url=url,
            resource_server_url=f"{url}{MCP_PATH}/{agent_name}",
            # The SDK can also check that a token was issued for this server
            # (its "resource"). Session tokens exist only in this server's
            # database, so a token for another server cannot pass anyway.
            validate_token_resource=False,
        ),
    )


def _case_opener(pool: ConnectionPool, agent_name: AgentName):
    """The function the tools of one endpoint use to reach the call's case."""

    @contextmanager
    def case() -> Generator[Case]:
        """The call's case, found from its token, with a connection of its
        own for the length of the call."""
        access = get_access_token()
        if access is None:  # the SDK refuses such calls first; this is a second lock
            raise ToolError(NOT_ALLOWED)
        with pool.connection() as conn:
            # Checked again: the session may have been closed since the token
            # was checked at the door, and it must be this agent's.
            session = find_session(conn, access.token)
            if session is None or session.agent_name != agent_name:
                raise ToolError(NOT_ALLOWED)
            store = PostgresStore(conn, agent=session.agent)
            try:
                yield Case(store, session.application_id, SessionClock(conn, access.token))
            except InvalidToken:
                raise ToolError(NOT_ALLOWED) from None

    return case


def _call_id(ctx: Context) -> str:
    meta = ctx.request_context.meta or {}
    value = meta.get(CALL_ID_KEY)
    if not isinstance(value, str) or not value:
        raise ToolError(f"not done: the request has no call ID (_meta {CALL_ID_KEY!r})")
    return value


def create_application_server(pool: ConnectionPool, url: str = f"http://{HOST}:{PORT}") -> MCPServer:
    """The application agent's endpoint, on the cases reached through pool.
    url is where the server is reached, without a path."""
    server = _new_server(
        pool, url, "application",
        "The bank's loan application system: actions on one application.",
    )  # fmt: skip
    case = _case_opener(pool, "application")

    @server.tool(description=descriptions.CHECK_APPLICATION)
    def check_application() -> str:
        with case() as c:
            return _check_application(c.store, c.application_id, c.clock())

    @server.tool(description=descriptions.READ_DOCUMENT)
    def read_document(slot: IncomeSlot) -> str:
        with case() as c:
            return _read_document(c.store, c.application_id, slot, c.clock())

    @server.tool(description=descriptions.VERIFY_IDENTITY)
    def verify_identity() -> str:
        with case() as c:
            return _verify_identity(c.store, c.application_id, c.clock())

    @server.tool(description=descriptions.MARK_INCOME_VERIFIED)
    def mark_income_verified(
        slot: IncomeSlot,
        reason: Annotated[str, Field(description=descriptions.INCOME_REASON)],
        ctx: Context,
    ) -> str:
        call = _call_id(ctx)
        with case() as c:
            return _mark_income_verified(c.store, c.application_id, c.clock(), call, slot, reason)

    @server.tool(description=descriptions.SEND_MESSAGE)
    def send_message(
        text: Annotated[str, Field(description=descriptions.MESSAGE_TEXT)],
        reason: Annotated[str, Field(description=descriptions.MESSAGE_REASON)],
        ctx: Context,
    ) -> str:
        call = _call_id(ctx)
        with case() as c:
            return _send_message(c.store, c.application_id, c.clock(), call, text, reason)

    @server.tool(description=descriptions.FINISH)
    def finish(
        outcome: OutcomeKind,
        reason: Reason,
        explanation: Annotated[str, Field(description=descriptions.EXPLANATION)],
        ctx: Context,
    ) -> str:
        call = _call_id(ctx)
        with case() as c:
            return _finish(
                c.store, c.application_id, c.clock(), call, outcome, reason, explanation
            )

    return server


def create_offer_server(
    pool: ConnectionPool, url: str = f"http://{HOST}:{PORT}", search: ServerSearch = rag_search
) -> MCPServer:
    """The offer agent's endpoint, on the cases reached through pool, with
    the document search run on each call's connection."""
    server = _new_server(
        pool, url, "offer",
        "The bank's loan offers: answering one customer about their offer.",
    )  # fmt: skip
    case = _case_opener(pool, "offer")
    d = offer_descriptions

    @server.tool(description=d.RETRIEVE_DOCUMENTS)
    def retrieve_documents(query: Annotated[str, Field(description=d.QUERY)], ctx: Context) -> str:
        call = _call_id(ctx)
        with case() as c:
            conn = c.store.connection
            return _retrieve_documents(
                c.store, c.application_id, c.clock(), call, query, lambda q: search(conn, q)
            )

    @server.tool(description=d.REQUEST_APPROVAL)
    def request_approval(
        request: Annotated[str, Field(description=d.REQUEST)],
        reason: Annotated[str, Field(description=d.REQUEST_REASON)],
        ctx: Context,
    ) -> str:
        call = _call_id(ctx)
        with case() as c:
            return _request_approval(c.store, c.application_id, c.clock(), call, request, reason)

    @server.tool(description=d.SEND_MESSAGE)
    def send_message(
        text: Annotated[str, Field(description=d.MESSAGE_TEXT)],
        reason: Annotated[str, Field(description=d.MESSAGE_REASON)],
        ctx: Context,
    ) -> str:
        call = _call_id(ctx)
        with case() as c:
            return _offer_send_message(c.store, c.application_id, c.clock(), call, text, reason)

    return server


class Endpoints:
    """One web app for both endpoints: each request goes to the endpoint its
    path names (/mcp/offer, and its metadata, to the offer endpoint; the rest
    to the application endpoint), and starting or stopping the app starts or
    stops both."""

    def __init__(self, application: Starlette, offer: Starlette) -> None:
        self._application = application
        self._offer = offer

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        is_offer = f"{MCP_PATH}/offer" in scope.get("path", "")
        await (self._offer if is_offer else self._application)(scope, receive, send)

    async def _lifespan(self, receive, send) -> None:
        async with AsyncExitStack() as stack:
            await receive()  # startup
            for app in (self._application, self._offer):
                await stack.enter_async_context(app.router.lifespan_context(app))
            await send({"type": "lifespan.startup.complete"})
            await receive()  # shutdown
        await send({"type": "lifespan.shutdown.complete"})


def create_app(
    pool: ConnectionPool, url: str = f"http://{HOST}:{PORT}", search: ServerSearch = rag_search
) -> Endpoints:
    """The web app with both endpoints. Each call stands on its own
    (stateless): the token on every call is all the server needs."""

    def http_app(server: MCPServer, agent_name: AgentName) -> Starlette:
        return server.streamable_http_app(
            streamable_http_path=f"{MCP_PATH}/{agent_name}",
            stateless_http=True,
            json_response=True,
            host=HOST,
        )

    return Endpoints(
        http_app(create_application_server(pool, url), "application"),
        http_app(create_offer_server(pool, url, search), "offer"),
    )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    url = f"http://{HOST}:{PORT}"
    with ConnectionPool(
        database_conninfo(), kwargs={"autocommit": True}, min_size=1, max_size=5
    ) as pool:
        with pool.connection() as conn:
            create_tables(conn)  # only what is missing, so safe at every start
        uvicorn.run(create_app(pool, url), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
