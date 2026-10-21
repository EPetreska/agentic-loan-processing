"""The MCP server: the bank's application system, offered to AI agents.

One server for every case. It offers the agent's six tools over HTTP (MCP's
streamable HTTP transport) and runs them on the cases in Postgres. Start it
with:

    uv run python -m loans.mcp.server

It listens on 127.0.0.1:8000/mcp, reachable from this laptop only.

Each call is checked before any tool runs:

- The token. Every call carries a session token in the standard
  "Authorization: Bearer <token>" header (see loans.mcp.sessions). A missing,
  wrong or closed token is refused with HTTP 401 and nothing is stored: it
  points to no case, and storing what an untrusted caller sends would let an
  attacker fill the database. The case comes from the token, never from a tool
  argument, so no tool can be pointed at another case.
- The call ID. send_message, mark_income_verified and finish need the ID of
  the LLM's tool call, so a retried call is not done twice. It travels in the
  request's _meta field (under CALL_ID_KEY), MCP's place for extra
  information about one call, so it is never a tool argument and the LLM never
  sees it.

The tools then run the same code as the direct tools (loans.tools), with the
same descriptions (loans.tools.descriptions), on a PostgresStore that adds the
session's model and prompt version to the audit log. The time comes from the
session's clock. Each call borrows its own database connection from a pool,
so calls at the same time cannot mix their transactions.

The SDK's token check asks for an "issuer": the OAuth server that hands out
tokens. Here the bank's workflow hands them out (it opens sessions in the
database), so the issuer is this server's own address; real OAuth login
comes with the loan officer's assistant.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Annotated, NamedTuple

import anyio
import uvicorn
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from psycopg_pool import ConnectionPool
from pydantic import Field

from loans.db.postgres_store import PostgresStore, create_tables, database_conninfo
from loans.mcp.sessions import InvalidToken, SessionClock, find_session
from loans.schemas.documents import IncomeSlot
from loans.schemas.outcome import OutcomeKind, Reason
from loans.tools import descriptions
from loans.tools.check_application import check_application as _check_application
from loans.tools.finish import finish as _finish
from loans.tools.mark_income_verified import mark_income_verified as _mark_income_verified
from loans.tools.read_document import read_document as _read_document
from loans.tools.send_message import send_message as _send_message
from loans.tools.verify_identity import verify_identity as _verify_identity

HOST = "127.0.0.1"
PORT = 8000

# The _meta key that carries the LLM's tool call ID. Custom _meta keys carry
# a prefix, as MCP asks, so they cannot clash with the protocol's own.
CALL_ID_KEY = "loans/call_id"

NOT_ALLOWED = "not allowed: unknown or closed session"


class SessionTokens:
    """Checks each call's token against the open sessions (the SDK's
    TokenVerifier)."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def _find(self, token: str):
        with self._pool.connection() as conn:
            return find_session(conn, token)

    async def verify_token(self, token: str) -> AccessToken | None:
        # The database call blocks, so it runs off the server's event loop.
        session = await anyio.to_thread.run_sync(self._find, token)
        if session is None:
            return None
        return AccessToken(token=token, client_id="case-session", scopes=[])


class Case(NamedTuple):
    """What a tool needs for the call's case."""

    store: PostgresStore
    application_id: str
    clock: SessionClock


def create_server(pool: ConnectionPool, url: str = f"http://{HOST}:{PORT}") -> MCPServer:
    """The server, on the cases reached through pool. url is where it is
    reached, without the /mcp path."""
    server = MCPServer(
        name="loans",
        instructions="The bank's loan application system: actions on one application.",
        token_verifier=SessionTokens(pool),
        auth=AuthSettings(
            issuer_url=url,
            resource_server_url=f"{url}/mcp",
            # The SDK can also check that a token was issued for this server
            # (its "resource"). Session tokens exist only in this server's
            # database, so a token for another server cannot pass anyway.
            validate_token_resource=False,
        ),
    )

    @contextmanager
    def case() -> Generator[Case]:
        """The call's case, found from its token, with a connection of its
        own for the length of the call."""
        access = get_access_token()
        if access is None:  # the SDK refuses such calls first; this is a second lock
            raise ToolError(NOT_ALLOWED)
        with pool.connection() as conn:
            # Checked again: the session may have been closed since the token
            # was checked at the door.
            session = find_session(conn, access.token)
            if session is None:
                raise ToolError(NOT_ALLOWED)
            store = PostgresStore(conn, agent=session.agent)
            try:
                yield Case(store, session.application_id, SessionClock(conn, access.token))
            except InvalidToken:
                raise ToolError(NOT_ALLOWED) from None

    def call_id(ctx: Context) -> str:
        meta = ctx.request_context.meta or {}
        value = meta.get(CALL_ID_KEY)
        if not isinstance(value, str) or not value:
            raise ToolError(f"not done: the request has no call ID (_meta {CALL_ID_KEY!r})")
        return value

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
        call = call_id(ctx)
        with case() as c:
            return _mark_income_verified(c.store, c.application_id, c.clock(), call, slot, reason)

    @server.tool(description=descriptions.SEND_MESSAGE)
    def send_message(
        text: Annotated[str, Field(description=descriptions.MESSAGE_TEXT)],
        reason: Annotated[str, Field(description=descriptions.MESSAGE_REASON)],
        ctx: Context,
    ) -> str:
        call = call_id(ctx)
        with case() as c:
            return _send_message(c.store, c.application_id, c.clock(), call, text, reason)

    @server.tool(description=descriptions.FINISH)
    def finish(
        outcome: OutcomeKind,
        reason: Reason,
        explanation: Annotated[str, Field(description=descriptions.EXPLANATION)],
        ctx: Context,
    ) -> str:
        call = call_id(ctx)
        with case() as c:
            return _finish(
                c.store, c.application_id, c.clock(), call, outcome, reason, explanation
            )

    return server


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    url = f"http://{HOST}:{PORT}"
    with ConnectionPool(
        database_conninfo(), kwargs={"autocommit": True}, min_size=1, max_size=5
    ) as pool:
        with pool.connection() as conn:
            create_tables(conn)  # only what is missing, so safe at every start
        server = create_server(pool, url)
        # Each call stands on its own (stateless): the token on every call is
        # all the server needs.
        app = server.streamable_http_app(stateless_http=True, json_response=True, host=HOST)
        uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
