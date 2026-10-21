"""Sessions: which case a caller may work on, across the network.

The MCP server holds every case and can be reached by any program, so each
call must say which case it is for and prove that the caller may touch it. A
call never names a case. Instead, when the bank's workflow (run.py) starts the
agent on a case, it opens a session for that case and gets a token, a random
code, which it hands to the agent. The agent sends the token with every call,
and the server finds the case from it. A token opens one case only, and stops
working when the session is closed. It does across the network what
make_tools does inside one program, where the case ID is built into the tools.

The workflow is trusted, so it opens and closes sessions by writing to the
database directly; the server only checks tokens. A case has at most one
open session, so two agents cannot work on it at the same time.

Only a hash of each token is kept, as with passwords: someone who can read
the table cannot use the tokens in it. (A token is long and random, so a
plain SHA-256 hash is enough; passwords need slow, salted hashes because
people choose guessable ones.)

The session also holds the case's clock, which moves one minute each time a
tool reads it and jumps forward when the customer takes days to reply, as
SimulatedClock does in memory; and the model and prompt version of the
agent, for the audit log.
"""

import hashlib
import secrets
from datetime import datetime
from typing import NamedTuple

import psycopg

from loans.audit import AgentVersion
from loans.store import UnknownApplication


class InvalidToken(LookupError):
    """No open session has this token."""


class Session(NamedTuple):
    application_id: str
    agent: AgentVersion | None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def open_session(
    conn: psycopg.Connection,
    application_id: str,
    start: datetime,
    agent: AgentVersion | None = None,
) -> str:
    """Open a session for one case, with its clock at start, and return its
    token. Refused if the case already has an open session."""
    token = secrets.token_urlsafe(32)
    try:
        conn.execute(
            "INSERT INTO case_sessions (token_hash, application_id, clock, model, prompt_version)"
            " VALUES (%s, %s, %s, %s, %s)",
            (
                _hash(token),
                application_id,
                start,
                agent.model if agent else None,
                agent.prompt_version if agent else None,
            ),
        )
    except psycopg.errors.ForeignKeyViolation:
        raise UnknownApplication(application_id) from None
    except psycopg.errors.UniqueViolation:
        raise ValueError(f"application {application_id} already has an open session") from None
    return token


def close_session(conn: psycopg.Connection, token: str) -> None:
    """Close the session: its token is refused from now on."""
    conn.execute(
        "UPDATE case_sessions SET closed_at = now() WHERE token_hash = %s AND closed_at IS NULL",
        (_hash(token),),
    )


def find_session(conn: psycopg.Connection, token: str) -> Session | None:
    """The case and agent of the open session with this token, or None if the
    token is unknown or its session is closed."""
    row = conn.execute(
        "SELECT application_id, model, prompt_version FROM case_sessions"
        " WHERE token_hash = %s AND closed_at IS NULL",
        (_hash(token),),
    ).fetchone()
    if row is None:
        return None
    application_id, model, prompt_version = row
    agent = AgentVersion(model=model, prompt_version=prompt_version) if model else None
    return Session(application_id, agent)


class SessionClock:
    """The case's clock, kept in its session: the same use as SimulatedClock,
    but shared by every program that has the token (the MCP server and the
    simulation). Refuses with InvalidToken once the session is closed."""

    def __init__(self, conn: psycopg.Connection, token: str) -> None:
        self._conn = conn
        self._hash = _hash(token)

    def _one(self, sql: str, params: tuple = ()) -> datetime:
        row = self._conn.execute(sql, (*params, self._hash)).fetchone()
        if row is None:
            raise InvalidToken("no open session has this token")
        return row[0]

    def __call__(self) -> datetime:
        """The time for one step, after which the clock moves one minute."""
        # One statement, so two programs reading the clock at the same time
        # cannot both get the same minute.
        return self._one(
            "UPDATE case_sessions SET clock = clock + interval '1 minute'"
            " WHERE token_hash = %s AND closed_at IS NULL RETURNING clock - interval '1 minute'"
        )

    @property
    def now(self) -> datetime:
        """The current time, without moving the clock."""
        return self._one(
            "SELECT clock FROM case_sessions WHERE token_hash = %s AND closed_at IS NULL"
        )

    def jump_to(self, time: datetime) -> None:
        """Move forward to this time; never backward."""
        self._one(
            "UPDATE case_sessions SET clock = greatest(clock, %s)"
            " WHERE token_hash = %s AND closed_at IS NULL RETURNING clock",
            (time,),
        )
