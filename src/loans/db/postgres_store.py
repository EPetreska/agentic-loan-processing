"""The application store in Postgres.

The rules are ApplicationStore's (loans.store); only where the cases are kept
differs. Each case is one row in the applications table, holding the case as
JSON, and its audit log is in the audit_events table, one row per event,
which Postgres lets programs only add to (see schema.sql).

Every change runs in one transaction: the case's row is locked, changed and
saved, together with any audit event logged inside the change. So two
programs changing the same case cannot overwrite each other's changes, and
the store's own changes (submit, upload, correct_fields, record_reply) are
never saved without their event. The tools that record something
(send_message, mark_income_verified, finish) still log their event in a
second step, after the change.

The store needs a connection in autocommit mode (see connect), so that each
change is its own transaction and a read never leaves one open.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from loans.audit import AgentVersion, AuditEvent
from loans.keyring import POSTGRES_PASSWORD, keyring_lookup
from loans.store import ApplicationStore, StoredApplication, UnknownApplication

SCHEMA = Path(__file__).with_name("schema.sql")

# The database docker compose starts (see docker-compose.yml).
DEFAULT_DATABASE_URL = "postgresql://loans@localhost:5432/loans"


def database_conninfo(**overrides) -> str:
    """The database's address from DATABASE_URL (see .env.example), with the
    password from the keyring (or PGPASSWORD, if set). overrides change parts
    of it, for example dbname="loans_test"."""
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    password = os.environ.get("PGPASSWORD") or keyring_lookup(*POSTGRES_PASSWORD)
    return make_conninfo(url, password=password, **overrides)


def connect(conninfo: str, **kwargs) -> psycopg.Connection:
    """A connection the store can use, for example
    connect("postgresql://loans@localhost:5432/loans", password=...)."""
    return psycopg.connect(conninfo, autocommit=True, **kwargs)


def create_tables(conn: psycopg.Connection) -> None:
    """Create the tables, if they are not there yet."""
    conn.execute(SCHEMA.read_text())


def _as_json(stored: StoredApplication) -> Jsonb:
    """The case as saved in its row: everything but the audit log."""
    return Jsonb(stored.model_dump(mode="json", exclude={"audit_log"}))


class PostgresStore(ApplicationStore):
    def __init__(
        self,
        conn: psycopg.Connection,
        run_dir: Path | None = None,
        agent: AgentVersion | None = None,
    ) -> None:
        if not conn.autocommit:
            raise ValueError("the store needs a connection in autocommit mode (see connect)")
        super().__init__(run_dir, agent)
        self._conn = conn

    @property
    def connection(self) -> psycopg.Connection:
        """The store's connection, for the workflow's other database work
        (sessions)."""
        return self._conn

    def has(self, application_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM applications WHERE application_id = %s", (application_id,)
        ).fetchone()
        return row is not None

    def get(self, application_id: str) -> StoredApplication:
        with self._conn.transaction():
            row = self._conn.execute(
                "SELECT data FROM applications WHERE application_id = %s", (application_id,)
            ).fetchone()
            if row is None:
                raise UnknownApplication(application_id)
            events = (
                self._conn.cursor(row_factory=dict_row)
                .execute(
                    "SELECT application_id, at, actor, event, details, reason, model,"
                    " prompt_version FROM audit_events WHERE application_id = %s ORDER BY id",
                    (application_id,),
                )
                .fetchall()
            )
        return StoredApplication.model_validate(row[0] | {"audit_log": events})

    @contextmanager
    def _adding(self, stored: StoredApplication) -> Generator[None]:
        with self._conn.transaction():
            try:
                self._conn.execute(
                    "INSERT INTO applications (application_id, data) VALUES (%s, %s)",
                    (stored.application_id, _as_json(stored)),
                )
            except psycopg.errors.UniqueViolation:
                raise ValueError(f"application {stored.application_id} already exists") from None
            yield

    @contextmanager
    def _changing(self, application_id: str) -> Generator[StoredApplication]:
        # FOR UPDATE locks the row until the transaction ends, so another
        # program that wants to change this case waits for this change.
        with self._conn.transaction():
            row = self._conn.execute(
                "SELECT data FROM applications WHERE application_id = %s FOR UPDATE",
                (application_id,),
            ).fetchone()
            if row is None:
                raise UnknownApplication(application_id)
            stored = StoredApplication.model_validate(row[0])
            yield stored
            self._conn.execute(
                "UPDATE applications SET data = %s WHERE application_id = %s",
                (_as_json(stored), application_id),
            )

    def _append_event(self, event: AuditEvent) -> None:
        try:
            self._conn.execute(
                "INSERT INTO audit_events (application_id, at, actor, event, details,"
                " reason, model, prompt_version) VALUES (%(application_id)s, %(at)s,"
                " %(actor)s, %(event)s, %(details)s, %(reason)s, %(model)s, %(prompt_version)s)",
                event.model_dump(),
            )
        except psycopg.errors.ForeignKeyViolation:
            raise UnknownApplication(event.application_id) from None
