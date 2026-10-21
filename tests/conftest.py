"""Shared test setup: a Postgres database for the database tests.

The database tests use their own database, loans_test, in the Postgres that
docker compose starts (see docker-compose.yml), so they never touch the
loans database. Each test starts from empty tables. If Postgres cannot be
reached, the database tests are skipped with the reason, and all other tests
run without Docker.

The address comes from DATABASE_URL (see .env.example) and the password from
the keyring, as for the server (see database_conninfo).
"""

import socket
import threading
import time
from pathlib import Path

import psycopg
import pytest
import uvicorn
from psycopg_pool import ConnectionPool

from loans.audit import AgentVersion
from loans.db.postgres_store import PostgresStore, connect, create_tables, database_conninfo
from loans.evals.case import load_case, submit_case
from loans.mcp.server import create_server
from loans.mcp.sessions import open_session

TEST_DATABASE = "loans_test"
CASES = Path(__file__).parent.parent / "evals" / "cases"
AGENT = AgentVersion(model="ollama:qwen3:4b", prompt_version="v1")


@pytest.fixture(scope="session")
def pg_conninfo() -> str:
    """The test database's address, created if it is not there yet."""
    try:
        with connect(database_conninfo(connect_timeout=3)) as admin:
            exists = admin.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DATABASE,)
            ).fetchone()
            if not exists:
                admin.execute(f"CREATE DATABASE {TEST_DATABASE}")
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable, start it with docker compose up -d ({e})")
    return database_conninfo(dbname=TEST_DATABASE, connect_timeout=3)


@pytest.fixture
def pg_conn(pg_conninfo):
    """A connection to the test database, with empty tables."""
    with connect(pg_conninfo) as conn:
        conn.execute("DROP TABLE IF EXISTS case_sessions, audit_events, applications")
        create_tables(conn)
        yield conn


@pytest.fixture
def pg_store(pg_conn) -> PostgresStore:
    return PostgresStore(pg_conn)


@pytest.fixture
def anyio_backend():
    """The async tests run on asyncio, as the agent does."""
    return "asyncio"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_url(pg_conninfo):
    """The MCP server's address, with the server running on the test
    database in a background thread, as it would run on its own."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    with ConnectionPool(pg_conninfo, kwargs={"autocommit": True}, min_size=1, max_size=5) as pool:
        app = create_server(pool, url).streamable_http_app(
            stateless_http=True, json_response=True, host="127.0.0.1"
        )
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.01)
        yield f"{url}/mcp"
        server.should_exit = True
        thread.join(5)


@pytest.fixture
def maria(pg_store, pg_conn):
    """Maria's case, submitted, with an open session: (case, ID, token)."""
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(pg_store, case)
    return case, app_id, open_session(pg_conn, app_id, case.info.start_time, AGENT)
