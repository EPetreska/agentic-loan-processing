"""Shared test setup: a Postgres database for the database tests.

The database tests use their own database, loans_test, in the Postgres that
docker compose starts (see docker-compose.yml), so they never touch the
loans database. Each test starts from empty tables. If Postgres cannot be
reached, the database tests are skipped with the reason, and all other tests
run without Docker.

The address comes from DATABASE_URL (see .env.example) and the password from
the keyring, as for the server (see database_conninfo).
"""

import psycopg
import pytest

from loans.db.postgres_store import PostgresStore, connect, create_tables, database_conninfo

TEST_DATABASE = "loans_test"


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
