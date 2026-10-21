-- The tables of the bank's application system. Safe to run again: it only
-- creates what is missing (see create_tables in postgres_store.py).

-- One row per case: everything about it as JSON (the form, the documents,
-- the messages, the replies, the ending), except the audit log.
CREATE TABLE IF NOT EXISTS applications (
    application_id text PRIMARY KEY,
    data jsonb NOT NULL
);

-- The audit log, one row per event, in the order the events happened.
CREATE TABLE IF NOT EXISTS audit_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    application_id text NOT NULL REFERENCES applications,
    at timestamptz NOT NULL,
    actor text NOT NULL,
    event text NOT NULL,
    details text NOT NULL,
    reason text,
    model text,
    prompt_version text
);

CREATE INDEX IF NOT EXISTS audit_events_by_case ON audit_events (application_id, id);

-- The audit log is append-only: Postgres itself refuses to change, delete or
-- empty it, whatever program asks. (Dropping the whole table is still
-- possible for the database owner; separate database roles for the services
-- come with the deployment.)
CREATE OR REPLACE FUNCTION refuse_audit_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'the audit log is append-only: % is not allowed', TG_OP;
END
$$;

CREATE OR REPLACE TRIGGER audit_events_append_only
    BEFORE UPDATE OR DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION refuse_audit_change();

CREATE OR REPLACE TRIGGER audit_events_not_emptied
    BEFORE TRUNCATE ON audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION refuse_audit_change();

-- Sessions: which case a caller may work on (see loans.mcp.sessions). Only a
-- hash of each token is kept, so the tokens cannot be read from this table.
CREATE TABLE IF NOT EXISTS case_sessions (
    token_hash text PRIMARY KEY,
    application_id text NOT NULL REFERENCES applications,
    -- The case's current time. It is simulated in the evals, so the MCP
    -- server and the simulation share it here.
    clock timestamptz NOT NULL,
    -- The running agent, for the audit log.
    model text,
    prompt_version text,
    -- Real times, not the case's clock.
    opened_at timestamptz NOT NULL DEFAULT now(),
    closed_at timestamptz
);

-- At most one open session per case, so two agents cannot work on one case
-- at the same time.
CREATE UNIQUE INDEX IF NOT EXISTS case_sessions_one_open
    ON case_sessions (application_id) WHERE closed_at IS NULL;
