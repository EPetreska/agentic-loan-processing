# Agentic BPM: loan applications

## How to run a case

The agent works as in a real deployment: the cases are kept in Postgres, and the agent reaches the bank's tools only through the MCP server, with a token for its one case.

1. Store the database password in the keyring, once:
   `openssl rand -base64 24 | tr -d '\n' | secret-tool store --label="Loans Postgres password" service loans-postgres account postgres`
2. Start Postgres:
   `POSTGRES_PASSWORD="$(secret-tool lookup service loans-postgres account postgres)" docker compose up -d`
3. Start the MCP server, in a second terminal (it keeps running):
   `uv run python -m loans.mcp.server`
4. Run a case:
   `uv run python -m loans.run 02_maria_job_change`
   Add `--show` to watch each step, or `--direct` to run without Postgres and the server (everything in memory, the same tools and rules).

Each run leaves its audit log and conversation in `runs/<run time>/<application ID>/`.

## Known limitations

- The bank's workflow (which holds the database password) and the agent (which holds only its case's token) run in one program, `run.py`, so they are kept apart by the code rather than by separate services; in deployment each becomes its own service with its own database user and only the rights it needs.
