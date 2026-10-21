"""Secrets from the system keyring.

API keys and the database password are kept in the system keyring (GNOME
keyring, read with secret-tool), never in .env or any file in the project.
Store a secret once with, for example:

    secret-tool store --label="LangSmith API key" service langsmith account apikey

Used by the runs (loans.run), the MCP server (loans.mcp.server) and the
tests, which is why it lives on its own: the server must not load the
agent's code just to read the database password.
"""

import subprocess

# Where each secret is kept: (service, account).

# The API keys, by the environment variable each is put in for a run (see
# load_keys in loans.run).
API_KEYS = {
    "LANGSMITH_API_KEY": ("langsmith", "apikey"),
    "ANTHROPIC_API_KEY": ("anthropic", "apikey"),
}

# The Postgres password (see docker-compose.yml).
POSTGRES_PASSWORD = ("loans-postgres", "postgres")


def keyring_lookup(service: str, account: str) -> str | None:
    """A secret from the system keyring, or None if it is not there."""
    try:
        found = subprocess.run(
            ["secret-tool", "lookup", "service", service, "account", account],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return found.stdout.strip() or None
