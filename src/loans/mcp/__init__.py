"""The bank's application system as an MCP server (server.py), with its
sessions (sessions.py) and the agent's client (client.py)."""

# The _meta key that carries the LLM's tool call ID from the client to the
# server. Custom _meta keys carry a prefix, as MCP asks, so they cannot clash
# with the protocol's own. Kept here so the client does not load the server's
# code to know it.
CALL_ID_KEY = "loans/call_id"
