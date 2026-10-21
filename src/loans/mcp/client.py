"""The MCP client: the server's tools, as LangChain tools for the agent.

load_tools asks the MCP server (loans.mcp.server) for its tools and turns
each into a LangChain tool the agent can call, so the server is the one
source of what the LLM reads about the tools. A small client of our own on
the official MCP SDK, rather than a LangChain MCP library: LangChain's
langchain-mcp-adapters needs version 1 of the SDK, and its replacement is in
beta; and the two jobs here, sending the token and the call ID, are the
security-critical part, so they stay in code we can read line by line.

- The token goes with every call, in the Authorization header. The LLM never
  sees it.
- The call ID is the ID of the LLM's tool call, filled in by LangGraph
  (InjectedToolCallId), as in loans.agent.tools, so it is not one of the
  parameters the LLM sees. It goes in the request's _meta field, so the
  server does not do a retried send_message or finish twice.
- The tool list is checked: the client refuses a server that offers any tool
  other than the expected six, so a changed or fake server cannot slip a new
  tool in front of the LLM.

Two kinds of failure are kept apart. When the bank's rules refuse an action
("not sent: ..."), the tool returns that text and the LLM can correct
itself, as with the direct tools; a wrong argument (an unknown slot) is
caught before the call and also goes back to the LLM. When the system fails
(a bad or closed token, the server down or too slow), ServerError stops the
run: it is not something the agent can fix, so the LLM never sees it.

Each call opens its own connection. The server keeps nothing between calls,
so there is no long-lived connection to keep alive during a pause of days,
and the extra milliseconds are nothing next to an LLM step.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx2
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.types import Tool
from pydantic import BaseModel, Field, create_model

from loans.mcp import CALL_ID_KEY

# The tools the agent expects from the server, and nothing else.
EXPECTED_TOOLS = frozenset(
    {
        "check_application",
        "read_document",
        "verify_identity",
        "mark_income_verified",
        "send_message",
        "finish",
    }
)

# How long to wait for the server before giving up, in seconds. Retrying a
# call that timed out comes with the tool-timeout failure case.
TIMEOUT_SECONDS = 30


class ServerError(RuntimeError):
    """The MCP server could not be used: a bad or closed token, the server
    down or too slow, or an unexpected tool list. Stops the run."""


@asynccontextmanager
async def connect(url: str, token: str | None) -> AsyncGenerator[ClientSession]:
    """An MCP connection to the server at url, sending the token."""
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    async with (
        create_mcp_http_client(headers=headers, timeout=httpx2.Timeout(TIMEOUT_SECONDS)) as http,
        streamable_http_client(url, http_client=http) as (read, write),
        ClientSession(read, write, read_timeout_seconds=TIMEOUT_SECONDS) as session,
    ):
        await session.initialize()
        yield session


def _problem(error: BaseException) -> str:
    """The first real error inside an error group, as text."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"


def server_is_running(url: str) -> bool:
    """True if something answers at url. Any answer counts: without a token
    the server refuses (401), which shows it is up."""
    try:
        httpx2.get(url, timeout=3)
    except httpx2.TransportError:
        return False
    return True


async def load_tools(url: str, token: str) -> list[BaseTool]:
    """The server's tools for the case the token opens, as LangChain tools."""
    try:
        async with connect(url, token) as session:
            listed = (await session.list_tools()).tools
    except Exception as e:  # including error groups from the connection
        raise ServerError(f"cannot load the tools from {url}: {_problem(e)}") from e
    names = {tool.name for tool in listed}
    if names != EXPECTED_TOOLS:
        raise ServerError(
            f"unexpected tools from {url}: offers {sorted(names - EXPECTED_TOOLS)}, "
            f"lacks {sorted(EXPECTED_TOOLS - names)}"
        )
    return [_as_langchain_tool(url, token, tool) for tool in listed]


def _args_model(tool: Tool) -> type[BaseModel]:
    """The tool's parameters as a model LangChain can check, from the JSON
    schema the server sends, plus the hidden call ID."""
    schema = tool.input_schema
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for name, spec in schema.get("properties", {}).items():
        if "enum" in spec:
            kind: Any = Literal[tuple(spec["enum"])]
        elif spec.get("type") == "string":
            kind = str
        else:
            # The six tools take only text and choices; anything else is a
            # server change this client was not written for.
            raise ServerError(f"{tool.name}: unsupported parameter {name!r}: {spec}")
        default = ... if name in required else spec.get("default")
        fields[name] = (kind, Field(default, description=spec.get("description")))
    fields["call_id"] = (Annotated[str, InjectedToolCallId], ...)
    return create_model(tool.name, **fields)


def _as_langchain_tool(url: str, token: str, tool: Tool) -> BaseTool:
    async def call(call_id: str, **args: Any) -> str:
        try:
            async with connect(url, token) as session:
                result = await session.call_tool(tool.name, args, meta={CALL_ID_KEY: call_id})
        except Exception as e:
            raise ServerError(f"{tool.name} failed: {_problem(e)}") from e
        text = "\n".join(part.text for part in result.content if part.type == "text")
        if result.is_error:
            # The rules answer with text, never an error, so an error here is
            # the system's (for example a session closed during the run).
            raise ServerError(f"{tool.name} refused by the server: {text}")
        return text

    return StructuredTool.from_function(
        coroutine=call,
        name=tool.name,
        description=tool.description or "",
        args_schema=_args_model(tool),
    )
