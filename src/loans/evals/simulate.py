"""simulate_case: runs one eval case from start to end, playing the scripted
customer.

It submits the application and starts the agent. Each time the agent stops
without ending the case (to wait for the customer), it decides what happens
next, from the case's customer.json:

- The agent has sent a new message and the script has a reply left: the
  clock moves forward by the reply's after_days (counted from the agent's
  last message), the reply is recorded in the store, its uploads and form
  corrections are applied, and the agent continues with the customer's reply.
- Otherwise the customer does not answer: the clock moves forward until
  NO_ANSWER_CANCEL_DAYS days have passed since the last contact (the agent's
  last message or the customer's last reply, as finish counts it), and the
  agent continues with that news.

A safety limit (MAX_PAUSES) stops the simulation if the agent keeps waiting
without ending the case.

When the agent completes the case, the simulation runs the bank's next step,
as the workflow would: the credit check (loans.services.credit_bureau).
Cancelled cases and cases sent to a human go no further. Time is simulated: the clock starts at the case's
start time and moves one minute per tool call, plus the jumps above.

A pause is the run stopping in the agent loop's wait box; the checkpointer
keeps the conversation under the case's thread ID, and the simulation resumes
it with the news. The run script uses this for one case; the eval runner
will use it for many.

The agent reaches its tools in one of two ways:

- Through the MCP server (mcp_url given), as in a real deployment. The store
  is then a PostgresStore, the bank's database, which the simulation uses as
  the bank's workflow and customer portal would: it opens a session for the
  case, hands the agent only the session's token, and closes the session at
  the end, also if the run fails. The clock is the session's, shared with
  the server.
- Directly (no mcp_url): the tools run inside this program, on the store
  given (usually in memory), with a clock of its own. Quicker, and needs no
  server or database.

The LLM sees the same tools either way, and the same rules apply.
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from loans import policy
from loans.agent.graph import build_graph, thread_config
from loans.agent.prompt import customer_reply_turn, first_turn, no_answer_turn
from loans.agent.tools import make_tools
from loans.checks.ending import last_contact
from loans.evals.case import EvalCase, Reply, submit_case
from loans.mcp.client import load_tools
from loans.mcp.sessions import SessionClock, close_session, open_session
from loans.services.credit_bureau import check_credit
from loans.schemas.application import RawFile
from loans.store import ApplicationStore, CustomerReply

# The most times the agent may stop to wait before the simulation gives up.
MAX_PAUSES = 10

# The file type of a simulated upload, by slot.
CONTENT_TYPES = {"id_front": "image/jpeg", "id_back": "image/jpeg"}
DEFAULT_CONTENT_TYPE = "application/pdf"


class SimulatedClock:
    """The case's time: starts at the case's start time, moves one minute
    each time it is read, and can jump forward."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def __call__(self) -> datetime:
        now = self._now
        self._now += timedelta(minutes=1)
        return now

    @property
    def now(self) -> datetime:
        """The current time, without moving the clock."""
        return self._now

    def jump_to(self, time: datetime) -> None:
        """Move forward to this time; never backward."""
        self._now = max(self._now, time)


@dataclass
class SimulationResult:
    application_id: str
    # The checkpointer thread the conversation was kept under.
    thread_id: str
    # The whole conversation the agent saw, oldest first.
    messages: list[BaseMessage]
    # How many times the agent stopped to wait.
    pauses: int
    # True if the safety limit stopped the simulation.
    stopped_by_limit: bool


Clock = SimulatedClock | SessionClock


@asynccontextmanager
async def _agent_tools(
    store: ApplicationStore, application_id: str, start: datetime, mcp_url: str | None
) -> AsyncGenerator[tuple[list[BaseTool], Clock]]:
    """The agent's tools for the case and the case's clock: through the MCP
    server if mcp_url is given, else directly."""
    if mcp_url is None:
        clock = SimulatedClock(start)
        yield make_tools(store, application_id, clock), clock
        return
    # The workflow opens a session for this one case; the agent gets only its
    # token, inside the tools.
    conn = store.connection  # a PostgresStore: the server works on the same database
    token = open_session(conn, application_id, start, store.agent)
    try:
        yield await load_tools(mcp_url, token), SessionClock(conn, token)
    finally:
        close_session(conn, token)


async def simulate_case(
    case: EvalCase,
    store: ApplicationStore,
    model: BaseChatModel,
    config: dict | None = None,
    max_pauses: int = MAX_PAUSES,
    on_message: Callable[[BaseMessage], None] | None = None,
    on_token: Callable[[str, str], None] | None = None,
    thread_id: str | None = None,
    application_id: str | None = None,
    mcp_url: str | None = None,
) -> SimulationResult:
    """Run the agent on one case until the case ends or the limit is hit.

    on_message, if given, is called with every new message as it happens: the
    message that starts the case, each answer of the agent, each tool result,
    and each message that resumes the case. on_token, if given, is called with
    the model's words while it writes them, as ("thinking", text) or
    ("answer", text). run.py --show uses both.

    thread_id names the checkpointer thread that keeps the conversation
    (default: the application ID). LangGraph also passes it to LangSmith,
    which groups the traces of one thread, so it must differ between runs:
    otherwise every run of the same case would end up in one thread. run.py
    gives each run's case its own application ID, so the default does.

    application_id is the ID the case's application is submitted under
    (default: the case ID). mcp_url, if given, is the MCP server the agent
    uses; store must then be a PostgresStore on the server's database."""
    application_id = submit_case(store, case, application_id)
    thread_id = thread_id or application_id
    config = {**(config or {}), **thread_config(thread_id)}
    async with _agent_tools(store, application_id, case.info.start_time, mcp_url) as (tools, clock):
        graph = build_graph(model, tools)
        replies = list(case.customer.replies)
        answered = 0  # how many of the agent's messages the customer has reacted to
        pauses = 0

        # The customer's first message as the bank has it on file.
        first_message = store.get(application_id).first_message
        start = HumanMessage(first_turn(first_message, clock.now.date()))
        if on_message:
            on_message(start)
        run_input = {"messages": [start]}
        while True:
            messages = await _run(graph, run_input, config, on_message, on_token)
            stored = store.get(application_id)
            if stored.ending is not None:
                if stored.ending.outcome == "complete":
                    result = check_credit(stored.metadata, clock())
                    store.record_credit_check(application_id, result)
                return SimulationResult(application_id, thread_id, messages, pauses, False)
            if pauses == max_pauses:
                return SimulationResult(application_id, thread_id, messages, pauses, True)
            pauses += 1

            sent = stored.messages
            last_sent = sent[-1].sent_at if sent else clock()
            if len(sent) > answered and replies:
                answered = len(sent)
                reply = replies.pop(0)
                clock.jump_to(last_sent + timedelta(days=reply.after_days))
                news = _apply_reply(store, application_id, case, reply, clock)
            else:
                answered = len(sent)
                days = policy.NO_ANSWER_CANCEL_DAYS
                # The same count finish uses: from the last message or reply.
                clock.jump_to((last_contact(stored) or clock()) + timedelta(days=days))
                news = no_answer_turn(days, clock.now.date())
            run_input = Command(resume=news)  # continue the paused run with the news


async def _run(graph, run_input, config: dict, on_message, on_token) -> list[BaseMessage]:
    """Run until the next pause or the end, and return the whole conversation.
    With on_message or on_token, the run is streamed: finished messages step
    by step ("updates"), and the model's words as it writes them
    ("messages")."""
    if on_message is None and on_token is None:
        await graph.ainvoke(run_input, config)
        return (await graph.aget_state(config)).values["messages"]

    modes = ["updates", "messages"] if on_token else ["updates"]
    async for mode, data in graph.astream(run_input, config, stream_mode=modes):
        if mode == "messages":
            chunk, _ = data
            if isinstance(chunk, AIMessageChunk):
                thinking = chunk.additional_kwargs.get("reasoning_content")
                if thinking:
                    on_token("thinking", thinking)
                if isinstance(chunk.content, str) and chunk.content:
                    on_token("answer", chunk.content)
        elif on_message:
            for node, values in data.items():
                if node.startswith("__"):  # for example __interrupt__: the pause itself
                    continue
                for message in (values or {}).get("messages", []):
                    on_message(message)
    return (await graph.aget_state(config)).values["messages"]


def _apply_reply(
    store: ApplicationStore, application_id: str, case: EvalCase, reply: Reply, clock
) -> str:
    """Record the reply, apply the customer's uploads and corrections, and
    return the message that tells the agent about the reply."""
    received_at = clock()  # the reply and its attachments arrive together
    store.record_reply(
        application_id,
        CustomerReply(
            received_at=received_at,
            message=reply.message,
            uploads=list(reply.uploads),
            corrected_fields=list(reply.changes),
        ),
    )
    for slot, file_name in reply.uploads.items():
        file = RawFile(
            filename=file_name, content_type=CONTENT_TYPES.get(slot, DEFAULT_CONTENT_TYPE)
        )
        store.upload(application_id, slot, case.documents[file_name], file, now=received_at)
    if reply.changes:
        store.correct_fields(application_id, reply.changes, now=received_at)

    return customer_reply_turn(
        reply.after_days,
        reply.message,
        list(reply.uploads),
        list(reply.changes),
        received_at.date(),
    )
