"""simulate_case: runs one eval case from start to end, playing the outside
world: the scripted customer, and time.

It is only for testing (the evals, and run.py on a laptop). It turns the
case's script into events and hands them to the bank's workflow
(loans.workflow), which reacts as it would to a real customer: it gives the
agent its tools, runs the bank's own steps, and the agent loop
(loans.agent.graph) runs until it pauses or ends. In a real bank the events
come from real customers, staff and the clock instead, and everything
outside this file stays the same.

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

When the agent ends the case, the workflow runs the bank's next steps
(loans.workflow.after_application). Time is simulated: the clock starts at the case's
start time and moves one minute per tool call, plus the jumps above.

A pause is the run stopping in the agent loop's wait box; the checkpointer
keeps the conversation under the case's thread ID, and the simulation resumes
it with the news. The run script uses this for one case; the eval runner
will use it for many.

The agent reaches its tools in one of two ways:

- Through the MCP server (mcp_url given), as in a real deployment. The store
  is then a PostgresStore, the bank's database, which the simulation uses as
  the customer portal would; the workflow opens the case's session, and the
  clock is the session's, shared with the server.
- Directly (no mcp_url): the tools run inside this program, on the store
  given (usually in memory), with the simulation's own clock. Quicker, and
  needs no server or database.

The LLM sees the same tools either way, and the same rules apply.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command

from loans import policy
from loans.agent.graph import build_graph, run_until_pause, thread_config
from loans.agent.prompt import customer_reply_turn, first_turn, no_answer_turn
from loans.checks.ending import last_contact
from loans.evals.case import EvalCase, Reply, submit_case
from loans.schemas.application import RawFile
from loans.store import ApplicationStore, CustomerReply
from loans.workflow import after_application, open_case_for_agent

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
    start_time = case.info.start_time
    tools_and_clock = open_case_for_agent(store, application_id, start_time, mcp_url, SimulatedClock(start_time))
    async with tools_and_clock as (tools, clock):
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
            messages = await run_until_pause(graph, run_input, config, on_message, on_token)
            stored = store.get(application_id)
            if stored.ending is not None:
                after_application(store, application_id, clock)
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
