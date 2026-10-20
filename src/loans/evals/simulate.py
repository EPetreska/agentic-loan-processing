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
without ending the case. Time is simulated: the clock starts at the case's
start time and moves one minute per tool call, plus the jumps above.

A pause is the run stopping in the agent loop's wait box; the checkpointer
keeps the conversation under the case's thread ID, and the simulation resumes
it with the news. The run script uses this for one case; the eval runner
will use it for many.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langgraph.types import Command

from loans import policy
from loans.agent.graph import build_graph, thread_config
from loans.agent.prompt import customer_reply_turn, first_turn, no_answer_turn
from loans.agent.tools import make_tools
from loans.checks.ending import last_contact
from loans.evals.case import EvalCase, Reply, submit_case
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
    # The whole conversation the agent saw, oldest first.
    messages: list[BaseMessage]
    # How many times the agent stopped to wait.
    pauses: int
    # True if the safety limit stopped the simulation.
    stopped_by_limit: bool


def simulate_case(
    case: EvalCase,
    store: ApplicationStore,
    model: BaseChatModel,
    config: dict | None = None,
    max_pauses: int = MAX_PAUSES,
    on_message: Callable[[BaseMessage], None] | None = None,
    on_token: Callable[[str, str], None] | None = None,
) -> SimulationResult:
    """Run the agent on one case until the case ends or the limit is hit.

    on_message, if given, is called with every new message as it happens: the
    message that starts the case, each answer of the agent, each tool result,
    and each message that resumes the case. on_token, if given, is called with
    the model's words while it writes them, as ("thinking", text) or
    ("answer", text). run.py --show uses both."""
    application_id = submit_case(store, case)
    clock = SimulatedClock(case.info.start_time)
    graph = build_graph(model, make_tools(store, application_id, clock), store, application_id)
    config = {**(config or {}), **thread_config(application_id)}
    replies = list(case.customer.replies)
    answered = 0  # how many of the agent's messages the customer has reacted to
    pauses = 0

    start = HumanMessage(first_turn(case.customer.first_message, clock.now.date()))
    if on_message:
        on_message(start)
    run_input = {"messages": [start]}
    while True:
        messages = _run(graph, run_input, config, on_message, on_token)
        stored = store.get(application_id)
        if stored.ending is not None:
            return SimulationResult(application_id, messages, pauses, False)
        if pauses == max_pauses:
            return SimulationResult(application_id, messages, pauses, True)
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


def _run(graph, run_input, config: dict, on_message, on_token) -> list[BaseMessage]:
    """Run until the next pause or the end, and return the whole conversation.
    With on_message or on_token, the run is streamed: finished messages step
    by step ("updates"), and the model's words as it writes them
    ("messages")."""
    if on_message is None and on_token is None:
        graph.invoke(run_input, config)
        return graph.get_state(config).values["messages"]

    modes = ["updates", "messages"] if on_token else ["updates"]
    for mode, data in graph.stream(run_input, config, stream_mode=modes):
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
    return graph.get_state(config).values["messages"]


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
