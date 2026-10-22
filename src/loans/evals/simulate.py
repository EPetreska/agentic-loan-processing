"""simulate_case: runs one eval case from start to end, playing the outside
world: the scripted customer, and time.

It is only for testing (the evals, and run.py on a laptop). It turns the
case's script into events and hands them to the bank's workflow
(loans.workflow), which reacts as it would to a real customer: it gives the
agent its tools, runs the bank's own steps, and the agent loop
(loans.agents) runs until it pauses or ends. In a real bank the events
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
(loans.workflow.after_application): the credit check and, if it passes, the
offer, sent at once.

A case can also start at the offer phase (case.json's starts_at: "offer"):
the application is then set up as completed without the application agent,
logged as eval setup, and the bank's steps and the offer phase follow as
below. This evaluates the offer agent alone, without spending the
application agent's run on every case.

If the case's script goes on after the offer (customer.json's offer_phase),
the simulation then plays the offer phase with the offer agent, on its own
thread: each scripted message from the customer starts the offer agent (the
first, together with the offer) or resumes it, and the agent runs until it
pauses. Then the customer signs the latest offer, and the workflow closes
the offer phase as accepted (loans.workflow.sign_offer); or, if the script
has no signature, the clock moves to the end of the offer's validity and the
workflow closes it as expired (loans.workflow.expire_if_due). Either way the
offer agent's paused run is ended without the LLM (loans.agents.loop.end_run).

The loan officer is scripted too (the case's officer.json): whenever the
offer agent pauses with a request waiting, the officer decides it after the
script's delay, through the workflow (loans.workflow.decide_approval; an
approval creates and sends the revised offer), and the agent is resumed
with the decision. Time is simulated: the clock starts at the case's
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command

from loans import policy
from loans.agents.application.graph import build_application_graph
from loans.agents.loop import end_run, run_until_pause, thread_config
from loans.agents.application.prompt import customer_reply_turn, first_turn, no_answer_turn
from loans.agents.offer import prompt as offer_prompt
from loans.agents.offer.graph import build_offer_graph
from loans.checks.ending import last_contact
from loans.evals.case import EvalCase, OfferPhaseScript, OfficerScript, Reply, submit_case
from loans.schemas.application import RawFile
from loans.store import ApplicationStore, CaseEnding
from loans.workflow import (
    Search,
    after_application,
    decide_approval,
    expire_if_due,
    open_case_for_agent,
    receive_reply,
    sign_offer,
    valid_until,
)

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
    # The checkpointer thread the application agent's conversation was kept
    # under (None for a case that starts at the offer phase).
    thread_id: str | None
    # The whole conversation the agent saw, oldest first.
    messages: list[BaseMessage]
    # How many times the agent stopped to wait.
    pauses: int
    # True if the safety limit stopped the simulation.
    stopped_by_limit: bool
    # The offer agent's thread and conversation, if the case went on to the
    # offer phase and the customer wrote (else None and empty).
    offer_thread_id: str | None = None
    offer_messages: list[BaseMessage] = field(default_factory=list)


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
    search: Search | None = None,
) -> SimulationResult:
    """Run the agent on one case until the case ends or the limit is hit.

    on_message, if given, is called with every new message as it happens: the
    message that starts the case, each answer of the agent, each tool result,
    and each message that resumes the case. on_token, if given, is called with
    the model's words while it writes them, as ("thinking", text) or
    ("answer", text). run.py --show uses both.

    thread_id names the case's part of the checkpointer thread that keeps
    the conversation (default: the application ID; the application agent's
    thread is then "<thread_id>/application"). LangGraph also passes it to LangSmith,
    which groups the traces of one thread, so it must differ between runs:
    otherwise every run of the same case would end up in one thread. run.py
    gives each run's case its own application ID, so the default does.

    application_id is the ID the case's application is submitted under
    (default: the case ID). mcp_url, if given, is the MCP server the agent
    uses; store must then be a PostgresStore on the server's database.

    search is the document search the offer agent uses directly (needed only
    for a case with an offer phase; through MCP the server runs it)."""
    application_id = submit_case(store, case, application_id)
    if case.info.starts_at == "offer":
        return await _start_at_offer(
            case, store, application_id, model, config, on_message, on_token,
            thread_id or application_id, mcp_url, search,
        )  # fmt: skip
    thread_id_base = thread_id or application_id
    config = {**(config or {}), **thread_config(thread_id_base, "application")}
    thread_id = config["configurable"]["thread_id"]
    start_time = case.info.start_time
    tools_and_clock = open_case_for_agent(
        store, application_id, start_time, mcp_url, SimulatedClock(start_time), "application"
    )
    async with tools_and_clock as (tools, clock):
        graph = build_application_graph(model, tools)
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
                result = SimulationResult(application_id, thread_id, messages, pauses, False)
                break
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

    # The application agent's access is closed; the offer phase opens the
    # case again, for the offer agent.
    base_config = {k: v for k, v in config.items() if k != "configurable"}
    await _offer_phase_if_scripted(
        case, store, result, model, base_config, on_message, on_token, thread_id_base, mcp_url, search
    )
    return result


async def _offer_phase_if_scripted(
    case: EvalCase,
    store: ApplicationStore,
    result: SimulationResult,
    model: BaseChatModel,
    base_config: dict,
    on_message: Callable[[BaseMessage], None] | None,
    on_token: Callable[[str, str], None] | None,
    thread_id_base: str,
    mcp_url: str | None,
    search: Search | None,
) -> None:
    """The offer phase, if the case's script has one and an offer was sent;
    the offer agent's thread and conversation go into result."""
    script = case.customer.offer_phase
    if script is None or not store.get(result.application_id).offers_sent:
        return
    offer_config = {**base_config, **thread_config(thread_id_base, "offer")}
    result.offer_messages = await _simulate_offer_phase(
        store, result.application_id, script, case.officer, model, mcp_url, search,
        offer_config, on_message, on_token,
    )  # fmt: skip
    if result.offer_messages:
        result.offer_thread_id = offer_config["configurable"]["thread_id"]


# The ending recorded for a case that starts at the offer phase.
EVAL_SETUP_CALL_ID = "eval-setup"


async def _start_at_offer(
    case: EvalCase,
    store: ApplicationStore,
    application_id: str,
    model: BaseChatModel,
    config: dict | None,
    on_message: Callable[[BaseMessage], None] | None,
    on_token: Callable[[str, str], None] | None,
    thread_id_base: str,
    mcp_url: str | None,
    search: Search | None,
) -> SimulationResult:
    """A case that starts at the offer phase: the application is set up as
    completed, marked as eval setup in the audit log (actor "system", never
    the agent), then the bank's steps run as after a real completion (the
    credit check, the offer sent), and the offer phase follows."""
    clock = SimulatedClock(case.info.start_time)
    now = clock()
    store.record_ending(
        application_id,
        CaseEnding(
            call_id=EVAL_SETUP_CALL_ID,
            outcome="complete",
            reason="all_checks_passed",
            explanation="Set up as completed for an eval that starts at the offer phase; no agent ran.",
            ended_at=now,
        ),
    )
    store.log_event(
        application_id, now, "system", "eval_setup",
        "the application set up as completed; the application agent did not run",
    )  # fmt: skip
    after_application(store, application_id, clock)
    result = SimulationResult(application_id, None, [], 0, False)
    await _offer_phase_if_scripted(
        case, store, result, model, dict(config or {}), on_message, on_token, thread_id_base, mcp_url, search
    )
    return result


async def _simulate_offer_phase(
    store: ApplicationStore,
    application_id: str,
    script: OfferPhaseScript,
    officer: OfficerScript | None,
    model: BaseChatModel,
    mcp_url: str | None,
    search: Search | None,
    config: dict,
    on_message: Callable[[BaseMessage], None] | None,
    on_token: Callable[[str, str], None] | None,
) -> list[BaseMessage]:
    """The offer phase, from the offer sent to the signature: each scripted
    message reaches the bank (receive_reply) and the offer agent, which is
    started with the offer and the first message and resumed with the later
    ones. Whenever the agent pauses with a request waiting and the officer's
    script has a decision left, the officer decides (decide_approval) and the
    agent is resumed with the decision. Then the customer signs the latest
    offer (sign_offer), or never signs and the offer expires (expire_if_due,
    once the offer's validity has passed), and the offer agent's paused run
    is ended without the LLM (end_run). Returns the offer agent's conversation (empty if the
    customer never wrote)."""
    sent = store.get(application_id).offers_sent[-1]
    decisions = list(officer.decisions) if officer else []
    messages: list[BaseMessage] = []
    tools_and_clock = open_case_for_agent(
        store, application_id, sent.sent_at, mcp_url, SimulatedClock(sent.sent_at), "offer", search
    )
    async with tools_and_clock as (tools, clock):
        graph = build_offer_graph(model, tools)

        async def run(run_input) -> list[BaseMessage]:
            return await run_until_pause(graph, run_input, config, on_message, on_token)

        last = sent.sent_at  # the offer, then the agent's last message
        for number, scripted in enumerate(script.messages):
            clock.jump_to(last + timedelta(days=scripted.after_days))
            now = clock()
            receive_reply(store, application_id, now, scripted.message)
            if number == 0:
                text = offer_prompt.first_turn(sent.notice, sent.sent_at.date(), scripted.message, now.date())
                start = HumanMessage(text)
                if on_message:
                    on_message(start)
                messages = await run({"messages": [start]})
            else:
                text = offer_prompt.customer_message_turn(scripted.after_days, scripted.message, now.date())
                messages = await run(Command(resume=text))

            # The loan officer decides a waiting request, if the script says how.
            while (request := store.get(application_id).pending_approval()) and decisions:
                decision = decisions.pop(0)
                clock.jump_to(request.requested_at + timedelta(days=decision.after_days))
                decided_at = clock()
                decide_approval(
                    store, application_id, officer.officer_id, decision.decision, decision.note,
                    decided_at, amount=decision.amount, term_months=decision.term_months,
                )  # fmt: skip
                if decision.decision == "approve":
                    revised = store.get(application_id).offers_sent[-1].notice
                    news = offer_prompt.officer_decision_turn("approved", decided_at.date(), revised)
                else:
                    news = offer_prompt.officer_decision_turn("declined", decided_at.date())
                messages = await run(Command(resume=news))

            sent_to_customer = store.get(application_id).messages
            last = max(last, now, sent_to_customer[-1].sent_at if sent_to_customer else now)

        latest = store.get(application_id).offers_sent[-1]
        if script.signs_after_days is not None:
            clock.jump_to(last + timedelta(days=script.signs_after_days))
            sign_offer(store, application_id, latest.offer_index, clock())
        else:
            # She never signs: the bank's daily check finds the offer expired.
            clock.jump_to(valid_until(latest))
            expire_if_due(store, application_id, clock())
        closing = store.get(application_id).offer_closing
        if messages:  # the offer agent ran, so its paused run is ended
            text = offer_prompt.closed_turn(closing.outcome, closing.closed_at.date())
            messages = await run(end_run(text))
    return messages


def _apply_reply(
    store: ApplicationStore, application_id: str, case: EvalCase, reply: Reply, clock
) -> str:
    """The scripted customer's reply: what Maria sends, from the script,
    handed to the bank's workflow to record (loans.workflow.receive_reply).
    Returns the message that tells the agent about the reply."""
    received_at = clock()  # the reply and its attachments arrive together
    uploads = {
        slot: (
            case.documents[file_name],
            RawFile(filename=file_name, content_type=CONTENT_TYPES.get(slot, DEFAULT_CONTENT_TYPE)),
        )
        for slot, file_name in reply.uploads.items()
    }
    receive_reply(store, application_id, received_at, reply.message, uploads, reply.changes)

    return customer_reply_turn(
        reply.after_days,
        reply.message,
        list(reply.uploads),
        list(reply.changes),
        received_at.date(),
    )
