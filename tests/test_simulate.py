"""Tests for the scripted-customer simulation, with a fake LLM.

The fake LLM plays back fixed answers, so these tests check the simulation
(replies, uploads, the clock, the safety limit), not the agent's judgment.
"""

from datetime import timedelta
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from loans.evals.case import load_case
from loans.evals.simulate import SimulatedClock, simulate_case
from loans.store import ApplicationStore

CASES = Path(__file__).parent.parent / "evals" / "cases"
FINISH_COMPLETE = {"outcome": "complete", "reason": "all_checks_passed", "explanation": "Done."}


class FakeModel(GenericFakeChatModel):
    """Plays back a fixed list of answers; accepts tools like a real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def step(name, args=None, call_id=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id or f"call-{name}"}],
    )


def wait(text="Waiting for the customer."):
    return AIMessage(content=text)


def ask(text="Please upload a recent pay stub.", call_id="call-ask"):
    return step("send_message", {"text": text, "reason": "Missing document."}, call_id)


def simulate(case_name, answers, max_pauses=10):
    store = ApplicationStore()
    case = load_case(CASES / case_name)
    result = simulate_case(case, store, FakeModel(messages=iter(answers)), max_pauses=max_pauses)
    return store, case, result


MARIA = [
    step("read_document", {"slot": "pay_stub"}, "c1"),
    ask(),
    wait(),
    # Maria replies 2 days later with her new pay stub.
    step("read_document", {"slot": "pay_stub"}, "c2"),
    step("verify_identity"),
    step("mark_income_verified", {"slot": "pay_stub", "reason": "New stub, new rate."}),
    step("finish", FINISH_COMPLETE),
]


def test_maria_replies_and_the_case_completes():
    store, case, result = simulate("02_maria_job_change", MARIA)
    stored = store.get(result.application_id)
    assert stored.ending.outcome == "complete"
    assert result.pauses == 1
    assert not result.stopped_by_limit


def test_the_reply_is_shown_to_the_agent():
    _, _, result = simulate("02_maria_job_change", MARIA)
    reply = next(m for m in result.messages if "the customer replied" in m.content)
    assert reply.content.splitlines() == [
        "Today is 2026-09-16. 2 days later, the customer replied.",
        'Message: "Here is my latest pay stub with the new pay rate."',
        "Uploaded: pay_stub",
    ]


def test_the_reply_upload_reaches_the_store_two_days_later():
    store, case, result = simulate("02_maria_job_change", MARIA)
    stored = store.get(result.application_id)
    assert stored.document_versions["pay_stub"] == 2  # the new upload
    upload = next(e for e in stored.audit_log if e.event == "document_uploaded")
    assert upload.actor == "customer"
    asked = stored.messages[0].sent_at
    assert upload.at == asked + timedelta(days=2)


def test_robert_does_not_answer_and_the_case_is_cancelled():
    answers = [
        ask(),
        wait(),
        # 26 days later, still no answer.
        step("finish", {"outcome": "cancelled", "reason": "no_answer", "explanation": "No reply."}),
    ]
    store, case, result = simulate("03_no_answer", answers)
    stored = store.get(result.application_id)
    assert stored.ending.reason == "no_answer"  # allowed: 26 days have passed
    assert stored.ending.ended_at >= stored.messages[0].sent_at + timedelta(days=26)
    news = [m.content for m in result.messages if "days have passed" in m.content]
    assert news == [
        "Today is 2026-10-10. 26 days have passed since the last message or reply, "
        "with no answer from the customer."
    ]


def test_without_a_new_message_the_customer_has_nothing_to_answer():
    # Maria's script has a reply, but the agent waits without asking anything.
    handoff = {"outcome": "sent_to_human", "reason": "agent_unsure", "explanation": "Unsure."}
    answers = [wait(), step("finish", handoff)]
    store, _, result = simulate("02_maria_job_change", answers)
    assert "days have passed" in result.messages[2].content
    assert store.get(result.application_id).document_versions["pay_stub"] == 1


def test_the_safety_limit_stops_an_agent_that_keeps_waiting():
    answers = [wait() for _ in range(5)]
    store, _, result = simulate("03_no_answer", answers, max_pauses=3)
    assert result.stopped_by_limit
    assert result.pauses == 3
    assert store.get(result.application_id).ending is None


def test_a_case_that_needs_no_messages_has_no_pauses():
    answers = [
        step("verify_identity"),
        step("mark_income_verified", {"slot": "pay_stub", "reason": "Matches."}),
        step("finish", FINISH_COMPLETE),
    ]
    store, _, result = simulate("01_clean_application", answers)
    assert result.pauses == 0
    assert store.get(result.application_id).ending.outcome == "complete"


def test_the_clock_moves_forward_and_never_back():
    start = load_case(CASES / "01_clean_application").info.start_time
    clock = SimulatedClock(start)
    assert clock() == start
    assert clock() == start + timedelta(minutes=1)
    clock.jump_to(start + timedelta(days=2))
    assert clock() == start + timedelta(days=2)
    clock.jump_to(start)  # backward: ignored
    assert clock() == start + timedelta(days=2, minutes=1)


def test_marias_reply_is_recorded_with_its_upload():
    store, case, result = simulate("02_maria_job_change", MARIA)
    stored = store.get(result.application_id)
    [reply] = stored.replies
    assert reply.message == "Here is my latest pay stub with the new pay rate."
    assert reply.uploads == ["pay_stub"]
    events = [(e.actor, e.event) for e in stored.audit_log if e.actor == "customer"]
    assert events == [("customer", "customer_replied"), ("customer", "document_uploaded")]
    replied, uploaded = [e for e in stored.audit_log if e.actor == "customer"]
    assert replied.at == uploaded.at  # the reply and its attachment arrive together


def test_after_a_reply_without_documents_the_wait_counts_from_the_reply():
    # Maria answers 2 days later, but the agent then waits without asking again:
    # it is woken 26 days after her reply, so cancelling is allowed.
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    case.customer.replies[0].uploads = {}  # she replies but sends nothing
    cancel = {"outcome": "cancelled", "reason": "documents_not_provided", "explanation": "x"}
    answers = [ask(), wait(), wait(), step("finish", cancel)]
    result = simulate_case(case, store, FakeModel(messages=iter(answers)))
    stored = store.get(result.application_id)
    assert stored.ending.reason == "documents_not_provided"
    assert stored.ending.ended_at >= stored.replies[0].received_at + timedelta(days=26)
    assert result.pauses == 2


def test_on_message_receives_every_step_in_order():
    seen = []
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    simulate_case(case, store, FakeModel(messages=iter(MARIA)), on_message=seen.append)
    kinds = [m.type for m in seen]
    # start, then (agent call, tool result) x2, the wait text, the news, ...
    assert kinds[:7] == ["human", "ai", "tool", "ai", "tool", "ai", "human"]
    assert seen[6].content.startswith("Today is 2026-09-16. 2 days later, the customer replied.")
    assert seen[-1].type == "tool" and seen[-1].name == "finish"


def test_streaming_gives_the_same_result_as_without():
    store_a, store_b = ApplicationStore(), ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    a = simulate_case(case, store_a, FakeModel(messages=iter(MARIA)))
    b = simulate_case(case, store_b, FakeModel(messages=iter(MARIA)), on_message=lambda m: None)
    assert [m.content for m in a.messages] == [m.content for m in b.messages]


def test_on_token_receives_the_agents_words_while_it_writes():
    words = []
    store = ApplicationStore()
    case = load_case(CASES / "03_no_answer")
    simulate_case(
        case,
        store,
        FakeModel(messages=iter([wait("Waiting for the pay stub.")])),
        max_pauses=0,  # stop at the first pause; the words are what matters here
        on_token=lambda kind, text: words.append((kind, text)),
    )
    assert "".join(text for kind, text in words if kind == "answer") == "Waiting for the pay stub."
    assert len(words) > 1  # in pieces, as written, not all at once
