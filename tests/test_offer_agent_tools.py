"""Tests for the offer agent's tools as the LLM sees them (the plain tools
are tested in their own files)."""

from datetime import timedelta
from pathlib import Path

import pytest

from loans.agents.offer.tools import make_offer_tools
from loans.evals.case import load_case, submit_case
from loans.offers import offer_for
from loans.rag.search import TOP_K, SearchResult
from loans.store import ApplicationStore
from loans.tools.messages import MAX_MESSAGE_LENGTH
from loans.tools.offer.request_approval import MAX_REQUEST_LENGTH
from loans.tools.offer.retrieve_documents import MAX_QUERY_LENGTH

CASES = Path(__file__).parent.parent / "evals" / "cases"
SECTION = SearchResult(chunk_id="bank-07", document="bank", title="Paying early",
                       text="No prepayment penalty.", source="the bank's own document", distance=0.1)  # fmt: skip


@pytest.fixture
def offered():
    """Maria's case with her offer made, and her tools wired for it."""
    store = ApplicationStore()
    case = load_case(CASES / "02_maria_job_change")
    app_id = submit_case(store, case)
    store.record_offer(app_id, offer_for(case.application), case.info.start_time)
    now = case.info.start_time + timedelta(days=1)
    queries = []

    def search(query):
        queries.append(query)
        return [SECTION]

    tools = {t.name: t for t in make_offer_tools(store, app_id, lambda: now, search)}
    return store, app_id, tools, queries


def call(tool, args, call_id="call-1"):
    """A tool call as LangGraph makes it: the call ID comes with the call."""
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": call_id}).content


def test_the_llm_sees_only_its_own_parameters(offered):
    _, _, tools, _ = offered
    params = {name: set(t.tool_call_schema.model_json_schema()["properties"]) for name, t in tools.items()}
    assert params == {
        "retrieve_documents": {"query"},
        "request_approval": {"request", "reason"},
        "send_message": {"text", "reason"},
    }


def test_the_descriptions_state_the_real_limits(offered):
    _, _, tools, _ = offered
    assert f"Returns the {TOP_K} closest sections" in tools["retrieve_documents"].description
    assert f"longer than {MAX_QUERY_LENGTH} characters" in tools["retrieve_documents"].description
    assert f"longer than {MAX_REQUEST_LENGTH} characters" in tools["request_approval"].description
    assert f"longer than {MAX_MESSAGE_LENGTH} characters" in tools["send_message"].description
    for tool in tools.values():
        assert "after the offer phase has closed" in tool.description


def test_the_search_runs_on_the_case_with_the_query(offered):
    store, app_id, tools, queries = offered
    result = call(tools["retrieve_documents"], {"query": "prepayment penalty"})
    assert queries == ["prepayment penalty"]
    assert "[section 1] Paying early" in result
    [search] = store.get(app_id).document_retrievals
    assert (search.retrieval_id, search.chunk_ids) == ("call-1", ["bank-07"])


def test_a_request_reaches_the_officer_with_the_call_id(offered):
    store, app_id, tools, _ = offered
    args = {"request": "Borrow $17,000 instead of $15,000.", "reason": "More than offered."}
    assert call(tools["request_approval"], args).startswith("sent to a loan officer (request 1)")
    assert call(tools["request_approval"], args) == call(tools["request_approval"], args)  # a retry
    [request] = store.get(app_id).approval_requests
    assert request.request_id == "call-1"


def test_a_message_reaches_the_customer_once(offered):
    store, app_id, tools, _ = offered
    args = {"text": "There is no fee for paying early.", "reason": "She asked about early repayment."}
    first = call(tools["send_message"], args)
    assert call(tools["send_message"], args) == first == "message sent (message 1 in this case)"
    assert [m.message_id for m in store.get(app_id).messages] == ["call-1"]
