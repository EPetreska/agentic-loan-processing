"""Tests for counting a run's LLM calls and tokens, and pricing them."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from loans.evals.pricing import Usage, cost_usd, usage_of


def answer(input_tokens, output_tokens):
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_calls_and_tokens_are_counted_from_the_model_answers():
    conversation = [
        HumanMessage("Process the application."),
        answer(1_000, 50),
        ToolMessage("form complete: yes", tool_call_id="call-1"),
        answer(1_200, 80),
    ]
    assert usage_of(conversation) == Usage(llm_calls=2, input_tokens=2_200, output_tokens=130)


def test_an_answer_without_token_counts_still_counts_as_a_call():
    assert usage_of([AIMessage(content="Waiting.")]) == Usage(llm_calls=1)


def test_a_local_model_costs_nothing():
    assert cost_usd("ollama:qwen3:4b", Usage(llm_calls=5, input_tokens=90_000, output_tokens=4_000)) == 0.0


def test_claude_is_priced_per_million_tokens():
    # Claude Sonnet 5.5: $2 per million input tokens, $10 per million output tokens.
    usage = Usage(llm_calls=3, input_tokens=500_000, output_tokens=20_000)
    assert cost_usd("anthropic:claude-sonnet-5-5", usage) == 1.2


def test_a_model_without_a_price_has_an_unknown_cost():
    assert cost_usd("anthropic:claude-unknown", Usage(input_tokens=10)) is None
    assert cost_usd("fake:model", Usage()) is None
