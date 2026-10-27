"""What a run cost: the LLM calls and tokens counted from the agents'
conversations, priced from a table of model prices.

LangChain attaches usage_metadata (input and output tokens) to every answer
a chat model gives, for Claude and for Ollama alike, so the counts come from
the conversations a run already returns, with no extra calls. A model that
runs locally (Ollama) costs nothing per token; a model missing from the table
has an unknown cost, never a guessed one.
"""

from collections.abc import Iterable
from decimal import Decimal

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

# USD per million input and output tokens, by model setting (LLM_MODEL).
# Anthropic's first-party API rates, checked on 2026-10-26 (Anthropic's model
# table of 2026-10-06). The project does not use prompt caching, so cached
# token prices are not needed. Claude Haiku 5.5's rate holds for prompts up
# to 100K tokens, far above this project's prompts.
PRICES_PER_MILLION_TOKENS = {
    "anthropic:claude-fable-5-1": (Decimal("10.00"), Decimal("50.00")),
    "anthropic:claude-opus-5-5": (Decimal("4.00"), Decimal("20.00")),
    "anthropic:claude-sonnet-5-5": (Decimal("2.00"), Decimal("10.00")),
    "anthropic:claude-haiku-5-5": (Decimal("0.10"), Decimal("0.50")),
}

# Providers whose models run on this machine: no cost per token.
LOCAL_PROVIDERS = {"ollama"}


class Usage(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def usage_of(messages: Iterable[BaseMessage]) -> Usage:
    """The LLM calls and tokens in a conversation: one call per answer the
    model gave, with the tokens LangChain recorded for it."""
    usage = Usage()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage.llm_calls += 1
        if message.usage_metadata:
            usage.input_tokens += message.usage_metadata.get("input_tokens", 0)
            usage.output_tokens += message.usage_metadata.get("output_tokens", 0)
    return usage


def cost_usd(model_spec: str, usage: Usage) -> float | None:
    """What the tokens cost, in USD: 0 for a local model, None if the model's
    price is not in the table."""
    if model_spec.partition(":")[0] in LOCAL_PROVIDERS:
        return 0.0
    prices = PRICES_PER_MILLION_TOKENS.get(model_spec)
    if prices is None:
        return None
    input_price, output_price = prices
    cost = (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
    return float(round(cost, 6))
