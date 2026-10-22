"""The offer agent's loop: the shared loop (loans.agents.loop) with the offer
agent's prompt, its three tools for one case, and its stopping rule: the run
never ends after a tool step.

The offer agent has no tool that ends anything. The offer phase is closed by
the workflow (accepted on the customer's signature, expired after
policy.OFFER_VALID_DAYS days), so the agent only ever pauses to wait for the
customer or a loan officer. Once the phase has closed, the workflow ends the
paused run with end_run(closed_turn(...)): in code, without calling the LLM,
since the agent has nothing left to decide.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from loans.agents.loop import build_graph
from loans.agents.offer.prompt import SYSTEM_PROMPT


def never_ends(messages: list[BaseMessage]) -> bool:
    """The stopping rule: no tool step ends the run, whatever a tool answers."""
    return False


def build_offer_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """The offer agent's loop for one case, with that case's tools."""
    return build_graph(model, tools, SYSTEM_PROMPT, never_ends, checkpointer)
