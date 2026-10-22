"""The application agent's loop: the shared loop (loans.agents.loop) with the
application agent's prompt, its six tools for one case, and its stopping
rule: the run ends when finish answers that the case ended.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from loans.agents.application.prompt import SYSTEM_PROMPT
from loans.agents.loop import build_graph
from loans.tools.application.finish import ENDED


def case_ended(messages: list[BaseMessage]) -> bool:
    """True if the last tool step ended the case: finish answered that it
    did. Only finish's own answer counts, not text from the LLM or from
    another tool (a document cannot end the case by containing the words)."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            return False  # the tool step's answers are all at the end
        if message.name == "finish" and str(message.content).startswith(f"{ENDED}:"):
            return True
    return False


def build_application_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """The application agent's loop for one case, with that case's tools."""
    return build_graph(model, tools, SYSTEM_PROMPT, case_ended, checkpointer)
