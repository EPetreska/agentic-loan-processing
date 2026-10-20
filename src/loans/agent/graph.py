"""The agent loop, built with LangGraph.

The LLM chooses a tool, the tool runs, the LLM sees the result and chooses
the next step. The loop stops when the case has ended (finish succeeded), or
when the LLM replies without calling a tool, for example to wait for the
customer.

Which model runs is a setting (LLM_MODEL, for example "ollama:qwen3:4b" or
"anthropic:claude-sonnet-5-5"), so switching models needs no code change.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from loans.agent.prompt import SYSTEM_PROMPT
from loans.store import ApplicationStore

# The context window for local Ollama models, in tokens. Ollama otherwise uses
# a small default and silently cuts off the start of a long conversation
# (which can include the instructions and the tool descriptions). 16,384
# holds a whole case with room to spare and fits in memory on a CPU-only
# machine; the instructions and tools alone are about 2,100 tokens.
OLLAMA_CONTEXT_TOKENS = 16_384


def make_model(spec: str) -> BaseChatModel:
    """The chat model for a setting such as "ollama:qwen3:4b"."""
    provider, _, name = spec.partition(":")
    if provider == "ollama" and name:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, num_ctx=OLLAMA_CONTEXT_TOKENS)
    if provider == "anthropic" and name:
        from langchain_anthropic import ChatAnthropic  # added when switching

        return ChatAnthropic(model=name)
    raise ValueError(f"unknown model setting {spec!r}, expected 'ollama:<name>' or 'anthropic:<name>'")


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    store: ApplicationStore,
    application_id: str,
) -> CompiledStateGraph:
    """The loop for one case."""
    model_with_tools = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        response = model_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def after_agent(state: MessagesState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: MessagesState) -> str:
        return END if store.get(application_id).ending is not None else "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    # A tool call the LLM gets wrong (for example an unknown slot) is returned
    # to it as an error message, so it can correct itself.
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()
