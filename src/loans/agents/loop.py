"""The agent loop, built with LangGraph: the loop both agents run (the
application agent and the offer agent, in loans.agents.application and
loans.agents.offer), each with its own prompt, tools and stopping rule.

The LLM chooses a tool, the tool runs, the LLM sees the result and chooses
the next step. When the LLM replies without calling a tool (for example to
wait for the customer), the run pauses in the wait box (LangGraph's
interrupt), and the checkpointer saves the conversation under the case's
thread ID (one per case and agent). The caller resumes it later with what
happened in the meantime (the customer's reply, time passing, or a loan
officer's decision), which the agent receives as a new message. A paused
run can also be ended by its caller (end_run), without calling the LLM, when
the work it waits for is over (the offer agent's run, once the offer phase
has closed); its closing message is kept as the last message. Whether the loop ends after a tool step is
the agent's own stopping rule (the application agent stops when finish
answers that the case ended).

The loop decides everything from the conversation alone (the LLM's answers
and the tools' answers) and never reads the bank's data itself: the agent
reaches the bank only through its tools, which with MCP run on the server.

The checkpointer is in memory for now, so a paused case lives as long as the
program; a real deployment passes a database checkpointer (Postgres) so a
case can be resumed days later, after a restart.

Which model runs is a setting (LLM_MODEL, for example "ollama:qwen3:4b" or
"anthropic:claude-sonnet-5-5"), so switching models needs no code change.
"""

from collections.abc import Callable
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt


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

        # reasoning=True returns a thinking model's reasoning separately
        # (additional_kwargs["reasoning_content"]) instead of mixed into the
        # answer between <think> tags, so the answer stays clean and the
        # reasoning can be shown on its own (run.py --show).
        return ChatOllama(model=name, num_ctx=OLLAMA_CONTEXT_TOKENS, reasoning=True)
    if provider == "anthropic" and name:
        from langchain_anthropic import ChatAnthropic  # added when switching

        return ChatAnthropic(model=name)
    raise ValueError(f"unknown model setting {spec!r}, expected 'ollama:<name>' or 'anthropic:<name>'")


def thread_config(case_thread: str, agent: Literal["application", "offer"]) -> dict:
    """The run settings that point LangGraph at one agent's saved
    conversation on one case (case_thread is usually the application ID).
    Each agent has its own thread, for example "app-123__offer": both agents
    work on the same case, and with a shared checkpointer one thread would
    make the offer agent continue the application agent's conversation.

    The thread ID never contains "/": LangGraph passes it to LangSmith, whose
    thread page puts it in a web address, and a "/" there breaks the page."""
    if "/" in case_thread:
        raise ValueError(f"a thread name cannot contain '/': {case_thread!r}")
    return {"configurable": {"thread_id": f"{case_thread}__{agent}"}}


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    ends_after_tools: Callable[[list[BaseMessage]], bool],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """The loop for one case: the agent's prompt, its tools for that case,
    and its stopping rule, which is asked after each tool step whether the
    run ends there (given the conversation). Runs need a thread_config in
    their config; without a checkpointer given, one in memory is used."""
    model_with_tools = model.bind_tools(tools)

    async def agent(state: MessagesState) -> dict:
        response = await model_with_tools.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [response]}

    def after_agent(state: MessagesState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "wait"

    def wait(state: MessagesState) -> Command:
        # Pauses the run here. When it is resumed, interrupt returns what the
        # caller passed in (Command(resume=...)), and the agent gets it as a
        # new message; a closing (end_run) is kept and ends the run instead.
        news = interrupt({"waiting": state["messages"][-1].content})
        if isinstance(news, dict) and _CLOSING in news:
            return Command(update={"messages": [HumanMessage(news[_CLOSING])]}, goto=END)
        return Command(update={"messages": [HumanMessage(news)]}, goto="agent")

    def after_tools(state: MessagesState) -> str:
        return END if ends_after_tools(state["messages"]) else "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    # A tool call the LLM gets wrong (for example an unknown slot) is returned
    # to it as an error message, so it can correct itself.
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("wait", wait, destinations=("agent", END))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "wait"])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# The key that marks a resume as a closing. A plain dict, not a class, so a
# database checkpointer can save it like any other resume value.
_CLOSING = "closing"


def end_run(closing: str) -> Command:
    """What ends a paused run instead of resuming it: the closing message is
    added to the conversation, and the run ends without calling the LLM."""
    return Command(resume={_CLOSING: closing})


async def run_until_pause(
    graph: CompiledStateGraph,
    run_input,
    config: dict,
    on_message: Callable[[BaseMessage], None] | None = None,
    on_token: Callable[[str, str], None] | None = None,
) -> list[BaseMessage]:
    """Run the agent until it pauses (to wait) or ends, and return the whole
    conversation. run_input starts a run ({"messages": [...]}) or resumes a
    paused one (Command(resume=news)). With on_message or on_token, the run is streamed: finished messages step
    by step ("updates"), and the model's words as it writes them
    ("messages")."""
    if on_message is None and on_token is None:
        await graph.ainvoke(run_input, config)
        return (await graph.aget_state(config)).values["messages"]

    modes = ["updates", "messages"] if on_token else ["updates"]
    async for mode, data in graph.astream(run_input, config, stream_mode=modes):
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
    return (await graph.aget_state(config)).values["messages"]
