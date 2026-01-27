try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import inspect
from rich.panel import Panel
from operator import add
from typing import Callable, TypedDict, List, Type, Dict, Any, Optional, Union, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from observix.llm import get_llm

from observix.agents.utils.logging import get_logger
from observix.agents.exceptions import ConfigurationError, WorkflowError
from observix.instrumentation import observe, capture_candidate_agents, capture_tools, init_observability


load_dotenv()
logger = get_logger(__name__)

# ===================== TOOL =====================

class Tool:
    """
    A wrapper around functions or LangChain tools for use in Observix Agents workflows.
    """
    def __init__(
        self,
        fn: Union[Callable[..., Any], BaseTool],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize a Tool.

        Args:
            fn: A callable function or a LangChain BaseTool.
            name: The name of the tool (required if fn is a function).
            description: The description of the tool (required if fn is a function).
        """
        if isinstance(fn, BaseTool):
            self._tool = fn
            self.name = fn.name
            self.description = fn.description
            
            # Wrap invoke for observability
            original_invoke = fn.invoke
            fn.invoke = observe(name=self.name, as_tool=True)(original_invoke)
            
            self.fn = fn.invoke
            self.async_fn = getattr(fn, "ainvoke", None)
            if self.async_fn:
                 original_ainvoke = fn.ainvoke
                 fn.ainvoke = observe(name=self.name, as_tool=True)(original_ainvoke)
                 self.async_fn = fn.ainvoke
                 
            self.is_langchain_tool = True

        elif callable(fn):
            if not name or not description:
                raise ConfigurationError("Custom Tool requires name and description")

            self.name = name
            self.description = description
            self.is_langchain_tool = False

            # Instrument function immediately
            instrumented_fn = observe(name=name, as_tool=True)(fn)
            
            if inspect.iscoroutinefunction(fn):
                self.fn = None
                self.async_fn = instrumented_fn
            else:
                self.fn = instrumented_fn
                self.async_fn = None

            self._tool = StructuredTool.from_function(
                func=self.fn if self.fn else self.async_fn, # Use instrumented wrapper
                coroutine=self.async_fn,
                name=name,
                description=description,
            )
        else:
            raise ConfigurationError("Tool fn must be callable or BaseTool")

    def _instrument_tool(self):
        # Deprecated: Logic moved to __init__
        pass

    def as_langchain_tool(self) -> BaseTool:
        """Returns the underlying LangChain tool."""
        return self._tool

    def log_call(self, tool_name: str, tool_args: Any, tool_result: Any) -> None:
        """Logs the tool call details."""
        logger.info(f"🔧 Tool Call: {tool_name}")
        logger.debug(f"Args: {tool_args}")
        logger.debug(f"Result: {tool_result}")


# ===================== AGENT =====================

class Agent:
    """
    Represents an AI agent capable of executing instructions and using tools.
    """
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        input_key: str,
        output_key: str,
        tools: Optional[List[Tool]] = None,
        model: str = "openai/gpt-4o",
        framework: str = "langchain",
        temperature: float = 0.0,
    ):
        """
        Initialize an Agent.

        Args:
            name: Human-readable name of the agent.
            description: Description of what the agent does.
            instructions: System instructions for the LLM.
            input_key: Key in the state dictionary containing the input.
            output_key: Key in the state dictionary where the result will be stored.
            tools: List of Tool objects available to the agent.
            model: Name of the LLM model to use.
            framework: LLM framework to use (default "langchain").
            temperature: Sampling temperature for the LLM.
        """
        self.name = name
        self.description = description
        self.instructions = instructions
        self.input_key = input_key
        self.output_key = output_key
        self.tools = tools or []

        # Auto-initialize observability
        init_observability()

        self.llm = get_llm(model=model, framework=framework, temperature=temperature)

        self.tool_map: Dict[str, Tool] = {}
        lc_tools = []

        for tool in self.tools:
            lc_tool = tool.as_langchain_tool()
            self.tool_map[lc_tool.name] = tool
            lc_tools.append(lc_tool)

        if lc_tools:
            self.llm = self.llm.bind_tools(lc_tools)

        self.system_prompt = self._build_prompt()

    def _build_prompt(self) -> str:
        return f"""
Agent Name: {self.name}
Description: {self.description}

Instructions:
{self.instructions}
"""

    @observe(as_agent=True)
    def __call__(self, state: dict) -> dict:
        """Synchronously invoke the agent."""
        if self.input_key not in state:
            raise WorkflowError(f"Input key '{self.input_key}' not found in state for agent '{self.name}'")

        # Capture tools for this agent's span
        capture_tools([{"name": t.name, "description": t.description} for t in self.tools])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=str(state[self.input_key])),
        ]

        logger.info(f"🤖 Agent '{self.name}' is thinking...")
        response = self.llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            call = response.tool_calls[0]
            tool = self.tool_map[call["name"]]
            args = call["args"]

            logger.info(f"🛠️ Agent '{self.name}' is using tool '{tool.name}'")
            result = tool.as_langchain_tool().invoke(args)
            tool.log_call(call["name"], args, result)

            return {self.output_key: str(result)}

        return {self.output_key: response.content}

    @observe(as_agent=True)
    async def ainvoke(self, state: dict) -> dict:
        """Asynchronously invoke the agent."""
        if self.input_key not in state:
            raise WorkflowError(f"Input key '{self.input_key}' not found in state for agent '{self.name}'")

        # Capture tools for this agent's span
        capture_tools([{"name": t.name, "description": t.description} for t in self.tools])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=str(state[self.input_key])),
        ]

        logger.info(f"🤖 Agent '{self.name}' is thinking (async)...")
        response = await self.llm.ainvoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            call = response.tool_calls[0]
            tool = self.tool_map[call["name"]]
            args = call["args"]

            logger.info(f"🛠️ Agent '{self.name}' is using tool '{tool.name}' (async)")
            if tool.async_fn:
                result = await tool.as_langchain_tool().ainvoke(args)
            else:
                result = tool.as_langchain_tool().invoke(args)

            tool.log_call(call["name"], args, result)
            return {self.output_key: str(result)}

        return {self.output_key: response.content}


# ===================== HUMAN NODE =====================

class HumanNode:
    """
    A node in the graph that waits for human intervention.
    """
    def __init__(self, name: str, prompt: str, output_key: str):
        """
        Initialize a HumanNode.

        Args:
            name: Name of the node.
            prompt: Prompt message to show to the human user.
            output_key: Key in the state dictionary where the feedback will be stored.
        """
        self.name = name
        self.prompt = prompt
        self.output_key = output_key

    def __call__(self, state: dict) -> dict:
        """
        Node execution. Expects 'human_feedback' to be injected by Graph.resume().
        """
        feedback = state.get("human_feedback")
        if feedback is None:
            logger.warning(f"HumanNode '{self.name}' executed but no 'human_feedback' found in state.")
        return {self.output_key: feedback}


# ===================== GRAPH =====================

class Graph:
    """
    A workflow graph built with LangGraph.
    """
    def __init__(
        self,
        state: Type[TypedDict],  # type: ignore
        name: Optional[str] = None,
        thread_id: str = "default",
        max_retries: int = 3,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        enable_memory: bool = True,
    ):
        """
        Initialize a Graph.

        Args:
            state: A TypedDict class defining the graph state.
            name: Optional name for the graph (useful for subgraphs).
            thread_id: Identifier for persistence across runs.
            max_retries: Maximum retries for conditional loops.
            checkpointer: A custom LangGraph checkpointer (e.g., Postgres).
            enable_memory: Whether to enable state persistence (defaults to MemorySaver if checkpointer is None).
        """
        self.state = state
        self.name = name or f"graph_{id(self)}"
        self.thread_id = thread_id
        self.max_retries = max_retries

        # Auto-initialize observability
        init_observability()

        self.nodes: List[Any] = []
        self.edges: List[tuple[str, str]] = []
        self.conditions: Dict[str, Dict[str, Any]] = {}
        self.human_nodes: List[str] = []

        self._pending_condition_node: Optional[str] = None
        self._compiled_graph = None

        if checkpointer:
            self.checkpointer = checkpointer
        elif enable_memory:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None

    # ---------- DSL ----------

    def add(self, node: Any) -> 'Graph':
        """
        Adds a node or a subgraph to the graph.

        Args:
            node: An Agent, HumanNode, or another Graph instance.
        """
        self.nodes.append(node)
        if hasattr(node, "name") and isinstance(node, HumanNode):
            self.human_nodes.append(node.name)
        return self

    def edge(self, from_node: str, to_node: str) -> 'Graph':
        """Explicitly defines an edge between two nodes."""
        self.edges.append((from_node, to_node))
        return self

    def when(self, node_name: str, condition_fn: Callable[[dict], str]) -> 'Graph':
        """Defines a conditional transition from a node."""
        self._pending_condition_node = node_name
        self.conditions[node_name] = {"fn": condition_fn, "routes": {}}
        return self

    def then(self, routes: Dict[str, str]) -> 'Graph':
        """Defines the routes for a previous 'when' condition."""
        if not self._pending_condition_node:
            raise ConfigurationError("then() must be called after when()")

        self.conditions[self._pending_condition_node]["routes"] = routes
        self._pending_condition_node = None
        return self

    # ---------- BUILD ----------

    def _build_graph(self) -> None:
        if not self.nodes:
            raise ConfigurationError("Graph must have at least one node.")

        graph = StateGraph(self.state)

        for node in self.nodes:
            if isinstance(node, Graph):
                # Build and compile subgraph
                if not node._compiled_graph:
                    node._build_graph()
                graph.add_node(node.name, node._compiled_graph)
            else:
                graph.add_node(node.name, node)

        graph.set_entry_point(getattr(self.nodes[0], "name", self.nodes[0].name))

        # 1. Process explicit edges (allows parallel fan-out/fan-in)
        for from_node, to_node in self.edges:
            target = END if to_node == "END" else to_node
            graph.add_edge(from_node, target)

        # 2. Process conditional edges
        for name, cond in self.conditions.items():
            routes = {
                key: (END if value == "END" else value)
                for key, value in cond["routes"].items()
            }
            graph.add_conditional_edges(name, cond["fn"], routes)

        # 3. Apply default sequential logic ONLY if no edges/conditions are defined for a node
        # This maintains the "simple DSL" feel while allowing power user overrides
        explicit_from_nodes = set(e[0] for e in self.edges) | set(self.conditions.keys())
        
        for i, node in enumerate(self.nodes):
            name = node.name
            if name not in explicit_from_nodes:
                if i < len(self.nodes) - 1:
                    graph.add_edge(name, self.nodes[i + 1].name)
                else:
                    graph.add_edge(name, END)

        self._compiled_graph = graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self.human_nodes,
        )

    # ---------- VISUALIZE ----------

    def visualize(self) -> None:
        """Visualizes the graph using Mermaid."""
        if not self._compiled_graph:
            self._build_graph()

        try:
            from rich.console import Console
            console = Console()
            console.print(
                Panel.fit(
                    self._compiled_graph.get_graph().draw_mermaid(),
                    title=f"WORKFLOW: {self.name.upper()}",
                    style="bold cyan",
                )
            )
        except Exception as e:
            logger.error(f"Failed to visualize graph: {e}")

    # ---------- RUN ----------

    @observe(name="Graph.run")
    def run(self, input_value: str, **kwargs) -> dict:
        """Synchronously starts the graph execution."""
        if not self._compiled_graph:
            self._build_graph()

        # Capture all agents and tools in the graph
        all_agents = []
        all_tools = []
        for node in self.nodes:
            if isinstance(node, Agent):
                all_agents.append({"name": node.name, "description": node.description})
                for tool in node.tools:
                    all_tools.append({"name": tool.name, "description": tool.description})
        
        capture_candidate_agents(all_agents)
        capture_tools(all_tools)

        state = {"input": input_value, "retries": 0, **kwargs}
        logger.info(f"🚀 Starting workflow (thread: {self.thread_id})")

        return self._compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": self.thread_id}},
        )

    @observe(name="Graph.run_async")
    async def run_async(self, input_value: str, **kwargs) -> dict:
        """Asynchronously starts the graph execution."""
        if not self._compiled_graph:
            self._build_graph()

        # Capture all agents and tools in the graph
        all_agents = []
        all_tools = []
        for node in self.nodes:
            if isinstance(node, Agent):
                all_agents.append({"name": node.name, "description": node.description})
                for tool in node.tools:
                    all_tools.append({"name": tool.name, "description": tool.description})
        
        capture_candidate_agents(all_agents)
        capture_tools(all_tools)

        state = {"input": input_value, "retries": 0, **kwargs}
        logger.info(f"🚀 Starting workflow async (thread: {self.thread_id})")

        return await self._compiled_graph.ainvoke(
            state,
            config={"configurable": {"thread_id": self.thread_id}},
        )

    # ---------- STREAM ----------

    def stream(self, input_value: str) -> Any:
        """Starts the graph execution and yields updates."""
        if not self._compiled_graph:
            self._build_graph()

        state = {"input": input_value, "retries": 0}
        logger.info(f"🌊 Streaming workflow (thread: {self.thread_id})")

        return self._compiled_graph.stream(
            state,
            config={"configurable": {"thread_id": self.thread_id}},
        )

    def astream(self, input_value: str) -> Any:
        """Asynchronously starts the graph execution and yields updates."""
        if not self._compiled_graph:
            self._build_graph()

        state = {"input": input_value, "retries": 0}
        logger.info(f"🌊 Streaming workflow async (thread: {self.thread_id})")

        return self._compiled_graph.astream(
            state,
            config={"configurable": {"thread_id": self.thread_id}},
        )

    # ---------- RESUME ----------

    def resume(self, human_input: str) -> dict:
        """Resumes the graph execution after an interrupt."""
        if not self._compiled_graph:
            raise WorkflowError("Graph must be run before it can be resumed.")

        config = {"configurable": {"thread_id": self.thread_id}}
        logger.info(f"⏭️ Resuming workflow (thread: {self.thread_id})")
        self._compiled_graph.update_state(config, {"human_feedback": human_input})
        return self._compiled_graph.invoke(None, config=config)

    async def resume_async(self, human_input: str) -> dict:
        """Asynchronously resumes the graph execution after an interrupt."""
        if not self._compiled_graph:
            raise WorkflowError("Graph must be run before it can be resumed.")

        config = {"configurable": {"thread_id": self.thread_id}}
        logger.info(f"⏭️ Resuming workflow async (thread: {self.thread_id})")
        await self._compiled_graph.aupdate_state(config, {"human_feedback": human_input})
        return await self._compiled_graph.ainvoke(None, config=config)

# ===================== HELPERS =====================

def MessagesState():
    """Returns a state TypedDict with a 'messages' list that appends instead of overwriting."""
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], add]
    return State
