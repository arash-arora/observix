import os
import time
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict, Optional

from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from observix.llm.langchain import ChatGroq
from observix import (
    observe,
    capture_context,
    get_current_observation_id,
    observation_context,
)

# =========================
# ENV + LLM
# =========================

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in env.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

# =========================
# TOOLS
# =========================


@observe(name="google_search", as_tool=True)
def google_search(query: str):
    """Performs a Google search to retrieve latest trends and information."""
    print(f"  [Tool] Searching Google for: {query}")
    capture_context(
        "Kubernetes is an open source container orchestration engine for automating deployment, scaling, and management of containerized applications."
    )
    time.sleep(0.5)
    return f"Search results for {query}: [Trend A, Trend B, Factor C]"


@observe(name="cms_upload", as_tool=True)
def cms_upload(content: str):
    """Uploads the finalized content to the Content Management System."""
    print("  [Tool] Uploading content to CMS...")
    time.sleep(0.5)
    return "Upload Successful (ID: 12345)"


# =========================
# STATE
# =========================


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    observation_id: Optional[int]


# =========================
# AGENTS
# =========================


# 1. Planner
@observe(name="planner_agent", as_agent=True)
def planner_node(state: AgentState):
    """Responsible for creating the initial content outline."""
    print("\n--- Planner Agent ---")

    response = llm.invoke(
        [
            SystemMessage(content="You are a Content Planner. Create a brief outline."),
            *state["messages"],
        ]
    )

    return {"messages": [response]}


# 2. Researcher (LLM decides tool call)
@observe(name="researcher_agent", as_agent=True)
def researcher_node(state: AgentState):
    """Conducts research on the given topic using search tools."""
    print("\n--- Researcher Agent ---")

    llm_with_tools = llm.bind_tools([google_search])

    response = llm_with_tools.invoke(
        [
            SystemMessage(
                content="You are a Researcher. Decide what to search. Use google_search tool when needed."
            ),
            state["messages"][-1],
        ]
    )

    messages = [response]

    # Check for tool calls and execute them immediately
    if response.tool_calls:
        tools_map = {"google_search": google_search, "cms_upload": cms_upload}
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            if name in tools_map:
                try:
                    # Execute tool (context automatically propagated since we are in same thread/span)
                    tool_result = tools_map[name](**args)
                except Exception as e:
                    tool_result = f"Error executing tool {name}: {str(e)}"
            else:
                tool_result = f"Error: Tool {name} not found."

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=call_id, name=name)
            )

    return {"messages": messages}


# 3. Writer
@observe(name="writer_agent", as_agent=True)
def writer_node(state: AgentState):
    """Drafts the blog post content based on research findings."""
    print("\n--- Writer Agent ---")

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a Writer. Write a short blog post based on the research."
            ),
            state["messages"][-1],
        ]
    )

    return {"messages": [response]}


# 4. Editor
@observe(name="editor_agent", as_agent=True)
def editor_node(state: AgentState):
    """Reviews and polishes the content for better flow and grammar."""
    print("\n--- Editor Agent ---")

    response = llm.invoke(
        [
            SystemMessage(content="You are an Editor. Review and polish the content."),
            state["messages"][-1],
        ]
    )

    return {"messages": [response]}


# 5. QC
@observe(name="qc_agent", as_agent=True)
def qc_node(state: AgentState):
    """Performs quality control checks and compliance verification."""
    print("\n--- QC Agent ---")

    response = llm.invoke(
        [
            SystemMessage(
                content="You are QC. Check for compliance. Respond 'APPROVED' if good."
            ),
            state["messages"][-1],
        ]
    )

    if "APPROVED" in response.content.upper():
        cms_upload(state["messages"][-1].content)

    return {"messages": [response]}


def qc_router(state: AgentState) -> Literal["APPROVED", "REJECTED"]:
    last_message = state["messages"][-1]
    if "APPROVED" in last_message.content.upper():
        return "APPROVED"
    return "REJECTED"


# =========================
# GRAPH
# =========================

workflow = StateGraph(AgentState)


# Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
# workflow.add_node("tools", custom_tool_node) # Removed tools node
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.add_node("qc", qc_node)

# Edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")

# Direct edge to writer (tools already handled in researcher)
workflow.add_edge("researcher", "writer")

workflow.add_edge("writer", "editor")
workflow.add_edge("editor", "qc")

workflow.add_conditional_edges("qc", qc_router, {"APPROVED": END, "REJECTED": "writer"})

app = workflow.compile()

# =========================
# RUNNER
# =========================


@observe(name="run_media_agency", as_type="Runner")
def run_agency():
    print("Starting Media Agency Workflow...")

    final_state = app.invoke(
        {"messages": [HumanMessage(content="Create a blog post about AI Agents.")]}
    )

    print("\nWorkflow Finished.")
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    try:
        run_agency()
        time.sleep(5)
    except Exception:
        import traceback

        traceback.print_exc()
