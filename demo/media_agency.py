import os
import time
from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from observix.llm.langchain import ChatGroq
from observix import observe, capture_context, capture_candidate_agents, capture_tools

# Initialize automatically from .env
# Ensure GROQ_API_KEY is in .env
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in env. Please set it in .env or environment variables.")

llm = ChatGroq(model="openai/gpt-oss-120b")

# --- Tools (Dummy) ---

@observe(name="google_search", as_tool=True)
def google_search(query: str):
    """Performs a Google search to retrieve latest trends and information."""
    print(f"  [Tool] Searching Google for: {query}")
    capture_context("Kubernetes is an open source container orchestration engine for automating deployment, scaling, and management of containerized applications.")
    time.sleep(0.5)
    return f"Search results for {query}: [Trend A, Trend B, Factor C]"

@observe(name="cms_upload", as_tool=True)
def cms_upload(content: str):
    """Uploads the finalized content to the Content Management System."""
    print("  [Tool] Uploading content to CMS...")
    time.sleep(0.5)
    return "Upload Successful (ID: 12345)"

# --- Agents ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str

# 1. Planner
@observe(name="planner_agent", as_agent=True)
def planner_node(state: AgentState):
    """Responsible for creating the initial content outline."""
    print("\n--- Planner Agent ---")
    messages = state["messages"]
    # Simulate LLM call with tracing? Ideally ChatGroq itself should be patched
    # or we wrap the invoke. For this demo, we just use the LLM and valid result.
    # We can wrap LLM calls if we want detailed LLM obs, but user asked for
    # "Agents" and "tools". Since Agents are nodes here, decorated.
    
    response = llm.invoke([
        SystemMessage(content="You are a Content Planner. Create a brief outline."),
        *messages
    ])
    return {"messages": [response]}

# 2. Researcher
@observe(name="researcher_agent", as_agent=True)
def researcher_node(state: AgentState):
    """Conducts research on the given topic using search tools."""
    print("\n--- Researcher Agent ---")
    last_message = state["messages"][-1]
    # Use tool
    search_data = google_search("latest trends in " + last_message.content[:20])
    
    response = llm.invoke([
        SystemMessage(
            content=f"You are a Researcher. Analyze these trends: {search_data}"
        ),
        last_message
    ])
    return {"messages": [response]}

# 3. Writer
@observe(name="writer_agent", as_agent=True)
def writer_node(state: AgentState):
    """Drafts the blog post content based on research findings."""
    print("\n--- Writer Agent ---")
    last_message = state["messages"][-1]
    
    response = llm.invoke([
        SystemMessage(
            content="You are a Writer. Write a short blog post based on the research."
        ),
        last_message
    ])
    return {"messages": [response], "next": "editor"}

# 4. Editor
@observe(name="editor_agent", as_agent=True)
def editor_node(state: AgentState):
    """Reviews and polishes the content for better flow and grammar."""
    print("\n--- Editor Agent ---")
    last_message = state["messages"][-1]
    
    response = llm.invoke([
        SystemMessage(content="You are an Editor. Review and polish the content."),
        last_message
    ])
    return {"messages": [response]}

# 5. QC
@observe(name="qc_agent", as_agent=True)
def qc_node(state: AgentState):
    """Performs quality control checks and compliance verification."""
    print("\n--- QC Agent ---")
    last_message = state["messages"][-1]
    
    # Simulate failed QC first? No, let's keep it simple.
    response = llm.invoke([
        SystemMessage(
            content="You are QC. Check for compliance. Respond 'APPROVED' if good."
        ),
        last_message
    ])
    
    if "APPROVED" in response.content.upper():
        # Use tool
        cms_upload(last_message.content)
        return {"messages": [response]}
    else:
        return {"messages": [response]}

def qc_router(state: AgentState) -> Literal["APPROVED", "REJECTED"]:
    messages = state["messages"]
    last_message = messages[-1]
    if "APPROVED" in last_message.content.upper():
        return "APPROVED"
    return "REJECTED"

# --- Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
workflow.add_node("qc", qc_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", "qc")
workflow.add_conditional_edges("qc", qc_router, {
    "APPROVED": END,
    "REJECTED": "writer"
})

app = workflow.compile()

@observe(name="run_media_agency", as_type="Runner")
def run_agency():
    capture_candidate_agents()
    capture_tools()
    print("Starting Media Agency Workflow...")
    final_state = app.invoke(
        {"messages": [HumanMessage(content="Create a blog post about AI Agents.")]}
    )
    print("\nWorkflow Finished.")
    print(final_state["messages"][-1].content)

if __name__ == "__main__":
    try:
        run_agency()
        # Wait for export
        import time
        time.sleep(5)
    except Exception:
        import traceback
        traceback.print_exc()
