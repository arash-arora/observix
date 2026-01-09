import os
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from obs_sdk import ChatGroq, trace_decorator

# Initialize automatically from .env
# Ensure GROQ_API_KEY is in .env

if not os.getenv("GROQ_API_KEY"):
    # Fallback for demo if not provided, though user said they will add it.
    print("WARNING: GROQ_API_KEY not found in env.")

llm = ChatGroq(model="openai/gpt-oss-120b")

# --- Tools (Dummy) ---

@trace_decorator(name="google_search")
def google_search(query: str):
    print(f"  [Tool] Searching Google for: {query}")
    time.sleep(0.5)
    return f"Search results for {query}: [Trend A, Trend B, Factor C]"

@trace_decorator(name="cms_upload")
def cms_upload(content: str):
    print("  [Tool] Uploading content to CMS...")
    time.sleep(0.5)
    return "Upload Successful (ID: 12345)"

# --- Agents ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str

# 1. Planner
def planner_node(state: AgentState):
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
def researcher_node(state: AgentState):
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
def writer_node(state: AgentState):
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
def editor_node(state: AgentState):
    print("\n--- Editor Agent ---")
    last_message = state["messages"][-1]
    
    response = llm.invoke([
        SystemMessage(content="You are an Editor. Review and polish the content."),
        last_message
    ])
    return {"messages": [response]}

# 5. QC
def qc_node(state: AgentState):
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

@trace_decorator(name="run_media_agency")
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
        # Wait for export
        import time
        time.sleep(5)
    except Exception as e:
        print(f"Error: {e}")
