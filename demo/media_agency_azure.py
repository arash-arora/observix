import os
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from observix import AzureOpenAI, trace_decorator

# Initialize automatically from .env
# Env vars: AZURE_OPENAI_KEY, AZURE_API_BASE

if not os.getenv("AZURE_OPENAI_KEY"):
    raise ValueError("AZURE_OPENAI_KEY not found in env. Please set it in .env or environment variables.")
if not os.getenv("AZURE_API_BASE"):
    raise ValueError("AZURE_API_BASE not found in env.")

# Default values if not strictly provided
# Env vars: AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_version=api_version
)

def invoke_native(messages: list[BaseMessage]) -> AIMessage:
    """Helper to convert LangChain messages to OpenAI format and back."""
    openai_messages = []
    for m in messages:
        if isinstance(m, SystemMessage):
            openai_messages.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
             openai_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
             openai_messages.append({"role": "assistant", "content": m.content})
        else:
             openai_messages.append({"role": "user", "content": str(m.content)})

    response = client.chat.completions.create(
        model=deployment_name,
        messages=openai_messages
    )
    
    content = response.choices[0].message.content
    return AIMessage(content=content)


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
@trace_decorator(name="planner_agent")
def planner_node(state: AgentState):
    print("\n--- Planner Agent ---")
    messages = state["messages"]
    
    # Prepend system instruction
    instructions = [SystemMessage(content="You are a Content Planner. Create a brief outline.")]
    full_messages = instructions + messages
    
    response = invoke_native(full_messages)
    return {"messages": [response]}

# 2. Researcher
@trace_decorator(name="researcher_agent")
def researcher_node(state: AgentState):
    print("\n--- Researcher Agent ---")
    last_message = state["messages"][-1]
    # Use tool
    search_data = google_search("latest trends in " + last_message.content[:20])
    
    response = invoke_native([
        SystemMessage(
            content=f"You are a Researcher. Analyze these trends: {search_data}"
        ),
        last_message
    ])
    return {"messages": [response]}

# 3. Writer
@trace_decorator(name="writer_agent")
def writer_node(state: AgentState):
    print("\n--- Writer Agent ---")
    last_message = state["messages"][-1]
    
    response = invoke_native([
        SystemMessage(
            content="You are a Writer. Write a short blog post based on the research."
        ),
        last_message
    ])
    return {"messages": [response], "next": "editor"}

# 4. Editor
@trace_decorator(name="editor_agent")
def editor_node(state: AgentState):
    print("\n--- Editor Agent ---")
    last_message = state["messages"][-1]
    
    response = invoke_native([
        SystemMessage(content="You are an Editor. Review and polish the content."),
        last_message
    ])
    return {"messages": [response]}

# 5. QC
@trace_decorator(name="qc_agent")
def qc_node(state: AgentState):
    print("\n--- QC Agent ---")
    last_message = state["messages"][-1]
    
    # Simulate failed QC first? No, let's keep it simple.
    response = invoke_native([
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
    print("Starting Media Agency Workflow (Azure Native)...")
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
