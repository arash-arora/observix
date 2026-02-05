import json
import os
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from observix import observe
from observix.llm.openai import AzureOpenAI

# Initialize automatically from .env
# Env vars: AZURE_OPENAI_KEY, AZURE_API_BASE

if not os.getenv("AZURE_OPENAI_KEY"):
    raise ValueError(
        "AZURE_OPENAI_KEY not found in env. Please set it in .env or environment variables."
    )
if not os.getenv("AZURE_API_BASE"):
    raise ValueError("AZURE_API_BASE not found in env.")

# Default values if not strictly provided
# Env vars: AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_version=api_version,
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
        elif isinstance(m, ToolMessage):
            openai_messages.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            )
        else:
            openai_messages.append({"role": "user", "content": str(m.content)})

    response = client.chat.completions.create(
        model=deployment_name, messages=openai_messages
    )

    content = response.choices[0].message.content
    return AIMessage(content=content)


# --- Tools (Dummy) ---


@observe(name="google_search", as_tool=True)
def google_search(query: str):
    print(f"  [Tool] Searching Google for: {query}")
    time.sleep(0.5)
    return f"Search results for {query}: [Trend A, Trend B, Factor C]"


@observe(name="cms_upload", as_tool=True)
def cms_upload(content: str):
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
    print("\n--- Planner Agent ---")
    messages = state["messages"]

    # Prepend system instruction
    instructions = [
        SystemMessage(content="You are a Content Planner. Create a brief outline.")
    ]
    full_messages = instructions + messages

    response = invoke_native(full_messages)
    return {"messages": [response]}


# 2. Researcher
# 2. Researcher
@observe(name="researcher_agent", as_agent=True)
def researcher_node(state: AgentState):
    print("\n--- Researcher Agent ---")
    last_message = state["messages"][-1]

    # Define tool schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Performs a Google search to retrieve latest trends and information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # Map for execution
    tools_map = {"google_search": google_search}

    # Prepare messages
    # Note: We duplicate logic from invoke_native here to include system prompt + generic conversion
    # ideally we refactor conversion logic, but keeping it inline for safety in this node

    openai_messages = []
    # Add System Prompt
    openai_messages.append(
        {
            "role": "system",
            "content": "You are a Researcher. Decide what to search to analyze trends. Use google_search tool when needed.",
        }
    )

    # Convert history
    # Note: State messages might be effectively handled by just taking the last one or full history if needed.
    # The original code took `state["messages"][-1]` mixed with system. Let's send the last message as User/Content.
    # Wait, the original code constructed list: [SystemMessage, last_message].
    # So we do the same.

    openai_messages.append({"role": "user", "content": last_message.content})

    # Call Azure OpenAI with tools
    response = client.chat.completions.create(
        model=deployment_name, messages=openai_messages, tools=tools, tool_choice="auto"
    )

    choice = response.choices[0]
    message = choice.message

    output_messages = []

    # Create valid AIMessage from response (including tool calls if any)
    # LangChain AIMessage supports tool_calls arg, but let's stick to standard if possible or just use content.
    # If we have tool calls, content might be None.

    ai_content = message.content or ""
    # We need to act as if we returned an AIMessage with tool calls so LangGraph/Next nodes understand?
    # Or just execute internal loop and return final result?
    # The previous media_agency returned [AIMessage, ToolMessage, ToolMessage].

    # For now, let's just return the AIMessage (with content) and ToolMessages.
    # Note: standard LangChain AIMessage doesn't easily convert from OpenAI object.
    # We will just synthesize a simple AIMessage.

    output_messages.append(AIMessage(content=ai_content))

    if message.tool_calls:
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args_str = tool_call.function.arguments
            call_id = tool_call.id

            if name in tools_map:
                try:
                    args = json.loads(args_str)
                    print(f"  [Researcher] Executing tool: {name}")
                    # Direct execution -> Context propagated automatically
                    result = tools_map[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Error: Tool {name} not found."

            output_messages.append(
                ToolMessage(content=str(result), tool_call_id=call_id, name=name)
            )

    return {"messages": output_messages}


# 3. Writer
@observe(name="writer_agent", as_agent=True)
def writer_node(state: AgentState):
    print("\n--- Writer Agent ---")
    last_message = state["messages"][-1]

    response = invoke_native(
        [
            SystemMessage(
                content="You are a Writer. Write a short blog post based on the research."
            ),
            last_message,
        ]
    )
    return {"messages": [response], "next": "editor"}


# 4. Editor
@observe(name="editor_agent", as_agent=True)
def editor_node(state: AgentState):
    print("\n--- Editor Agent ---")
    last_message = state["messages"][-1]

    response = invoke_native(
        [
            SystemMessage(content="You are an Editor. Review and polish the content."),
            last_message,
        ]
    )
    return {"messages": [response]}


# 5. QC
@observe(name="qc_agent", as_agent=True)
def qc_node(state: AgentState):
    print("\n--- QC Agent ---")
    last_message = state["messages"][-1]

    # Simulate failed QC first? No, let's keep it simple.
    response = invoke_native(
        [
            SystemMessage(
                content="You are QC. Check for compliance. Respond 'APPROVED' if good."
            ),
            last_message,
        ]
    )

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
workflow.add_conditional_edges("qc", qc_router, {"APPROVED": END, "REJECTED": "writer"})

app = workflow.compile()


@observe(name="run_media_agency")
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
