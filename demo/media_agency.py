import time
from typing import TypedDict
from dotenv import load_dotenv
from observix.agents import Agent, Tool, Graph, END
from observix.agents.utils.logging import setup_logging
from observix import init_observability, observe, capture_context

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()

# --- State ---

class AgentState(TypedDict):
    input: str
    plan: str
    research: str
    draft: str
    polished_content: str
    qc_feedback: str

# --- Tools ---

def google_search_fn(query: str):
    """Performs a Google search to retrieve latest trends and information."""
    print(f"  [Tool] Searching Google for: {query}")
    capture_context("Kubernetes is an open source container orchestration engine for automating deployment, scaling, and management of containerized applications.")
    time.sleep(0.5)
    return f"Search results for {query}: [Trend A, Trend B, Factor C]"

def cms_upload_fn(content: str):
    """Uploads the finalized content to the Content Management System."""
    print("  [Tool] Uploading content to CMS...")
    time.sleep(0.5)
    return "Upload Successful (ID: 12345)"

google_search_tool = Tool(
    fn=google_search_fn, 
    name="google_search", 
    description="Performs a Google search to retrieve latest trends and information."
)
cms_upload_tool = Tool(
    fn=cms_upload_fn, 
    name="cms_upload", 
    description="Uploads the finalized content to the Content Management System."
)

# --- Agents ---

planner_agent = Agent(
    name="Planner",
    description="Responsible for creating the initial content outline.",
    input_key="input",
    output_key="plan",
    instructions="You are a Content Planner. Create a brief outline based on the user query.",
    model="groq/openai/gpt-oss-120b"
)

researcher_agent = Agent(
    name="Researcher",
    description="Conducts research on the given topic using search tools.",
    input_key="plan",
    output_key="research",
    instructions="You are a Researcher. Use the google_search tool to analyze trends based on the outline and provide data.",
    tools=[google_search_tool],
    model="groq/openai/gpt-oss-120b"
)

writer_agent = Agent(
    name="Writer",
    description="Drafts the blog post content based on research findings.",
    input_key="research",
    output_key="draft",
    instructions="You are a Writer. Write a short blog post based on the research findings provided.",
    model="groq/openai/gpt-oss-120b"
)

editor_agent = Agent(
    name="Editor",
    description="Reviews and polishes the content for better flow and grammar.",
    input_key="draft",
    output_key="polished_content",
    instructions="You are an Editor. Review and polish the content provided by the writer.",
    model="groq/openai/gpt-oss-120b"
)

qc_agent = Agent(
    name="QC",
    description="Performs quality control checks and compliance verification.",
    input_key="polished_content",
    output_key="qc_feedback",
    instructions="You are QC. Check for compliance. Respond 'APPROVED' if good, otherwise 'REJECTED' with a reason.",
    tools=[cms_upload_tool],
    model="groq/openai/gpt-oss-120b"
)

# --- Router Functions ---

def qc_router(state: AgentState) -> str:
    feedback = state.get("qc_feedback", "").upper()
    if "APPROVED" in feedback:
        return "approved"
    return "rejected"

# --- Graph ---

graph = (
    Graph(AgentState)
    .add(planner_agent)
    .add(researcher_agent)
    .add(writer_agent)
    .add(editor_agent)
    .add(qc_agent)
    .when("QC", qc_router)
    .then({
        "approved": END,
        "rejected": "Writer"
    })
)

# --- Runner ---

@observe(name="run_media_agency")
def run_agency():
    print("Starting Media Agency Workflow (Refactored)...")
    
    input_query = "Create a blog post about AI Agents."
    result = graph.run(input_query)
    
    print("\nWorkflow Finished.")
    print("Final Polished Content:")
    print(result.get("polished_content", "No content generated."))

if __name__ == "__main__":
    try:
        run_agency()
        # Wait for export
        time.sleep(5)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
