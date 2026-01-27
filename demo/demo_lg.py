from typing_extensions import TypedDict
from langchain_community.tools import DuckDuckGoSearchRun

from observix.agents import Agent, Tool, Graph
from observix.agents.utils.logging import setup_logging

from observix.agents import END

setup_logging()


# ========== STATE ==========

class State(TypedDict):
    input: str
    analysis_output: str
    search_output: str
    validation_output: str
    retries: int


# ========== PREDEFINED DUCKDUCKGO SEARCH TOOL ==========

duck_search_tool = Tool(fn=DuckDuckGoSearchRun())

# ========== CONDITION FUNCTION ==========

def check_validator(state: dict) -> str:
    decision = state["validation_output"].strip().upper()
    return "end" if decision == "APPROVED" else "loop"


# ========== AGENTS ==========

analyzer_agent = Agent(
    name="QueryAnalyzer",
    description="Extracts intents and keywords from user queries",
    instructions=""
                 "You are a query analyzer. Identify the core intent or question "
                 "that the user is asking.",
    input_key="input",
    output_key="analysis_output",
    model="groq/openai/gpt-oss-120b"
)

search_agent = Agent(
    name="WebSearcher",
    description="Performs web search via DuckDuckGo",
    instructions=""
                 "You have access to a web search tool named `duckduckgosearchrun`. "
                 "Use it to fetch up-to-date information when needed, "
                 "and return a concise summary of relevant facts.",
    input_key="analysis_output",
    output_key="search_output",
    tools=[duck_search_tool],
    model="groq/openai/gpt-oss-120b"
)

validator_agent = Agent(
    name="Validator",
    description="Decides final approval of info relevance",
    instructions=""
                 "Based on the search results, decide whether the information is "
                 "satisfactory. Only return one word: APPROVED or REJECTED.",
    input_key="search_output",
    output_key="validation_output",
    model="groq/openai/gpt-oss-120b"
)

# ========== GRAPH ==========

graph = (
    Graph(State, max_retries=2)
        .add(analyzer_agent)
        .add(search_agent)
        .add(validator_agent)
        .when("Validator", check_validator)
        .then({
            "end": END,
            "loop": "WebSearcher",
        })
)

# ========== RUN DEMO ==========

if __name__ == "__main__":
    graph.visualize()

    user_query = "What are the key features of Python 3.12?"

    result = graph.run(user_query)

    print("\n✅ FINAL DECISION:", result)
