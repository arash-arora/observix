from typing import List, Optional, Callable, Any
from rich.console import Console
from observix.agents.integrations.crewai_adapter import CrewAILLMAdapter


console = Console()


def _import_crewai():
    try:
        from crewai import Agent as CrewAgent
        from crewai import Task, Crew
        from crewai.tools import BaseTool
        return CrewAgent, Task, Crew, BaseTool
    except ImportError:
        raise ImportError(
            "CrewAI is not installed. Install with: pip install 'observix[crewai]'"
        )


# ===================== TOOL =====================

class Tool:
    def __init__(self, name: str, description: str, fn: Callable[..., Any]):
        _, _, _, BaseTool = _import_crewai()

        # ✅ Create Pydantic-safe dynamic class
        CustomTool = type(
            name,
            (BaseTool,),
            {
                "__annotations__": {
                    "name": str,
                    "description": str,
                },
                "name": name,
                "description": description,
                "_run": lambda self, *args, **kwargs: fn(*args, **kwargs),
            },
        )

        self.tool = CustomTool()

    def as_crewai_tool(self):
        return self.tool


# ===================== AGENT =====================

class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        tools=None,
        verbose=True,
        allow_delegation=False,
        model="groq/llama3-70b-8192",
        temperature=0.0,
    ):
        CrewAgent, _, _, _ = _import_crewai()

        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []

        # 🔥 Use custom adapter
        llm_adapter = CrewAILLMAdapter(model=model, temperature=temperature)

        self.agent = CrewAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[t.as_crewai_tool() for t in self.tools],
            verbose=verbose,
            allow_delegation=allow_delegation,
            llm=llm_adapter,
        )

    def as_crewai_agent(self):
        return self.agent



# ===================== CREW (WRAPPER) =====================

class CrewFlow:
    def __init__(self):
        self.agents: List[Agent] = []
        self.tasks: List = []

    def add(self, agent: Agent, task_description: Optional[str] = None):
        """
        Add an agent and auto-generate a task for it
        """
        _, Task, _, _ = _import_crewai()

        self.agents.append(agent)

        description = task_description or f"Execute your role: {agent.goal}"

        task = Task(
            description=description,
            expected_output="A clear and useful response for the user",
            agent=agent.as_crewai_agent(),
        )

        self.tasks.append(task)
        return self

    def run(self, input_text: str):
        _, _, Crew, _ = _import_crewai()

        console.print("\n[bold cyan]🚀 Starting CrewAI Workflow[/bold cyan]")

        crew = Crew(
            agents=[a.as_crewai_agent() for a in self.agents],
            tasks=self.tasks,
            verbose=True,
        )

        result = crew.kickoff(inputs={"input": input_text})

        console.print("\n[bold green]✅ CrewAI Workflow Finished[/bold green]")
        return result
