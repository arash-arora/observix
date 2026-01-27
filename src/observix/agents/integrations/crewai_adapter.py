from typing import List, Union
from langchain_core.messages import BaseMessage
from observix.llm import get_llm


class CrewAILLMAdapter:
    """
    Custom CrewAI LLM wrapper that delegates to observix.llm.get_llm()
    """

    def __init__(self, model: str, temperature: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.llm = get_llm(model=model, temperature=temperature, **kwargs)

    def call(self, prompt: str) -> str:
        """
        CrewAI calls this internally
        """
        messages = [("user", prompt)]
        result = self.llm.invoke(messages)
        return result.content if hasattr(result, "content") else str(result)

    async def acall(self, prompt: str) -> str:
        messages = [("user", prompt)]
        result = await self.llm.ainvoke(messages)
        return result.content if hasattr(result, "content") else str(result)

    @property
    def model_name(self):
        return self.model
    
    def bind_tools(self, tools: list):
        if hasattr(self.llm, "bind_tools"):
            self.llm = self.llm.bind_tools(tools)
        return self
