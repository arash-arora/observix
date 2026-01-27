from typing import Protocol, List
from langchain_core.messages import BaseMessage


class BaseLLM(Protocol):
    def invoke(self, messages: List[BaseMessage]):
        ...

    async def ainvoke(self, messages: List[BaseMessage]):
        ...

    def bind_tools(self, tools: list):
        ...
