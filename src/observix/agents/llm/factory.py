from typing import Any
from observix.llm import get_llm as _get_llm
from observix.agents.llm.base import BaseLLM

def get_llm(
    model: str,
    temperature: float = 0.0,
    framework: str = "langchain",
    **kwargs: Any,
) -> BaseLLM:
    """
    Delegates to the centralized observix.llm.get_llm factory.
    """
    return _get_llm(
        model=model,
        temperature=temperature,
        framework=framework,
        **kwargs
    )
