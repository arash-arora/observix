from typing import Any, Dict, Optional

from langchain_openai import AzureChatOpenAI as _AzureChatOpenAI
from langchain_openai import ChatOpenAI as _ChatOpenAI

from observix.instrumentation import trace_decorator


class ChatOpenAI(_ChatOpenAI):
    """
    Wrapper around langchain_openai.ChatOpenAI that
    automatically traces calls using obs_sdk.
    """
    
    @trace_decorator(name="ChatOpenAI.invoke")
    def invoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        return super().invoke(input, config=config, **kwargs)

    @trace_decorator(name="ChatOpenAI.ainvoke")
    async def ainvoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        return await super().ainvoke(input, config=config, **kwargs)

class AzureChatOpenAI(_AzureChatOpenAI):
    """
    Wrapper around langchain_openai.AzureChatOpenAI that
    automatically traces calls using obs_sdk.
    """
    
    @trace_decorator(name="AzureChatOpenAI.invoke")
    def invoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        return super().invoke(input, config=config, **kwargs)

    @trace_decorator(name="AzureChatOpenAI.ainvoke")
    async def ainvoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        return await super().ainvoke(input, config=config, **kwargs)
