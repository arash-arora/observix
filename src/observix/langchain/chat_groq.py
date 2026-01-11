from typing import Any, Dict, Optional

from langchain_groq import ChatGroq as _ChatGroq

from observix.instrumentation import trace_decorator


class ChatGroq(_ChatGroq):
    """
    Wrapper around langchain_groq.ChatGroq that automatically traces
    calls using obs_sdk.
    """
    
    @trace_decorator(name="ChatGroq.invoke")
    def invoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        # The trace_decorator handles span creation, observation recording,
        # and error handling.
        return super().invoke(input, config=config, **kwargs)

    @trace_decorator(name="ChatGroq.ainvoke")
    async def ainvoke(
        self,
        input: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        return await super().ainvoke(input, config=config, **kwargs)
