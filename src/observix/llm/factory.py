import os
from typing import Optional

def get_llm(
    model: str,
    temperature: float = 0.0,
    framework: str = "langchain",
    name: Optional[str] = None,
    is_async: bool = False,
    **kwargs,
):
    """
    Centralized factory to get an LLM instance.
    Supported formats:
    - openai/<model_name>
    - groq/<model_name>
    - azure/<model_name>
    """
    provider, model_name = _parse_model(model)

    if framework == "langchain":
        return _get_langchain_llm(provider, model_name, temperature, **kwargs)
    elif framework == "openai":
        return _get_openai_llm(provider, model_name, name=name, is_async=is_async, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def _get_langchain_llm(provider: str, model_name: str, temperature: float, **kwargs):
    if provider == "openai":
        from observix.llm.langchain import ChatOpenAI
        api_key = kwargs.pop("api_key", os.getenv("OPENAI_API_KEY"))
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "groq":
        from observix.llm.langchain import ChatGroq
        api_key = kwargs.pop("api_key", os.getenv("GROQ_API_KEY"))
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "azure":
        from observix.llm.langchain import AzureChatOpenAI
        api_key = kwargs.pop("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        azure_endpoint = kwargs.pop("azure_endpoint", os.getenv("AZURE_OPENAI_ENDPOINT"))
        api_version = kwargs.pop("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
        return AzureChatOpenAI(
            azure_deployment=model_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            temperature=temperature,
            **kwargs,
        )
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "To use Gemini models, install: pip install langchain-google-genai"
            )
        google_api_key = kwargs.pop("google_api_key", os.getenv("GOOGLE_API_KEY"))
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=google_api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider for LangChain: {provider}")



def _get_openai_llm(
    provider: str, 
    model_name: str, 
    name: Optional[str] = None, 
    is_async: bool = False, 
    **kwargs
):
    if provider == "openai":
        if is_async:
            from observix.llm.openai import AsyncOpenAI
            return AsyncOpenAI(name=name, model=model_name, **kwargs)
        else:
            from observix.llm.openai import OpenAI
            return OpenAI(name=name, model=model_name, **kwargs)
    elif provider == "azure":
        if is_async:
            from observix.llm.openai import AsyncAzureOpenAI
            return AsyncAzureOpenAI(name=name, model=model_name, **kwargs)
        else:
            from observix.llm.openai import AzureOpenAI
            return AzureOpenAI(name=name, model=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported provider for OpenAI: {provider}")



def _parse_model(model: str):
    if "/" not in model:
        # Default to openai if no provider is specified
        return "openai", model

    provider, model_name = model.split("/", 1)
    return provider.lower(), model_name
