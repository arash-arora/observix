from typing import Any

from openai import AzureOpenAI as _AzureOpenAI
from openai import OpenAI as _OpenAI
from openai.resources.chat.completions import Completions

from observix.instrumentation import observe


class WrappedCompletions(Completions):
    """
    Wrapper for openai.resources.chat.completions.Completions
    """
    def __init__(self, client):
        super().__init__(client)

    @observe(name="OpenAI.chat.completions.create")
    def create(self, *args, **kwargs) -> Any:
        return super().create(*args, **kwargs)

class OpenAI(_OpenAI):
    """
    Wrapper around openai.OpenAI that automatically traces chat completions
    using obs_sdk.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Monkey-patch the chat.completions object instance
        original_create = self.chat.completions.create
        
        @observe(name="OpenAI.chat.completions.create")
        def wrapped_create(*args, **kwargs):
            return original_create(*args, **kwargs)
            
        self.chat.completions.create = wrapped_create

class AzureOpenAI(_AzureOpenAI):
    """
    Wrapper around openai.AzureOpenAI that automatically traces chat completions
    using obs_sdk.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Monkey-patch the chat.completions object instance
        original_create = self.chat.completions.create
        
        @observe(name="AzureOpenAI.chat.completions.create")
        def wrapped_create(*args, **kwargs):
            return original_create(*args, **kwargs)
            
        self.chat.completions.create = wrapped_create
