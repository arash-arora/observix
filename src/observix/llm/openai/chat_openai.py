from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
from openai import AzureOpenAI as _AzureOpenAI, AsyncAzureOpenAI as _AsyncAzureOpenAI

from observix.instrumentation import observe

class OpenAI(_OpenAI):
    """
    Wrapper around openai.OpenAI that automatically traces chat completions
    using observix.
    """
    def __init__(self, name="OpenAI.chat.completions.create", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Monkey-patch the chat.completions object instance
        original_create = self.chat.completions.create
        if not name:
            name = "OpenAI.chat.completions.create"
        
        if instrument:
            @observe(name=name)
            def wrapped_create(*args, **kwargs):
                return original_create(*args, **kwargs)
                
            self.chat.completions.create = wrapped_create
        else:
            # Keep original unwrapped version
            self.chat.completions.create = original_create

class AzureOpenAI(_AzureOpenAI):
    """
    Wrapper around openai.AzureOpenAI that automatically traces chat completions
    using observix.
    """
    def __init__(self, name="AzureOpenAI.chat.completions.create", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Monkey-patch the chat.completions object instance
        original_create = self.chat.completions.create
        if not name:
            name = "AzureOpenAI.chat.completions.create"
        
        if instrument:
            @observe(name=name)
            def wrapped_create(*args, **kwargs):
                return original_create(*args, **kwargs)
                
            self.chat.completions.create = wrapped_create
        else:
            # Keep original unwrapped version
            self.chat.completions.create = original_create

class AsyncOpenAI(_AsyncOpenAI):
    """
    Wrapper around openai.AsyncOpenAI that automatically traces chat completions
    using observix.
    """
    def __init__(self, name="AsyncOpenAI.chat.completions.create", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)

        original_create = self.chat.completions.create

        if not name:
            name = "AsyncOpenAI.chat.completions.create"

        @observe(name=name)
        async def wrapped_create(*args, **kwargs):
            return await original_create(*args, **kwargs)

        self.chat.completions.create = wrapped_create


class AsyncAzureOpenAI(_AsyncAzureOpenAI):
    """
    Wrapper around openai.AsyncAzureOpenAI that automatically traces chat completions
    using observix.
    """
    def __init__(self, name="AsyncAzureOpenAI.chat.completions.create", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)

        original_create = self.chat.completions.create
        if not name:
            name = "AsyncAzureOpenAI.chat.completions.create"

        if instrument:
            @observe(name=name)
            async def wrapped_create(*args, **kwargs):
                return await original_create(*args, **kwargs)

            self.chat.completions.create = wrapped_create
        else:
            # Keep original unwrapped version
            self.chat.completions.create = original_create
