from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
from openai import AzureOpenAI as _AzureOpenAI, AsyncAzureOpenAI as _AsyncAzureOpenAI

from observix.instrumentation import observe, capture_tools, capture_candidate_agents


class OpenAI(_OpenAI):
    """
    Wrapper around openai.OpenAI that automatically traces chat completions
    using observix.
    """

    def __init__(self, name="OpenAI", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Monkey-patch the chat.completions object instance
        original_create = self.chat.completions.create
        if not name:
            name = "OpenAI"

        if instrument:

            @observe(name=name)
            def wrapped_create(*args, **kwargs):
                # Auto-capture tools if present
                if "tools" in kwargs:
                    capture_tools(kwargs["tools"])
                if "functions" in kwargs:
                    capture_tools(kwargs["functions"])  # Treat functions as tools

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

    def __init__(self, name="AzureOpenAI", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Monkey-patch the chat.completions object instance
        original_create = self.chat.completions.create
        if not name:
            name = "AzureOpenAI"

        if instrument:

            @observe(name=name)
            def wrapped_create(*args, **kwargs):
                # Auto-capture tools if present
                if "tools" in kwargs:
                    capture_tools(kwargs["tools"])
                if "functions" in kwargs:
                    capture_tools(kwargs["functions"])

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

    def __init__(self, name="AsyncOpenAI", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)

        original_create = self.chat.completions.create

        if not name:
            name = "AsyncOpenAI"

        if instrument:

            @observe(name=name)
            async def wrapped_create(*args, **kwargs):
                # Auto-capture tools if present
                if "tools" in kwargs:
                    capture_tools(kwargs["tools"])
                if "functions" in kwargs:
                    capture_tools(kwargs["functions"])

                return await original_create(*args, **kwargs)

            self.chat.completions.create = wrapped_create
        else:
            self.chat.completions.create = original_create


class AsyncAzureOpenAI(_AsyncAzureOpenAI):
    """
    Wrapper around openai.AsyncAzureOpenAI that automatically traces chat completions
    using observix.
    """

    def __init__(self, name="AsyncAzureOpenAI", *args, instrument=True, **kwargs):
        super().__init__(*args, **kwargs)

        original_create = self.chat.completions.create
        if not name:
            name = "AsyncAzureOpenAI"

        if instrument:

            @observe(name=name)
            async def wrapped_create(*args, **kwargs):
                # Auto-capture tools if present
                if "tools" in kwargs:
                    capture_tools(kwargs["tools"])
                if "functions" in kwargs:
                    capture_tools(kwargs["functions"])

                return await original_create(*args, **kwargs)

            self.chat.completions.create = wrapped_create
        else:
            # Keep original unwrapped version
            self.chat.completions.create = original_create
