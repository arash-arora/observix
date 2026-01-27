class AgentsError(Exception):
    """Base class for all observix agents errors."""
    pass

class WorkflowError(AgentsError):
    """Raised when there is an error in the workflow execution."""
    pass

class ConfigurationError(AgentsError):
    """Raised when there is a configuration error."""
    pass

class ModelProviderError(AgentsError):
    """Raised when there is an error with the LLM provider."""
    pass
