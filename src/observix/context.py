from contextvars import ContextVar
from typing import Optional, Generator
from contextlib import contextmanager

# ContextVars for dynamic configuration (request-scoped)
current_api_key: ContextVar[Optional[str]] = ContextVar("current_api_key", default=None)
current_host: ContextVar[Optional[str]] = ContextVar("current_host", default=None)

@contextmanager
def observability_context(api_key: Optional[str], host: Optional[str]) -> Generator[None, None, None]:
    token_key = None
    token_host = None
    if api_key:
        token_key = current_api_key.set(api_key)
    if host:
        token_host = current_host.set(host)
    
    try:
        yield
    finally:
        if token_host:
            current_host.reset(token_host)

# Import dependencies for SpanProcessor
from typing import Optional
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span, Context

class MultiTenantSpanProcessor(SpanProcessor):
    """
    Injects the current context API key into span attributes at start time.
    This ensures that even when spans are exported in background threads (BatchSpanProcessor),
    the correct API key is attached to the span data.
    """
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        api_key = current_api_key.get()
        if api_key:
            span.set_attribute("_observix_api_key", api_key)
            
    def on_end(self, span: Span) -> None:
        pass
        
    def shutdown(self) -> None:
        pass
        
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
