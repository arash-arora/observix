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
        if token_key:
            current_api_key.reset(token_key)
        if token_host:
            current_host.reset(token_host)
