import os
import time
import json
import random
import inspect
import asyncio
import datetime as dt
import functools
from datetime import datetime
from dotenv import load_dotenv
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.context import get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from observix.schema import Observation
import observix.exporter as exporter_module
from observix.exporter import HttpObservationExporter, HttpTraceExporter

# Load env variables from .env file if present
load_dotenv()

_TRACER_NAME = "observix"

# Track if observability has been initialized to prevent duplicate processors
_observability_initialized = False

# ContextVar to track the current observation ID for parent-child relationship
_current_observation_id: ContextVar[Optional[int]] = ContextVar(
    "current_observation_id", default=None
)


def init_observability(
    url: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Initialize the observability SDK.
    Configuration can be provided via arguments or environment variables:
    - OBSERVIX_HOST or OBSERVIX_URL (e.g., http://localhost:8000)
    - OBSERVIX_API_KEY
    
    This function is idempotent - calling it multiple times will not add duplicate processors.
    """
    global _observability_initialized
    
    url = url or os.getenv("OBSERVIX_HOST") or os.getenv("OBSERVIX_URL")
    api_key = api_key or os.getenv("OBSERVIX_API_KEY")

    if not url:
        print("[OBSERVIX] Warning: OBSERVIX_HOST not found. Observability disabled.")
        return
        
    if not api_key:
        print("[OBSERVIX] Warning: OBSERVIX_API_KEY not found. Observability disabled.")
        return
    
    # Prevent duplicate initialization
    if _observability_initialized:
        print("[OBSERVIX] Already initialized. Skipping duplicate initialization.")
        return

    # --- initialize provider only once ---
    provider = trace.get_tracer_provider()
    if (
        isinstance(provider, trace.ProxyTracerProvider) or 
        not isinstance(provider, TracerProvider)
    ):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    # Use BatchSpanProcessor for production performance
    exporter = HttpTraceExporter(url, api_key)
    processor = BatchSpanProcessor(exporter)
    
    # Add MultiTenant context injector BEFORE the batch processor
    # This ensures attributes are set before export happens
    from observix.context import MultiTenantSpanProcessor
    provider.add_span_processor(MultiTenantSpanProcessor())
    
    provider.add_span_processor(processor)

    # --- initialize observation exporter ---
    exporter_module.observation_exporter_instance = HttpObservationExporter(
        url, api_key
    )
    
    _observability_initialized = True
    print(f"[OBSERVIX] Initialized with Host: {url}")


# Try to auto-initialize if configured
if (os.getenv("OBSERVIX_HOST") or os.getenv("OBSERVIX_URL")) and os.getenv("OBSERVIX_API_KEY"):
    try:
        init_observability()
    except Exception as e:
        print(f"[OBSERVIX] Auto-initialization failed: {e}")


def _extract_token_usage(result: Any) -> Optional[Dict[str, int]]:
    usage = None
    # OpenAI object style
    if hasattr(result, "usage") and result.usage:
        usage = result.usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    
    # Dict style (e.g. dict response)
    if isinstance(result, dict) and "usage" in result:
        usage = result["usage"]
        if isinstance(usage, dict):
             return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            
    # Langchain AIMessage style
    if hasattr(result, "response_metadata"):
        meta = getattr(result, "response_metadata", {})
        if "token_usage" in meta:
             usage = meta["token_usage"]
             # Langchain usages might differ in keys, but usually standard
             return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            
    return None


# Pricing Registry (USD per 1M tokens)
PRICING_REGISTRY = {
    # OpenAI
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    # Azure OpenAI often uses gpt-35-turbo
    "gpt-35-turbo": {"input": 0.50, "output": 1.50}, 
    # Anthropic
    "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

def _calculate_cost(model_name: str, usage: Dict[str, int]) -> float:
    if not model_name or not usage:
        return 0.0
    
    # Normalize model name
    model_key = model_name.lower()
    pricing = None
    
    # Exact match
    if model_key in PRICING_REGISTRY:
        pricing = PRICING_REGISTRY[model_key]
    else:
        # Partial match (e.g. "gpt-4o-custom" -> "gpt-4o")
        # Sort by length descending to match longest prefix (most specific)
        for k in sorted(PRICING_REGISTRY.keys(), key=len, reverse=True):
            if k in model_key:
                pricing = PRICING_REGISTRY[k]
                break
                
    if not pricing:
        return 0.0
        
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost

def _extract_cost(result: Any, model_name: Optional[str] = None, token_usage: Optional[Dict[str, int]] = None) -> Optional[float]:
    # 1. Try explicit cost in result
    cost = None
    
    # OpenAI object style logic not standard for cost, but some proxies provide it
    if hasattr(result, "cost"):
        cost = float(result.cost)
        
    # Check "usage" dict for cost (some providers)
    elif isinstance(result, dict) and "usage" in result:
        usage = result["usage"]
        if isinstance(usage, dict) and "cost" in usage:
             cost = float(usage["cost"])
        elif isinstance(usage, dict) and "total_cost" in usage:
             cost = float(usage["total_cost"])

    # Langchain AIMessage style
    elif hasattr(result, "response_metadata"):
        meta = getattr(result, "response_metadata", {})
        if "cost" in meta:
             cost = float(meta["cost"])
        elif "total_cost" in meta:
             cost = float(meta["total_cost"])
        # Check token_usage for cost
        elif "token_usage" in meta:
             usage = meta["token_usage"]
             if isinstance(usage, dict) and "cost" in usage:
                  cost = float(usage["cost"])
             elif isinstance(usage, dict) and "total_cost" in usage:
                  cost = float(usage["total_cost"])
    
    # 2. If no explicit cost, calculate from usage
    if cost is None and model_name and token_usage:
        cost = _calculate_cost(model_name, token_usage)
        
    return cost

def _extract_model_name(kwargs: Dict, result: Any) -> Optional[str]:
    # Check input kwargs first
    if "model" in kwargs:
        return str(kwargs["model"])
    if "model_name" in kwargs:
        return str(kwargs["model_name"])
    if "deployment_name" in kwargs:
        return str(kwargs["deployment_name"])
    if "engine" in kwargs:
        return str(kwargs["engine"])
        
    # Check result object
    if hasattr(result, "model"):
        return str(result.model)
    if hasattr(result, "model_name"):
        return str(result.model_name)
    
    # Check dict result
    if isinstance(result, dict):
        if "model" in result:
            return str(result["model"])
        if "model_name" in result:
            return str(result["model_name"])
        
    # Check response_metadata (LangChain)
    if hasattr(result, "response_metadata"):
        meta = getattr(result, "response_metadata", {})
        if "model_name" in meta:
            return str(meta["model_name"])
        if "model" in meta:
            return str(meta["model"])
        
    return None

def _clean_obj(obj: Any, depth=0, max_depth=3) -> Any:
    """
    Clean object for serialization.
    Handles Pydantic models, LangChain messages, Secrets, and arbitrary objects.
    """
    if depth > max_depth:
        return str(obj)

    if obj is None:
        return None
        
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle Pydantic SecretStr/SecretBytes
    if hasattr(obj, "get_secret_value"):
        return "**********"
    
    # Handle Objects with sensible dict serialization
    if hasattr(obj, "model_dump"): # Pydantic v2
        return _clean_obj(obj.model_dump(), depth, max_depth)
    if hasattr(obj, "dict") and callable(obj.dict): # Pydantic v1 / LangChain
        try:
             d = obj.dict() 
             return _clean_obj(d, depth, max_depth)
        except:
             pass
    
    # Handle Lists/Tuples
    if isinstance(obj, (list, tuple)):
        return [_clean_obj(x, depth + 1, max_depth) for x in obj]
    
    # Handle Dicts
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Redact common secret keys
            # Avoid redacting "max_tokens", "token_usage" by being more specific
            is_secret = False
            if isinstance(k, str):
                k_lower = k.lower()
                if any(s in k_lower for s in ["api_key", "secret", "password", "auth_token", "access_token"]):
                    is_secret = True
                elif k_lower == "token": # Exact match for "token" is likely a secret
                    is_secret = True
            
            if is_secret:
                new_dict[k] = "**********"
            else:
                new_dict[k] = _clean_obj(v, depth + 1, max_depth)
        return new_dict

    # Handle LangChain BaseMessage
    if hasattr(obj, "content") and hasattr(obj, "role"): 
         ctx = {"role": getattr(obj, "role"), "content": getattr(obj, "content")}
         if hasattr(obj, "id"): ctx["id"] = getattr(obj, "id")
         return ctx

    # Filter out complex Client objects
    type_str = str(type(obj))
    if "client" in type_str.lower() or "connection" in type_str.lower() or "session" in type_str.lower():
         return f"<{type_str}>"

    # Default
    return str(obj)

def observe(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_observation: bool = True,
):
    def wrapper(func: Callable):
        func_name = name or func.__name__
        tracer = trace.get_tracer(_TRACER_NAME)

        @functools.wraps(func)
        def inner(*args, **kwargs):
            # Capture start time properties
            start_time = time.time()
            
            obs_id = random.getrandbits(63)
            # Get parent observation ID from context
            parent_obs_id = _current_observation_id.get()
            
            # Set current observation ID for children
            token = _current_observation_id.set(obs_id)
            
            # Start Span
            parent_context = get_current()
            
            with tracer.start_as_current_span(
                func_name,
                context=parent_context,
                kind=SpanKind.INTERNAL
            ) as span:
                span.set_attribute("observation_id", str(obs_id))
                if parent_obs_id:
                     span.set_attribute("parent_observation_id", str(parent_obs_id))
                
                if attributes:
                    for k, v in attributes.items(): 
                        span.set_attribute(k, v)

                def create_observation(
                    span, args, kwargs, result=None, exc=None,
                    obs_id=None, parent_obs_id=None
                ):
                    exporter = exporter_module.observation_exporter_instance
                    if not record_observation or exporter is None:
                        return

                    end_time_val = time.time()
                    
                    obs = Observation(
                        id=obs_id or random.getrandbits(63),
                        parent_observation_id=parent_obs_id, # Link to parent
                        name=func_name,
                        type="function",
                        start_time=int(start_time * 1e9),
                        end_time=int(end_time_val * 1e9), # nanoseconds
                        metadata_json=json.dumps(span.attributes, default=str),
                        observation_type="decorator",
                        trace_id=f"{span.get_span_context().trace_id:032x}",
                        created_at=datetime.now(dt.timezone.utc),
                    )
                    
                    # Careful with JSON dumping arbitrary objects
                    try:
                         cleaned_args = _clean_obj(args)
                         cleaned_kwargs = _clean_obj(kwargs)
                         obs.input_text = json.dumps(
                             {"args": cleaned_args, "kwargs": cleaned_kwargs}
                         )
                    except Exception:
                         obs.input_text = json.dumps(
                             {"args": str(args), "kwargs": str(kwargs)}
                         )
                    
                    if result is not None:
                        try:
                             # Try dumping result
                             cleaned_result = _clean_obj(result)
                             obs.output_text = json.dumps(cleaned_result)
                        except Exception:
                             obs.output_text = str(result)
                    else:
                        if exc:
                             obs.output_text = f"Error: {str(exc)}"
                        else:
                             obs.output_text = ""
    
                    if result is not None:
                         obs.token_usage = _extract_token_usage(result)
                         obs.model = _extract_model_name(kwargs, result)
                         obs.total_cost = _extract_cost(result, obs.model, obs.token_usage)
                    else:
                         obs.token_usage = None
                         obs.model = _extract_model_name(kwargs, None)
                         obs.total_cost = None

                    obs.model_parameters = None # Could extract from kwargs if needed
                    
                    if exc:
                         obs.error = str(exc)
    
                    try:
                        exporter.enqueue(obs)
                    except Exception as obs_exc:
                        # do not break tracing if observation fails
                        print(
                            f"[ObsWarning] Failed to create observation: {obs_exc}"
                        )

                if asyncio.iscoroutinefunction(func):
                    # We need to define an async wrapper to await the function
                    async def async_inner_wrapper(*args, **kwargs):
                        try:
                            result = await func(*args, **kwargs)
                            create_observation(
                                span, args, kwargs, result=result,
                                obs_id=obs_id, parent_obs_id=parent_obs_id
                            )
                            return result
                        except Exception as exc:
                            span.record_exception(exc)
                            span.set_status(Status(StatusCode.ERROR))
                            create_observation(
                                span, args, kwargs, exc=exc,
                                obs_id=obs_id, parent_obs_id=parent_obs_id
                            )
                            raise
                        finally:
                            span.set_attribute(
                                "duration_ms", (time.time() - start_time) * 1000
                            )
                            _current_observation_id.reset(token)
                    
                    return asyncio.run(async_inner_wrapper(*args, **kwargs))
                else:
                    try:
                        result = func(*args, **kwargs)
                        create_observation(
                            span, args, kwargs, result=result,
                            obs_id=obs_id, parent_obs_id=parent_obs_id
                        )
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR))
                        create_observation(
                            span, args, kwargs, exc=exc,
                            obs_id=obs_id, parent_obs_id=parent_obs_id
                        )
                        raise
                    finally:
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )
                        _current_observation_id.reset(token)
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_main_wrapper(*args, **kwargs):
                # Capture start time properties
                start_time = time.time()
                
                obs_id = random.getrandbits(63)
                parent_obs_id = _current_observation_id.get()
                token = _current_observation_id.set(obs_id)
                
                # Start Span
                parent_context = get_current()
                
                with tracer.start_as_current_span(
                    func_name,
                    context=parent_context,
                    kind=SpanKind.INTERNAL
                ) as span:
                    span.set_attribute("observation_id", str(obs_id))
                    if parent_obs_id:
                         span.set_attribute(
                             "parent_observation_id", str(parent_obs_id)
                         )
                    
                    if attributes:
                        for k, v in attributes.items(): 
                            span.set_attribute(k, v)

                    def create_observation_async(
                        span, args, kwargs, result=None, exc=None,
                        obs_id=None, parent_obs_id=None
                    ):
                        exporter = exporter_module.observation_exporter_instance
                        if not record_observation or exporter is None:
                            return

                        end_time_val = time.time()
                        
                        obs = Observation(
                            id=obs_id or random.getrandbits(63),
                            parent_observation_id=parent_obs_id,
                            name=func_name,
                            type="function",
                            start_time=int(start_time * 1e9),
                            end_time=int(end_time_val * 1e9),
                            metadata_json=json.dumps(span.attributes, default=str),
                            observation_type="decorator",
                            trace_id=f"{span.get_span_context().trace_id:032x}",
                            created_at=datetime.utcnow(),
                        )
                        
                        try:
                             cleaned_args = _clean_obj(args)
                             cleaned_kwargs = _clean_obj(kwargs)
                             obs.input_text = json.dumps(
                                 {"args": cleaned_args, "kwargs": cleaned_kwargs}
                             )
                        except Exception:
                             obs.input_text = json.dumps(
                                 {"args": str(args), "kwargs": str(kwargs)}
                             )
                        
                        if result is not None:
                            try:
                                 cleaned_result = _clean_obj(result)
                                 obs.output_text = json.dumps(cleaned_result)
                            except Exception:
                                 obs.output_text = str(result)
                        else:
                            if exc:
                                 obs.output_text = f"Error: {str(exc)}"
                            else:
                                 obs.output_text = ""
        
                        if result is not None:
                             obs.token_usage = _extract_token_usage(result)
                             obs.model = _extract_model_name(kwargs, result)
                             obs.total_cost = _extract_cost(result, obs.model, obs.token_usage)
                        else:
                             obs.token_usage = None
                             obs.model = _extract_model_name(kwargs, None)
                             obs.total_cost = None

                        obs.model_parameters = None
                        
                        if exc:
                             obs.error = str(exc)
        
                        try:
                            exporter.enqueue(obs)
                        except Exception as obs_exc:
                            print(
                                f"[ObservixWarning] Failed to create observation: {obs_exc}"
                            )

                    try:
                        result = await func(*args, **kwargs)
                        create_observation_async(
                            span, args, kwargs, result=result,
                            obs_id=obs_id, parent_obs_id=parent_obs_id
                        )
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR))
                        create_observation_async(
                            span, args, kwargs, exc=exc,
                            obs_id=obs_id, parent_obs_id=parent_obs_id
                        )
                        raise
                    finally:
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )
                        _current_observation_id.reset(token)
            
            return async_main_wrapper
        
        else:
            return inner

    return wrapper

def record_score(
    name: str,
    score: float,
    trace_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None
):
    """
    Record an evaluation score as an observation.
    
    Args:
        name: Name of the metric (e.g. "faithfulness")
        score: The numerical score value
        trace_id: ID of the trace being evaluated (optional, defaults to current context)
        observation_id: ID of the specific observation being evaluated (optional)
        metadata: Additional metadata
        reason: Explanation for the score
    """
    exporter = exporter_module.observation_exporter_instance
    if exporter is None:
        return

    # Use current trace context if not provided
    if not trace_id:
        span = trace.get_current_span()
        if span.get_span_context().is_valid:
            trace_id = f"{span.get_span_context().trace_id:032x}"
        else:
            trace_id = f"{random.getrandbits(128):032x}"

    obs_id = random.getrandbits(63)
    start_time = time.time()
    
    meta = metadata or {}
    if reason:
        meta["reason"] = reason
    if observation_id:
        meta["evaluated_observation_id"] = observation_id

    obs = Observation(
        id=obs_id,
        name=name,
        type="score",
        start_time=int(start_time * 1e9),
        end_time=int(start_time * 1e9), # Instantaneous
        input_text=json.dumps(meta) if meta else None,
        output_text=str(score), # Store score as string in output_text
        trace_id=trace_id,
        metadata_json=json.dumps(meta, default=str),
        created_at=datetime.utcnow(),
    )

    try:
        exporter.enqueue(obs)
    except Exception as e:
        print(f"[ObservixWarning] Failed to record score: {e}")


def capture_context(context: Any):
    """
    Explicitly capture retrieval context or other context data for the current span.
    This is useful for accurate RAG evaluations (e.g. context_precision, recall).
    
    Args:
        context: The context data (list of strings, string, or JSON-serializable object).
    """
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return

    try:
        if isinstance(context, (str, list, dict)):
            # Store directly if simple type, else json dump
            if isinstance(context, str):
                 span.set_attribute("context", context)
            else:
                 span.set_attribute("context", json.dumps(context, default=str))
        else:
            span.set_attribute("context", str(context))
    except Exception as e:
         print(f"[ObservixWarning] Failed to capture context: {e}")
