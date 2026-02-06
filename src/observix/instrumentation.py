import os
import time
import json
import random
import inspect
import asyncio
import datetime as dt
import functools
import uuid
from datetime import datetime
from dotenv import load_dotenv
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, List

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

# Global Registries for Agents and Tools
_REGISTERED_AGENTS: Dict[str, Dict[str, str]] = {}
_REGISTERED_TOOLS: Dict[str, Dict[str, str]] = {}


def init_observability(url: Optional[str] = None, api_key: Optional[str] = None):
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
    if isinstance(provider, trace.ProxyTracerProvider):
        # Only set if it's the default ProxyTracerProvider
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    elif not isinstance(provider, TracerProvider):
        # If it's some other provider (not SDK TracerProvider), we might want to warn or skip
        # For safety, if it's already set (and not proxy), let's reuse it or skip setting.
        pass

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
if (os.getenv("OBSERVIX_HOST") or os.getenv("OBSERVIX_URL")) and os.getenv(
    "OBSERVIX_API_KEY"
):
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


def _extract_cost(
    result: Any,
    model_name: Optional[str] = None,
    token_usage: Optional[Dict[str, int]] = None,
) -> Optional[float]:
    # 1. Try explicit cost in result
    cost = None

    if hasattr(result, "cost"):
        cost = float(result.cost)

    elif isinstance(result, dict) and "usage" in result:
        usage = result["usage"]
        if isinstance(usage, dict) and "cost" in usage:
            cost = float(usage["cost"])
        elif isinstance(usage, dict) and "total_cost" in usage:
            cost = float(usage["total_cost"])

    elif hasattr(result, "response_metadata"):
        meta = getattr(result, "response_metadata", {})
        if "cost" in meta:
            cost = float(meta["cost"])
        elif "total_cost" in meta:
            cost = float(meta["total_cost"])
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


def _extract_tool_calls(result: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Extract tool selection (decisions) from LLM result.
    Returns list of dicts: [{"name": "tool_name", "arguments": {...}}]
    """
    tool_calls = []

    # 1. OpenAI ChatCompletion object
    if (
        hasattr(result, "choices")
        and isinstance(result.choices, list)
        and len(result.choices) > 0
    ):
        choice = result.choices[0]
        if hasattr(choice, "message"):
            message = choice.message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    name = tc.function.name
                    args = tc.function.arguments
                    try:
                        args = json.loads(args)
                    except:
                        pass
                    tool_calls.append({"name": name, "arguments": args})

    # 2. LangChain AIMessage
    if (
        hasattr(result, "tool_calls")
        and isinstance(result.tool_calls, list)
        and result.tool_calls
    ):
        for tc in result.tool_calls:
            # LangChain tool_call is usually dict like {'name': 'foo', 'args': {}, 'id': ...}
            name = tc.get("name")
            args = tc.get("args")
            tool_calls.append({"name": name, "arguments": args})

    # 3. OpenAI Dict Response
    if isinstance(result, dict) and "choices" in result:
        choices = result["choices"]
        if isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict) and "message" in choice:
                message = choice["message"]
                if isinstance(message, dict) and "tool_calls" in message:
                    tcs = message["tool_calls"]
                    if isinstance(tcs, list):
                        for tc in tcs:
                            func = tc.get("function", {})
                            name = func.get("name")
                            args = func.get("arguments")
                            try:
                                args = json.loads(args)
                            except:
                                pass
                            tool_calls.append({"name": name, "arguments": args})

    return tool_calls if tool_calls else None


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
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return _clean_obj(obj.model_dump(), depth, max_depth)
    if hasattr(obj, "dict") and callable(obj.dict):  # Pydantic v1 / LangChain
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
            is_secret = False
            if isinstance(k, str):
                k_lower = k.lower()
                if any(
                    s in k_lower
                    for s in [
                        "api_key",
                        "secret",
                        "password",
                        "auth_token",
                        "access_token",
                    ]
                ):
                    is_secret = True
                elif k_lower == "token":  # Exact match for "token" is likely a secret
                    is_secret = True

            if is_secret:
                new_dict[k] = "**********"
            else:
                new_dict[k] = _clean_obj(v, depth + 1, max_depth)
        return new_dict

    # Handle LangChain BaseMessage
    if hasattr(obj, "content") and hasattr(obj, "role"):
        ctx = {"role": getattr(obj, "role"), "content": getattr(obj, "content")}
        if hasattr(obj, "id"):
            ctx["id"] = getattr(obj, "id")
        return ctx

    # Filter out complex Client objects
    type_str = str(type(obj))
    if (
        "client" in type_str.lower()
        or "connection" in type_str.lower()
        or "session" in type_str.lower()
    ):
        return f"<{type_str}>"

    # Default
    return str(obj)


def _prepare_metadata(attributes: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Prepare metadata JSON from span attributes.
    Parses known JSON-stringified attributes back into objects to ensure nested structure.
    """
    if not attributes:
        return None

    # usage of mappingproxy in OTel, convert to dict
    meta = dict(attributes)

    # Keys that are known to be JSON dumps and should be restored
    json_keys = ["candidate_agents", "tools", "context", "metadata"]

    for k, v in meta.items():
        if k in json_keys and isinstance(v, str):
            try:
                # Attempt to parse JSON strings back to objects
                meta[k] = json.loads(v)
            except Exception:
                # Keep as string if parsing fails
                pass

    return json.dumps(meta, default=str)


def observe(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_observation: bool = True,
    as_type: Optional[str] = None,  # "agent" or "tool"
    as_agent: bool = False,
    as_tool: bool = False,
):
    def wrapper(func: Callable):
        func_name = name or func.__name__

        # --- Registration Logic ---
        is_agent = as_agent or (as_type and as_type.lower() == "agent")
        is_tool = as_tool or (as_type and as_type.lower() == "tool")

        if is_agent or is_tool:
            doc = func.__doc__
            if not doc or not doc.strip():
                entity_type = "Agent" if is_agent else "Tool"
                raise ValueError(
                    f"Observix Error: {entity_type} '{func_name}' must have a docstring to be registered."
                )

            description = doc.strip()
            # Register
            entry = {"name": func_name, "description": description}
            if is_agent:
                _REGISTERED_AGENTS[func_name] = entry
            else:
                _REGISTERED_TOOLS[func_name] = entry

        tracer = trace.get_tracer(_TRACER_NAME)

        def create_observation_record(
            span,
            args,
            kwargs,
            start_time,
            obs_id,
            parent_obs_id,
            result=None,
            exc=None,
        ):
            exporter = exporter_module.observation_exporter_instance
            if not record_observation or exporter is None:
                return

            # Determine type
            obs_type = "function"
            if is_agent:
                obs_type = "agent"
            elif is_tool:
                obs_type = "tool"

            end_time_val = time.time()

            obs = Observation(
                id=obs_id or random.getrandbits(63),
                parent_observation_id=parent_obs_id,  # Link to parent
                name=func_name,
                type=obs_type,
                start_time=int(start_time * 1e9),
                end_time=int(end_time_val * 1e9),  # nanoseconds
                metadata_json=_prepare_metadata(span.attributes),
                observation_type="decorator",
                trace_id=f"{span.get_span_context().trace_id:032x}",
                created_at=datetime.utcnow(),
            )

            # Careful with JSON dumping arbitrary objects
            try:
                cleaned_args = _clean_obj(args)
                cleaned_kwargs = _clean_obj(kwargs)
                obs.input_text = json.dumps(
                    {"args": cleaned_args, "kwargs": cleaned_kwargs}
                )
            except Exception:
                obs.input_text = json.dumps({"args": str(args), "kwargs": str(kwargs)})

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

                tool_decisions = _extract_tool_calls(result)
                if tool_decisions:
                    try:
                        current_meta = (
                            json.loads(obs.metadata_json) if obs.metadata_json else {}
                        )
                    except:
                        current_meta = {}

                    current_meta["tool_calls"] = tool_decisions
                    obs.metadata_json = json.dumps(current_meta, default=str)
            else:
                obs.token_usage = None
                obs.model = _extract_model_name(kwargs, None)
                obs.total_cost = None

            obs.model_parameters = None  # Could extract from kwargs if needed

            if exc:
                obs.error = str(exc)

            try:
                exporter.enqueue(obs)
            except Exception as obs_exc:
                # do not break tracing if observation fails
                print(f"[ObsWarning] Failed to create observation: {obs_exc}")

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def inner(*args, **kwargs):
                start_time = time.time()
                obs_id = random.getrandbits(63)
                parent_obs_id = _current_observation_id.get()
                token = _current_observation_id.set(obs_id)
                parent_context = get_current()

                with tracer.start_as_current_span(
                    func_name, context=parent_context, kind=SpanKind.INTERNAL
                ) as span:
                    span.set_attribute("observation_id", str(obs_id))
                    if parent_obs_id:
                        span.set_attribute("parent_observation_id", str(parent_obs_id))

                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)

                    if as_type and as_type.lower() == "runner":
                        capture_candidate_agents()
                        capture_tools()

                    try:
                        result = await func(*args, **kwargs)
                        create_observation_record(
                            span,
                            args,
                            kwargs,
                            start_time,
                            obs_id,
                            parent_obs_id,
                            result=result,
                        )
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR))
                        create_observation_record(
                            span,
                            args,
                            kwargs,
                            start_time,
                            obs_id,
                            parent_obs_id,
                            exc=exc,
                        )
                        raise
                    finally:
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )
                        _current_observation_id.reset(token)

            return inner
        else:

            @functools.wraps(func)
            def inner(*args, **kwargs):
                start_time = time.time()
                obs_id = random.getrandbits(63)
                parent_obs_id = _current_observation_id.get()
                token = _current_observation_id.set(obs_id)
                parent_context = get_current()

                with tracer.start_as_current_span(
                    func_name, context=parent_context, kind=SpanKind.INTERNAL
                ) as span:
                    span.set_attribute("observation_id", str(obs_id))
                    if parent_obs_id:
                        span.set_attribute("parent_observation_id", str(parent_obs_id))

                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)

                    if as_type and as_type.lower() == "runner":
                        capture_candidate_agents()
                        capture_tools()

                    try:
                        result = func(*args, **kwargs)
                        create_observation_record(
                            span,
                            args,
                            kwargs,
                            start_time,
                            obs_id,
                            parent_obs_id,
                            result=result,
                        )
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR))
                        create_observation_record(
                            span,
                            args,
                            kwargs,
                            start_time,
                            obs_id,
                            parent_obs_id,
                            exc=exc,
                        )
                        raise
                    finally:
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )
                        _current_observation_id.reset(token)

            return inner

    return wrapper


def record_score(
    name: str,
    score: float,
    trace_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
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
        end_time=int(start_time * 1e9),  # Instantaneous
        input_text=json.dumps(meta) if meta else None,
        output_text=str(score),  # Store score as string in output_text
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


def capture_candidate_agents(agents: Any = None):
    """
    Explicitly capture candidate agents for the current span.
    If 'agents' is None, it defaults to all registered agents (decorated with @observe(as_agent=True)).

    Args:
        agents: The candidate agents (list of strings, structs, or JSON-serializable objects).
                If None, uses _REGISTERED_AGENTS.
    """
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return

    try:
        # Default to registered agents if not provided
        if agents is None:
            agents = list(_REGISTERED_AGENTS.values())

        # Use _clean_obj to safely prepare the agents for serialization
        cleaned_agents = _clean_obj(agents)
        span.set_attribute("candidate_agents", json.dumps(cleaned_agents, default=str))
    except Exception as e:
        print(f"[ObservixWarning] Failed to capture candidate agents: {e}")


def capture_tools(tools: Any = None):
    """
    Explicitly capture tools for the current span.
    If 'tools' is None, it defaults to all registered tools (decorated with @observe(as_tool=True)).

    Args:
        tools: The tools available or used (list of strings, structs, or JSON-serializable objects).
                If None, uses _REGISTERED_TOOLS.
    """
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return

    try:
        # Default to registered tools if not provided
        if tools is None:
            tools = list(_REGISTERED_TOOLS.values())

        # Use _clean_obj to safely prepare the tools for serialization
        cleaned_tools = _clean_obj(tools)
        span.set_attribute(
            "tools", json.dumps(cleaned_tools, default=str)
        )  # Use "tools" as attribute name
    except Exception as e:
        print(f"[ObservixWarning] Failed to capture tools: {e}")


def get_current_observation_id() -> Optional[int]:
    """Returns the current observation ID from the context."""
    return _current_observation_id.get()


from contextlib import contextmanager


@contextmanager
def observation_context(observation_id: Optional[int]):
    """
    Context manager to manually set the current observation ID.
    Useful for propagating context across execution boundaries (e.g. in graph nodes).
    """
    token = _current_observation_id.set(observation_id)
    try:
        yield
    finally:
        _current_observation_id.reset(token)
