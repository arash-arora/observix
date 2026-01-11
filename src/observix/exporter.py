import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import httpx
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from observix.schema import Observation

logger = logging.getLogger(__name__)


def _format_timestamp(nanos: int) -> Optional[str]:
    if not nanos:
        return None
    return datetime.fromtimestamp(nanos / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class HttpTraceExporter(SpanExporter):
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        payload = []
        for span in spans:
            # ClickHouse Map(String, String) requires all values to be strings
            attrs = {}
            if span.attributes:
                for k, v in span.attributes.items():
                    attrs[k] = str(v)

            resource_attrs = {}
            if span.resource and span.resource.attributes:
                for k, v in span.resource.attributes.items():
                    resource_attrs[k] = str(v)

            span_dict = {
                "trace_id": f"{span.context.trace_id:032x}",
                "span_id": f"{span.context.span_id:016x}",
                "parent_span_id": (
                    f"{span.parent.span_id:016x}" if span.parent else None
                ),
                "name": span.name,
                "kind": span.kind.name,
                "start_time": _format_timestamp(span.start_time),
                "end_time": _format_timestamp(span.end_time),
                "status": {
                    "code": span.status.status_code.name,
                    "message": span.status.description
                },
                "attributes": attrs,
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": {k: str(v) for k, v in event.attributes.items()} if event.attributes else {}
                    } for event in span.events
                ],
                "links": [
                    {
                        "context": {
                             "trace_id": f"{link.context.trace_id:032x}",
                             "span_id": f"{link.context.span_id:016x}"
                        },
                        "attributes": {k: str(v) for k, v in link.attributes.items()} if link.attributes else {}
                    } for link in span.links
                ],
                "resource": {
                    "attributes": resource_attrs
                }
            }
            payload.append(span_dict)

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key or ""
        }

        try:
            # Sync HTTP request because SpanExporter.export is blocking
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self.url}/api/v1/ingest/traces", 
                    json=payload, 
                    headers=headers
                )
                response.raise_for_status()
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export traces to backend: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


class HttpObservationExporter:
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.executor = ThreadPoolExecutor(max_workers=1)

    def enqueue(self, observation: Observation):
        # Submit to thread pool to avoid blocking
        self.executor.submit(self._send_observation, observation)

    def _send_observation(self, observation: Observation):
        # Convert SQLAlchemy object or schema object to dict
        # Observation is a pydantic model or SQLAlchemy model?
        # In schema.py it is SQLAlchemy model.
        
        obs_dict = {
            "id": observation.id,
            "trace_id": observation.trace_id,
            "parent_observation_id": observation.parent_observation_id,
            "name": observation.name,
            "type": observation.type,
            "model": observation.model,
            "start_time": observation.start_time, # ns
            "end_time": observation.end_time,     # ns
            "input_text": observation.input_text,
            "output_text": observation.output_text,
            "token_usage": observation.token_usage,
            "model_parameters": observation.model_parameters,
            "metadata_json": observation.metadata_json,
            "extra": observation.extra,
            "observation_type": observation.observation_type,
            # "error": getattr(observation, "error", None), 
            # Error field is not in schema.py but backend expects it?
            # Ah, I added it to backend ClickHouse schema.
            # I should verify schema.py again if it has error.
            # checked schema.py: it does NOT have error.
            # But instrumentation.py sets obs.error = str(exc)
            # So standard python object attr setting works on SA models before flush?
            # Yes, but won't be in __dict__ strictly if not defined in mapper?
            # Actually, instrumentation code sets it. Let's assume it's attached.
            "error": getattr(observation, "error", None),
            "user_id": observation.user_id,
            # project_id is handled by backend via API Key
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key or ""
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self.url}/api/v1/ingest/observations",
                    json=[obs_dict], # API expects list
                    headers=headers
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to export observation to backend: {e}")
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

# Global instance place holder
observation_exporter_instance: Optional[HttpObservationExporter] = None
