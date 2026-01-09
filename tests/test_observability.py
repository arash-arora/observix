import json
from unittest.mock import MagicMock, patch

import pytest

from obs_sdk import init_observability, trace_decorator
from obs_sdk.exporter import ClickHouseTraceExporter
from obs_sdk.schema import Observation, Trace


@pytest.fixture
def mock_engine():
    with patch("obs_sdk.exporter.create_engine") as mock:
        yield mock

@pytest.fixture
def mock_session_maker(mock_engine):
    with patch("obs_sdk.exporter.sessionmaker") as mock:
        session = MagicMock()
        mock.return_value = MagicMock(return_value=session)
        yield mock

@pytest.fixture
def mock_exporter_instance():
    # Reset singleton
    with patch(
        "obs_sdk.instrumentation.exporter_module.observation_exporter_instance",
        new=None
    ):
        yield

@pytest.fixture
def mock_observation_exporter():
    # Mock _insert_observation instead of enqueue to let enqueue logic run
    with patch(
        "obs_sdk.exporter.ClickHouseObservationExporter._insert_observation"
    ):  # Removed unused 'as mock_insert'
        yield

def test_trace_decorator(
    mock_engine,
    mock_session_maker,
    mock_exporter_instance,
    mock_observation_exporter
):
    # Setup
    url = "clickhouse+native://localhost"
    init_observability(url, project_id="test_proj", user_id="test_user")

    @trace_decorator(name="test_func")
    def sample_function(x, y):
        return x + y

    result = sample_function(1, 2)

    # Verify
    assert result == 3
    
    # Wait for executor? Enqueue submits to executor. 
    # But we didn't mock executor, so it's a real ThreadPoolExecutor.
    # We need to wait for it or mock executor to be synchronous.
    # Alternatively, mocking ThreadPoolExecutor to run immediately.
    pass

    # Better approach: access the singleton and mock its executor or _insert
    from obs_sdk.instrumentation import exporter_module
    exporter_instance = exporter_module.observation_exporter_instance
    # We can replace executor with a dummy that runs immediately
    exporter_instance.executor = MagicMock()
    exporter_instance.executor.submit = lambda fn, *args: fn(*args)
    # And mock _insert_observation
    exporter_instance._insert_observation = MagicMock()

    # Re-run logic
    @trace_decorator(name="test_func_2")
    def sample_function_2(x, y):
        return x + y
    
    sample_function_2(1, 2)
    
    assert exporter_instance._insert_observation.called
    observation = exporter_instance._insert_observation.call_args[0][0]
    
    assert isinstance(observation, Observation)
    assert observation.name == "test_func_2"
    # Enqueue should have populated project_id
    assert observation.project_id == "test_proj"
    assert observation.user_id == "test_user"
    
    inputs = json.loads(observation.input_text)
    assert inputs["args"] == [1, 2]

def test_trace_export(mock_engine, mock_session_maker):
    exporter = ClickHouseTraceExporter("url", "proj", "user")
    
    # Create a dummy span
    span = MagicMock()
    span.context.trace_id = 12345
    span.name = "span_name"
    span.start_time = 1_000_000_000 # 1 sec in ns
    span.end_time = 1_001_000_000   # 1.001 sec in ns (1 ms duration)
    span.attributes = {"key": "value"}
    span.events = []
    
    # Export
    exporter.export([span])
    
    # Verify session add called
    session_cls = mock_session_maker.return_value
    session = session_cls()
    assert session.add.called
    trace_obj = session.add.call_args[0][0]
    assert isinstance(trace_obj, Trace)
    assert trace_obj.name == "span_name"
    assert trace_obj.duration_ms == 1.0 
    
