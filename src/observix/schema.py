from clickhouse_sqlalchemy import engines, types
from sqlalchemy import Column, DateTime, Float, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Trace(Base):
    __tablename__ = "traces"
    __table_args__ = (
        engines.MergeTree(order_by=('id',)),
    )

    id = Column(types.UInt64, primary_key=True)

    trace_id = Column(String, nullable=False)
    user_id = Column(types.Nullable(String))
    project_id = Column(types.Nullable(String))

    name = Column(String, nullable=False)
    service_name = Column(types.Nullable(String))
    component = Column(types.Nullable(String))

    start_time = Column(types.Int64, nullable=False)
    end_time = Column(types.Int64, nullable=False)

    attributes = Column(types.Nullable(Text))
    events = Column(types.Nullable(Text))
    duration_ms = Column(types.Nullable(Float))

    # 👇 CLICKHOUSE FRIENDLY RELATIONSHIP
    observations = relationship(
        "Observation",
        back_populates="trace",
        primaryjoin="Trace.trace_id == foreign(Observation.trace_id)"
    )


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        engines.MergeTree(order_by=('id',)),
    )

    id = Column(types.UInt64, primary_key=True)

    trace_id = Column(String, nullable=False)
    parent_observation_id = Column(types.Nullable(types.UInt64))

    user_id = Column(types.Nullable(String))
    project_id = Column(types.Nullable(String))

    start_time = Column(types.Int64, nullable=False)
    end_time = Column(types.Int64, nullable=False)

    input_text = Column(types.Nullable(Text))
    output_text = Column(types.Nullable(Text))
    token_usage = Column(types.Nullable(Text))
    type = Column(String, nullable=False)
    model = Column(types.Nullable(String))
    model_parameters = Column(types.Nullable(Text))
    metadata_json = Column(types.Nullable(Text))
    name = Column(types.Nullable(String))
    extra = Column(types.Nullable(Text))
    observation_type = Column(types.Nullable(String))
    error = Column(types.Nullable(String))
    total_cost = Column(types.Nullable(Float))

    created_at = Column(DateTime)

    # 👇 CLICKHOUSE FRIENDLY RELATIONSHIP
    trace = relationship(
        "Trace",
        back_populates="observations",
        primaryjoin="foreign(Observation.trace_id) == Trace.trace_id"
    )
