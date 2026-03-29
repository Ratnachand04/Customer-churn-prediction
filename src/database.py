"""Database utilities for local and deployed environments.

By default this uses SQLite for local development.
For deployment, set DATABASE_URL to a managed Postgres URL.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)


metadata = MetaData()

prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String(255), nullable=False),
    Column("prediction", Integer, nullable=False),
    Column("confidence", Float, nullable=True),
    Column("input_payload", Text, nullable=True),
    Column("created_at", DateTime, nullable=False),
)

training_runs = Table(
    "training_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String(255), nullable=False),
    Column("accuracy", Float, nullable=True),
    Column("precision", Float, nullable=True),
    Column("recall", Float, nullable=True),
    Column("f1_score", Float, nullable=True),
    Column("status", String(255), nullable=False),
    Column("created_at", DateTime, nullable=False),
)


def _default_sqlite_url():
    root = Path(__file__).resolve().parent.parent
    db_dir = root / "data" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_dir / 'churn_app.db'}"


def get_database_url():
    return os.getenv("DATABASE_URL", _default_sqlite_url())


def get_engine():
    return create_engine(get_database_url(), future=True, pool_pre_ping=True)


def init_db():
    """Create required tables if they do not exist."""
    engine = get_engine()
    metadata.create_all(engine)


def log_prediction(model_name, prediction, confidence, input_payload):
    engine = get_engine()
    payload = json.dumps(input_payload, default=str)
    with engine.begin() as conn:
        conn.execute(
            insert(prediction_logs),
            {
                "model_name": model_name,
                "prediction": int(prediction),
                "confidence": float(confidence) if confidence is not None else None,
                "input_payload": payload,
                "created_at": datetime.utcnow(),
            },
        )


def log_training_result(model_name, metrics, status="success"):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(training_runs),
            {
                "model_name": model_name,
                "accuracy": metrics.get("Accuracy") if metrics else None,
                "precision": metrics.get("Precision") if metrics else None,
                "recall": metrics.get("Recall") if metrics else None,
                "f1_score": metrics.get("F1 Score") if metrics else None,
                "status": status,
                "created_at": datetime.utcnow(),
            },
        )


def fetch_recent_predictions(limit=50):
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                prediction_logs.c.model_name,
                prediction_logs.c.prediction,
                prediction_logs.c.confidence,
                prediction_logs.c.created_at,
            )
            .order_by(prediction_logs.c.id.desc())
            .limit(int(limit))
        ).mappings().all()
    return [dict(r) for r in rows]


def fetch_prediction_history(limit=500):
    """Fetch prediction history rows for UI history page."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                prediction_logs.c.model_name,
                prediction_logs.c.prediction,
                prediction_logs.c.confidence,
                prediction_logs.c.created_at,
                prediction_logs.c.input_payload,
            )
            .order_by(prediction_logs.c.id.desc())
            .limit(int(limit))
        ).mappings().all()
    return [dict(r) for r in rows]
