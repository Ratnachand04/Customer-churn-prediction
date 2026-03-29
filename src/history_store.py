"""JSON history store for prediction events."""

import json
from datetime import datetime
from pathlib import Path


def _history_file_path():
    root = Path(__file__).resolve().parent.parent
    history_dir = root / "data" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / "prediction_history.json"


def append_prediction_history(model_name, prediction, confidence, payload):
    """Append a prediction record to JSON history file."""
    history_file = _history_file_path()

    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "prediction": int(prediction),
            "confidence": float(confidence) if confidence is not None else None,
            "input": payload,
        }
    )

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_prediction_history(limit=100):
    """Read latest history records from JSON store."""
    history_file = _history_file_path()
    if not history_file.exists():
        return []

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return list(reversed(data))[: int(limit)]
