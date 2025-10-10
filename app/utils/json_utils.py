import json
from datetime import datetime, date
from typing import Any


def json_serializer(obj: Any) -> str:
    """Custom JSON serializer that handles datetime objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not JSON serializable")


def serialize_for_database(data: dict) -> dict:
    """Serialize data dictionary for database storage, handling datetime objects."""
    return json.loads(json.dumps(data, default=json_serializer))
