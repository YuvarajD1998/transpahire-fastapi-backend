import json
import re
from datetime import datetime, date
from typing import Any


def safe_parse_json(text: str) -> dict:
    """Parse JSON from LLM output with regex fallback for markdown-wrapped responses."""
    if not text:
        raise ValueError("Empty response text")
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(match.group())


def json_serializer(obj: Any) -> str:
    """Custom JSON serializer that handles datetime objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not JSON serializable")


def serialize_for_database(data: dict) -> dict:
    """Serialize data dictionary for database storage, handling datetime objects."""
    return json.loads(json.dumps(data, default=json_serializer))
