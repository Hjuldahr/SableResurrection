# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)