from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

def clock(zone: str | None = None) -> str:
    """
    Returns the current date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS±HH:MM).
    """
    try:
        if not zone:
            tz = timezone.utc
        else:
            tz = ZoneInfo(zone)
    except ZoneInfoNotFoundError:
        return f"Error: Unknown or invalid time zone '{zone}'."
        
    return datetime.now(tz).isoformat(timespec='seconds')