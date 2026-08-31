from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CLOCK_META = {
    "type": "function",
    "function": {
    "name": "clock",
        "description": "Get the current system date and time. Use this whenever you need to know the time, date, day of the week, or make relative time references like 'tomorrow' or 'next week'.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "The IANA time zone name string (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). If not specified or unknown, leave this empty to default to UTC."
                }
            }
        }
    }
}

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