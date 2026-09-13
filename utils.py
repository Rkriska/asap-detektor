from __future__ import annotations

from datetime import date, datetime
from typing import Any

# Visual-only PM2.5 bands.
# These are intended for communication in the UI, not for recalculating ISPU
# or modifying the model output.
AQ_BANDS = (
    (15.5, "Baik", "#2E7D32", "🟢"),
    (55.4, "Sedang", "#F9A825", "🟡"),
    (150.4, "Tidak Sehat", "#EF6C00", "🟠"),
    (250.4, "Sangat Tidak Sehat", "#C62828", "🔴"),
    (float("inf"), "Berbahaya", "#6A1B9A", "🟣"),
)


def get_air_quality_label(pm25: float | None) -> dict[str, str]:
    if pm25 is None:
        return {
            "label": "Tidak diketahui",
            "color": "#78909C",
            "emoji": "❔",
        }

    value = float(pm25)
    for upper, label, color, emoji in AQ_BANDS:
        if value <= upper:
            return {
                "label": label,
                "color": color,
                "emoji": emoji,
            }

    return {
        "label": "Tidak diketahui",
        "color": "#78909C",
        "emoji": "❔",
    }


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(value)


def format_distance(km: float | None) -> str:
    if km is None:
        return "-"
    return f"{format_number(km, 1)} km"


def format_date_id(value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, str):
        parsed = None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                return value
        value = parsed

    months = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
        7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
    }

    try:
        return f"{value.day} {months[value.month]} {value.year}"
    except (AttributeError, KeyError):
        return str(value)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None
