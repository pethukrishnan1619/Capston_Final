"""
Tool implementations for the multi‑agent RAG system.

This module defines input schemas using pydantic and exposes functions to call
external APIs or perform computations.  Each tool function returns a
dictionary with an `ok` flag and either a `result` or `error` field.

Available tools:
* Weather forecast (using the Open‑Meteo API via a geocoding step)
* Calculator supporting arithmetic and common scientific operations
* Custom text processing functions (word count, extract numbers, case conversion, etc.)
"""

from __future__ import annotations

import re
import math
import requests
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "WeatherToolInput",
    "weather_tool_call",
    "CalculationInput",
    "calculator_tool_call",
    "PythonCustomToolInput",
    "python_custom_tool_call",
]


class WeatherToolInput(BaseModel):
    """Input schema for the weather tool."""
    location: str = Field(..., description="City name like Chennai, Mumbai, London")
    days: int = Field(3, ge=1, le=7, description="Forecast days (1 to 7)")


def weather_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the Open‑Meteo API to fetch a weather forecast.

    This function performs a two‑step process:
    1. Geocode the location name to latitude and longitude using the Open‑Meteo
       geocoding API.  If no results are found, an error is returned.
    2. Fetch a multi‑day forecast for the resolved coordinates.  The forecast
       includes maximum and minimum temperatures, precipitation and wind speed.

    Args:
        data: A dictionary containing keys ``location`` and ``days``.

    Returns:
        A dictionary with ``ok`` flag, and on success, keys ``location``,
        ``forecast_days`` and ``forecast``.  On failure, returns ``ok=False``
        and an ``error`` message.
    """
    try:
        inp = WeatherToolInput(**data)
    except ValidationError as e:
        return {"ok": False, "error": str(e)}

    loc = inp.location.strip()
    if not loc:
        return {"ok": False, "error": "Location cannot be empty."}

    # Geocode
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        gr = requests.get(geo_url, params={"name": loc, "count": 1}, timeout=15)
        gr.raise_for_status()
        gj = gr.json()
    except Exception as e:
        return {"ok": False, "error": f"Geocoding failed: {e}"}

    if not gj.get("results"):
        return {"ok": False, "error": f"Location not found: {loc}"}

    place = gj["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    # Forecast
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "forecast_days": inp.days,
        "timezone": "auto",
    }
    try:
        fr = requests.get(forecast_url, params=params, timeout=20)
        fr.raise_for_status()
        fj = fr.json()
    except Exception as e:
        return {"ok": False, "error": f"Forecast failed: {e}"}

    daily = fj.get("daily", {})
    out: List[Dict[str, Any]] = []
    n = len(daily.get("time", []))
    for i in range(n):
        out.append({
            "date": daily["time"][i],
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "precip_mm": daily["precipitation_sum"][i],
            "wind_max_kmh": daily["wind_speed_10m_max"][i],
        })

    return {
        "ok": True,
        "location": f"{place.get('name')}, {place.get('country')}",
        "forecast_days": inp.days,
        "forecast": out,
    }


class CalculationInput(BaseModel):
    """Input schema for the calculator tool."""
    expression: str = Field(
        ..., description="Math expression, e.g. sin(pi/2) + sqrt(16) - 3**2"
    )


def calculator_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate an arithmetic or scientific expression.

    Supported operations include +, -, *, /, %, ** and common mathematical
    functions such as sqrt, log, sin, cos, tan, exp, factorial.  Constants
    ``pi`` and ``e`` are also available.  The expression is evaluated in a
    restricted environment with no access to Python built‑ins to prevent
    arbitrary code execution.

    Args:
        data: A dictionary with a single key ``expression``.

    Returns:
        A dictionary with ``ok`` flag and ``result`` on success, or ``error``
        on failure.
    """
    try:
        inp = CalculationInput(**data)
    except ValidationError as e:
        return {"ok": False, "error": str(e)}

    expr = inp.expression.strip()
    if not expr:
        return {"ok": False, "error": "Expression cannot be empty."}

    allowed_names: Dict[str, Any] = {
        "abs": abs,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "factorial": math.factorial,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"Calculation failed: {e}"}


class PythonCustomToolInput(BaseModel):
    """Input schema for the custom Python tool with ten operations."""
    operation: str = Field(
        ...,
        description=(
            "Allowed operations: "
            "word_count | extract_numbers | title_case | reverse_text | "
            "lower_case | upper_case | remove_punctuation | "
            "remove_extra_spaces | sentence_count | unique_words"
        ),
    )
    text: str = Field(..., description="Input text to process")


def python_custom_tool_call(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one of ten safe text processing operations.

    The following operations are supported:
    ``word_count`` returns counts of words and characters (with and without spaces);
    ``extract_numbers`` returns all numeric substrings; ``title_case`` capitalises
    each word; ``reverse_text`` reverses the string; ``lower_case`` and
    ``upper_case`` adjust case; ``remove_punctuation`` strips punctuation;
    ``remove_extra_spaces`` collapses multiple spaces; ``sentence_count`` counts
    sentences based on punctuation; and ``unique_words`` returns sorted unique
    words (case‑insensitive).

    Args:
        data: Dictionary with ``operation`` and ``text`` keys.

    Returns:
        A dict with ``ok`` flag and ``result`` or ``error``.
    """
    try:
        inp = PythonCustomToolInput(**data)
    except ValidationError as e:
        return {"ok": False, "error": str(e)}

    operation = inp.operation.lower().strip()
    text = inp.text or ""

    try:
        if operation == "word_count":
            words = re.findall(r"\b\w+\b", text)
            return {
                "ok": True,
                "operation": operation,
                "result": {
                    "word_count": len(words),
                    "character_count": len(text),
                    "character_count_no_spaces": len(text.replace(" ", "")),
                },
            }

        elif operation == "extract_numbers":
            numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            return {"ok": True, "operation": operation, "result": numbers}

        elif operation == "title_case":
            return {"ok": True, "operation": operation, "result": text.title()}

        elif operation == "reverse_text":
            return {"ok": True, "operation": operation, "result": text[::-1]}

        elif operation == "lower_case":
            return {"ok": True, "operation": operation, "result": text.lower()}

        elif operation == "upper_case":
            return {"ok": True, "operation": operation, "result": text.upper()}

        elif operation == "remove_punctuation":
            cleaned = re.sub(r"[^\w\s]", "", text)
            return {"ok": True, "operation": operation, "result": cleaned}

        elif operation == "remove_extra_spaces":
            cleaned = re.sub(r"\s+", " ", text).strip()
            return {"ok": True, "operation": operation, "result": cleaned}

        elif operation == "sentence_count":
            endings = re.findall(r"[.!?]+", text)
            return {"ok": True, "operation": operation, "result": len(endings)}

        elif operation == "unique_words":
            words = re.findall(r"\b\w+\b", text.lower())
            return {"ok": True, "operation": operation, "result": sorted(list(set(words)))}

        else:
            return {"ok": False, "error": f"Unsupported operation: {operation}"}

    except Exception as e:
        return {"ok": False, "error": str(e)}