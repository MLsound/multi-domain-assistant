"""
Mock MCP (Model Context Protocol) tool for environmental data enrichment.

Replaces the hardcoded dictionary that was embedded in graph.py:48.
In production this would call a real weather/environmental API.
Invoked by RetrievalAgent when the Science domain probability exceeds the
configured threshold (settings.science_threshold).
"""

from __future__ import annotations

import logging
import random

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_environmental_data() -> dict:
    """Fetch current environmental conditions for science-domain context enrichment.

    Returns temperature, humidity, precipitation, soil temperature and moisture
    relevant to photovoltaic yield and building energy calculations.
    """
    logger.debug("Fetching environmental metadata via MCP mock tool")
    return {
        "temperature_celsius": round(random.uniform(15.0, 35.0), 1),
        "humidity_percent": round(random.uniform(30.0, 80.0), 1),
        "precipitation_mm": round(random.uniform(0.0, 5.0), 1) if random.random() > 0.8 else 0.0,
        "wind_speed_ms": round(random.uniform(0.0, 15.0), 1),
        "wind_direction_degrees": random.randint(0, 359),
        "air_pressure_hpa": round(random.uniform(980.0, 1030.0), 1),
        "soil_temperature_celsius": round(random.uniform(10.0, 25.0), 1),
        "soil_moisture_percent": round(random.uniform(10.0, 50.0), 1),
    }
