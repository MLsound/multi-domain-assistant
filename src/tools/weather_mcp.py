"""
Mock MCP (Model Context Protocol) tool for environmental data enrichment.

Replaces the hardcoded dictionary that was embedded in graph.py:48.
In production this would call a real weather/environmental API.
Invoked by RetrievalAgent when the Science domain probability exceeds the
configured threshold (settings.science_threshold).
"""

from __future__ import annotations

import logging

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
        "temperature_celsius": 22.0,
        "humidity_percent": 45.0,
        "precipitation_mm": 0.0,
        "soil_temperature_celsius": 18.5,
        "soil_moisture_percent": 32.0,
    }
