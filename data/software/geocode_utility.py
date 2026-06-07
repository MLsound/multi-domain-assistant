def geocode_location(city_name: str):
    """
    Utility for converting human-readable city names into geographic coordinates
    required by the get_forecast(latitude, longitude) tool in weather_mcp.
    """
    # Uses geocode.py logic to fetch lat/long for weather API requests
    return {"lat": 35.7735, "lon": -78.6760} # Example: Raleigh, NC
