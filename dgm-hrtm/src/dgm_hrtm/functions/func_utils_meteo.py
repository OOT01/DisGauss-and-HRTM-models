import math

import requests


# =========================================================
# METEO FUNCTIONS
# =========================================================
def wind_to_uv(speed, direction_deg):
    """
    Convert meteorological wind convention into Cartesian components.

    Meteorological convention:
    the wind comes FROM direction_deg.
    """
    theta = math.radians(270 - direction_deg)
    u_x = speed * math.cos(theta)
    u_y = speed * math.sin(theta)
    return u_x, u_y


def fetch_meteo_historical(coords, date, hour):
    lat, lon = coords
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": (
            "temperature_2m,relative_humidity_2m,surface_pressure,"
            "wind_speed_10m,wind_direction_10m,precipitation,cloud_cover"
        ),
        "models": "ecmwf_ifs",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    target_time = f"{date}T{hour:02d}:00"
    times = data["hourly"]["time"]

    if target_time not in times:
        raise ValueError(f"Hour {target_time} not found in historical data.")

    i = times.index(target_time)

    wind_speed = data["hourly"]["wind_speed_10m"][i]
    wind_dir = data["hourly"]["wind_direction_10m"][i]
    u_x, u_y = wind_to_uv(wind_speed, wind_dir)

    return {
        "time": times[i],
        "temperature_2m": data["hourly"]["temperature_2m"][i],
        "relative_humidity_2m": data["hourly"]["relative_humidity_2m"][i],
        "surface_pressure": data["hourly"]["surface_pressure"][i],
        "precipitation": data["hourly"]["precipitation"][i],
        "cloud_cover": data["hourly"]["cloud_cover"][i],
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_dir,
        "u_x": u_x,
        "u_y": u_y,
    }


def fetch_meteo_current(coords):
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,surface_pressure,"
            "wind_speed_10m,wind_direction_10m,precipitation,cloud_cover"
        ),
        "hourly": "precipitation_probability",
        "models": "ecmwf_ifs",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    wind_speed = current["wind_speed_10m"]
    wind_dir = current["wind_direction_10m"]
    u_x, u_y = wind_to_uv(wind_speed, wind_dir)

    precip_prob = None
    if "hourly" in data and "precipitation_probability" in data["hourly"]:
        times = data["hourly"]["time"]
        current_time = current["time"]
        if current_time in times:
            i = times.index(current_time)
            precip_prob = data["hourly"]["precipitation_probability"][i]

    return {
        "time": current["time"],
        "temperature_2m": current["temperature_2m"],
        "relative_humidity_2m": current["relative_humidity_2m"],
        "surface_pressure": current["surface_pressure"],
        "precipitation": current["precipitation"],
        "cloud_cover": current["cloud_cover"],
        "precipitation_probability": precip_prob,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_dir,
        "u_x": u_x,
        "u_y": u_y,
    }


def fetch_meteo_forecast(coords, date, hour):
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,surface_pressure,"
            "wind_speed_10m,wind_direction_10m,precipitation,"
            "cloud_cover,precipitation_probability"
        ),
        "models": "ecmwf_ifs",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    target_time = f"{date}T{hour:02d}:00"
    times = data["hourly"]["time"]

    if target_time not in times:
        raise ValueError(f"Hour {target_time} not found in forecast data.")

    i = times.index(target_time)

    wind_speed = data["hourly"]["wind_speed_10m"][i]
    wind_dir = data["hourly"]["wind_direction_10m"][i]
    u_x, u_y = wind_to_uv(wind_speed, wind_dir)

    return {
        "time": times[i],
        "temperature_2m": data["hourly"]["temperature_2m"][i],
        "relative_humidity_2m": data["hourly"]["relative_humidity_2m"][i],
        "surface_pressure": data["hourly"]["surface_pressure"][i],
        "precipitation": data["hourly"]["precipitation"][i],
        "cloud_cover": data["hourly"]["cloud_cover"][i],
        "precipitation_probability": data["hourly"]["precipitation_probability"][i],
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_dir,
        "u_x": u_x,
        "u_y": u_y,
    }


def get_meteo(coords, mode, date=None, hour=None):
    mode = mode.lower()

    if mode == "historical":
        if date is None or hour is None:
            raise ValueError("Historical mode requires Date and Hour.")
        return fetch_meteo_historical(coords, date, hour)

    elif mode == "current":
        return fetch_meteo_current(coords)

    elif mode == "forecast":
        if date is None or hour is None:
            raise ValueError("Forecast mode requires Date and Hour.")
        return fetch_meteo_forecast(coords, date, hour)

    else:
        raise ValueError("Mode must be 'historical', 'current', or 'forecast'.")