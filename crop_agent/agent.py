from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Literal

import dotenv
from google.adk.agents import Agent

dotenv.load_dotenv()

# ── Load Disease Database ──────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "crop_diseases.json")


def _load_db() -> list[dict]:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


DISEASE_DATABASE = _load_db()


# ── Function Tools ─────────────────────────────────────────────────────────

def search_crop_diseases(
    crop: Literal["paddy", "chilli", "tomato", "cinnamon", "tea", "maize", "potato", "pepper"],
    symptom_keyword: str | None = None,
) -> dict:
    """Search the crop disease & pest database for matching diagnoses and management plans.

    Args:
        crop: Type of crop being cultivated.
        symptom_keyword: Optional symptom keyword (e.g. 'spots', 'yellowing', 'curling', 'blast', 'burn').

    Returns:
        Matching disease records, favorable conditions, organic/chemical treatments, and prevention rules.
    """
    matches = []
    crop_lower = crop.lower().strip()
    kw_lower = symptom_keyword.lower().strip() if symptom_keyword else None
    database = _load_db()

    for record in database:
        if record["crop"].lower() == crop_lower:
            if kw_lower:
                symptoms_str = " ".join(record.get("symptoms", [])).lower()
                name_str = record.get("disease_name", "").lower()
                cat_str = record.get("category", "").lower()

                if not (kw_lower in symptoms_str or kw_lower in name_str or kw_lower in cat_str):
                    continue
            matches.append(record)

    if not matches:
        return {
            "status": "error",
            "message": f"No disease record found for crop '{crop}' with keyword '{symptom_keyword or 'all'}'.",
        }

    return {
        "status": "success",
        "count": len(matches),
        "crop": crop,
        "results": matches,
    }


import time
import ssl

DISTRICT_COORDINATES = {
    "anuradhapura": {"name": "Anuradhapura, Sri Lanka", "lat": 8.3114, "lon": 80.4037},
    "nuwara eliya": {"name": "Nuwara Eliya, Sri Lanka", "lat": 6.9497, "lon": 80.7891},
    "jaffna": {"name": "Jaffna, Sri Lanka", "lat": 9.6615, "lon": 80.0255},
    "kandy": {"name": "Kandy, Sri Lanka", "lat": 7.2906, "lon": 80.6337},
    "kurunegala": {"name": "Kurunegala, Sri Lanka", "lat": 7.4863, "lon": 80.3647},
    "matale": {"name": "Matale, Sri Lanka", "lat": 7.4675, "lon": 80.6234},
    "badulla": {"name": "Badulla, Sri Lanka", "lat": 6.9934, "lon": 81.0550},
    "colombo": {"name": "Colombo, Sri Lanka", "lat": 6.9271, "lon": 79.8612},
    "gampaha": {"name": "Gampaha, Sri Lanka", "lat": 7.0840, "lon": 79.9925},
    "kalutara": {"name": "Kalutara, Sri Lanka", "lat": 6.5854, "lon": 79.9607},
    "galle": {"name": "Galle, Sri Lanka", "lat": 6.0535, "lon": 80.2210},
    "matara": {"name": "Matara, Sri Lanka", "lat": 5.9549, "lon": 80.5550},
    "hambantota": {"name": "Hambantota, Sri Lanka", "lat": 6.1429, "lon": 81.1212},
    "polonnaruwa": {"name": "Polonnaruwa, Sri Lanka", "lat": 7.9403, "lon": 81.0188},
    "trincomalee": {"name": "Trincomalee, Sri Lanka", "lat": 8.5874, "lon": 81.2152},
    "batticaloa": {"name": "Batticaloa, Sri Lanka", "lat": 7.7170, "lon": 81.7000},
    "ampara": {"name": "Ampara, Sri Lanka", "lat": 7.2886, "lon": 81.6738},
    "puttalam": {"name": "Puttalam, Sri Lanka", "lat": 8.0362, "lon": 79.8283},
    "kegalle": {"name": "Kegalle, Sri Lanka", "lat": 7.2513, "lon": 80.3464},
    "ratnapura": {"name": "Ratnapura, Sri Lanka", "lat": 6.6828, "lon": 80.3992},
    "monaragala": {"name": "Monaragala, Sri Lanka", "lat": 6.8714, "lon": 81.3487},
    "vavuniya": {"name": "Vavuniya, Sri Lanka", "lat": 8.7542, "lon": 80.4982},
    "mannar": {"name": "Mannar, Sri Lanka", "lat": 8.9810, "lon": 79.9044},
    "kilinochchi": {"name": "Kilinochchi, Sri Lanka", "lat": 9.3803, "lon": 80.3970},
    "mullaitivu": {"name": "Mullaitivu, Sri Lanka", "lat": 9.2671, "lon": 80.8142},
}

def _deg_to_compass(deg: float | None) -> str:
    """Convert wind direction in degrees (0-360) to 16-point cardinal compass string."""
    if deg is None:
        return "N/A"
    val = int((deg / 22.5) + 0.5)
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return directions[val % 16]


def _resolve_district(location: str) -> tuple[float, float, str] | None:
    """Normalize location query string and match against 25 Sri Lankan administrative district coordinates."""
    raw = location.lower().strip()
    cleaned = raw.replace("-", " ").replace("_", " ")
    for word in ["district", "city", "town", "province", "sri lanka"]:
        cleaned = cleaned.replace(word, "").strip()

    if cleaned in DISTRICT_COORDINATES:
        info = DISTRICT_COORDINATES[cleaned]
        return info["lat"], info["lon"], info["name"]

    no_space_cleaned = cleaned.replace(" ", "")
    for key, info in DISTRICT_COORDINATES.items():
        if key.replace(" ", "") == no_space_cleaned:
            return info["lat"], info["lon"], info["name"]

    for key, info in DISTRICT_COORDINATES.items():
        if key in cleaned or (len(cleaned) >= 4 and cleaned in key):
            return info["lat"], info["lon"], info["name"]

    return None


WMO_WEATHER_CODES = {
    0: "Clear Sky ☀️",
    1: "Mainly Clear 🌤️",
    2: "Partly Cloudy ⛅",
    3: "Overcast ☁️",
    45: "Fog 🌫️",
    48: "Depositing Rime Fog 🌫️",
    51: "Light Drizzle 🌧️",
    53: "Moderate Drizzle 🌧️",
    55: "Dense Drizzle 🌧️",
    56: "Freezing Drizzle 🌧️",
    57: "Dense Freezing Drizzle 🌧️",
    61: "Slight Rain 🌧️",
    63: "Moderate Rain 🌧️",
    65: "Heavy Rain 🌧️",
    66: "Light Freezing Rain 🌧️",
    67: "Heavy Freezing Rain 🌧️",
    71: "Slight Snow ❄️",
    73: "Moderate Snow ❄️",
    75: "Heavy Snow ❄️",
    77: "Snow Grains ❄️",
    80: "Slight Rain Showers 🌦️",
    81: "Moderate Rain Showers 🌦️",
    82: "Violent Rain Showers ⛈️",
    85: "Slight Snow Showers 🌨️",
    86: "Heavy Snow Showers 🌨️",
    95: "Thunderstorm 🌩️",
    96: "Thunderstorm with Hail ⛈️",
    99: "Heavy Thunderstorm with Hail ⛈️",
}


def get_live_weather(location: str) -> dict:
    """Fetch high-accuracy real-time weather conditions and telemetry for Sri Lankan agricultural districts.

    Args:
        location: City or district name (e.g. 'Anuradhapura', 'Kurunegala', 'Nuwara Eliya', 'Jaffna', 'Kandy').

    Returns:
        Current temperature (°C), relative humidity (%), daily precipitation (mm), wind telemetry, and WMO condition text.
    """
    max_retries = 3
    last_err = None
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    res = _resolve_district(location)
    if res:
        lat, lon, place_name = res
    else:
        # Fallback geocoding with strict country_code=LK filter
        lat, lon, place_name = None, None, None
        for attempt in range(max_retries):
            try:
                encoded_loc = urllib.parse.quote(location)
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1&country_code=LK&language=en&format=json"
                req = urllib.request.Request(geo_url, headers={"User-Agent": "CropAdvisorAgent/1.0"})
                with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as response:
                    geo_data = json.loads(response.read().decode("utf-8"))

                if geo_data.get("results"):
                    top_res = geo_data["results"][0]
                    lat, lon = top_res["latitude"], top_res["longitude"]
                    place_name = f"{top_res.get('name', location)}, {top_res.get('country', 'Sri Lanka')}"
                    break
            except Exception as err:
                last_err = err
                time.sleep(0.5)

        if lat is None or lon is None:
            # Default to Sri Lanka central agricultural coordinates (Anuradhapura) if not resolved
            lat, lon, place_name = 8.3114, 80.4037, f"{location}, Sri Lanka (Fallback)"

    # Step 3: Fetch full telemetry weather forecast from Open-Meteo
    for attempt in range(max_retries):
        try:
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,showers,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,weather_code,apparent_temperature,is_day"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
                f"&timezone=Asia/Colombo"
            )
            w_req = urllib.request.Request(weather_url, headers={"User-Agent": "CropAdvisorAgent/1.0"})
            with urllib.request.urlopen(w_req, timeout=10, context=ssl_ctx) as response:
                weather_data = json.loads(response.read().decode("utf-8"))

            current = weather_data.get("current", {})
            daily = weather_data.get("daily", {})

            w_code = current.get("weather_code", 0)
            condition_text = WMO_WEATHER_CODES.get(w_code, "Partly Cloudy ⛅")

            daily_max = daily.get("temperature_2m_max", [None])[0] if daily.get("temperature_2m_max") else None
            daily_min = daily.get("temperature_2m_min", [None])[0] if daily.get("temperature_2m_min") else None
            daily_precip = daily.get("precipitation_sum", [0.0])[0] if daily.get("precipitation_sum") else 0.0
            rain_prob = daily.get("precipitation_probability_max", [0])[0] if daily.get("precipitation_probability_max") else 0

            wind_dir_deg = current.get("wind_direction_10m")
            wind_cardinal = _deg_to_compass(wind_dir_deg)

            return {
                "status": "success",
                "location": place_name,
                "coordinates": {"latitude": lat, "longitude": lon},
                "temperature_celsius": current.get("temperature_2m"),
                "feels_like_celsius": current.get("apparent_temperature"),
                "daily_temp_max_celsius": daily_max,
                "daily_temp_min_celsius": daily_min,
                "relative_humidity_percent": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation", 0.0),
                "rain_mm": current.get("rain", 0.0),
                "showers_mm": current.get("showers", 0.0),
                "daily_precipitation_sum_mm": round(float(daily_precip or 0.0), 1),
                "rain_probability_percent": int(rain_prob or 0),
                "cloud_cover_percent": current.get("cloud_cover"),
                "pressure_hpa": current.get("pressure_msl"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "wind_direction_deg": wind_dir_deg,
                "wind_direction_cardinal": wind_cardinal,
                "is_daytime": bool(current.get("is_day", 1)),
                "condition": condition_text,
            }
        except Exception as err:
            last_err = err
            time.sleep(0.5)

    return {"status": "error", "message": f"Failed to retrieve weather data: {last_err}"}


def get_agricultural_helpline() -> dict:
    """Get official Sri Lanka Department of Agriculture (DOA) advisory service and hotline details.

    Returns:
        Hotline numbers, official web portal, and Agrarian Service Center advice.
    """
    return {
        "status": "success",
        "hotline": "+94 11 286 1500 (Ministry of Agriculture Advisory Service / Krushi Upades)",
        "department": "Department of Agriculture Sri Lanka (DOA)",
        "website": "https://www.doa.gov.lk",
        "agrarian_centers": "Visit your local Govi Jana Seva (Agrarian Service Center) for free chemical & sample testing.",
    }


# ── Agent Definition ───────────────────────────────────────────────────────

root_agent = Agent(
    name="crop_advisor",
    model="gemini-3.1-flash-lite",
    description="Sri Lankan Agricultural Advisory & Crop Disease Diagnostics Agent.",
    instruction="""
You are the official CropAdvisor virtual assistant specializing in Sri Lankan agriculture and crop diagnostics.
You assist farmers and extension officers with crop disease identification, pest control, weather risk analysis, and agronomic advice.

When a user mentions a crop and symptoms or location:
1. Always call `search_crop_diseases` with the target crop and symptom keyword.
2. If a location is provided (or implied), call `get_live_weather` to check if humidity/temperature levels exacerbate fungal or pest threats.
3. Structure your response clearly using markdown:
   - 🔍 **Diagnostic Identification**: State the likely disease or pest (including English/Sinhala terms if applicable).
   - ⛅ **Environmental & Weather Context**: Explain how current weather/humidity levels impact the condition.
   - 🌿 **Organic & Cultural Management**: Immediate eco-friendly or cultural steps the farmer can take.
   - 🧪 **Recommended Treatment (DOA Standards)**: Specify Department of Agriculture approved treatment and dosages.
   - 📞 **Official Contact**: Include the official +94 11 286 1500 hotline details via `get_agricultural_helpline`.

Always remain encouraging, polite, and practical. Ground every diagnosis strictly in tool outputs.
""",
    tools=[
        search_crop_diseases,
        get_live_weather,
        get_agricultural_helpline,
    ],
)
