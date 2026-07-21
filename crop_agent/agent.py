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

def get_live_weather(location: str) -> dict:
    """Fetch real-time weather conditions for any Sri Lankan district or global agricultural area.

    Args:
        location: City or district name (e.g. 'Anuradhapura', 'Kurunegala', 'Nuwara Eliya', 'Jaffna', 'Colombo').

    Returns:
        Current temperature (°C), relative humidity (%), precipitation, and wind speed.
    """
    max_retries = 3
    last_err = None

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    for attempt in range(max_retries):
        try:
            # Step 1: Geocoding via Open-Meteo API
            encoded_loc = urllib.parse.quote(location)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1&language=en&format=json"
            
            req = urllib.request.Request(geo_url, headers={"User-Agent": "CropAdvisorAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as response:
                geo_data = json.loads(response.read().decode("utf-8"))

            if not geo_data.get("results"):
                return {"status": "error", "message": f"Location '{location}' could not be geocoded."}

            top_res = geo_data["results"][0]
            lat = top_res["latitude"]
            lon = top_res["longitude"]
            place_name = top_res.get("name", location)
            country = top_res.get("country", "")

            # Step 2: Fetch current weather & relative humidity
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,is_day"
            )
            w_req = urllib.request.Request(weather_url, headers={"User-Agent": "CropAdvisorAgent/1.0"})
            with urllib.request.urlopen(w_req, timeout=10, context=ssl_ctx) as response:
                weather_data = json.loads(response.read().decode("utf-8"))

            current = weather_data.get("current", {})

            return {
                "status": "success",
                "location": f"{place_name}, {country}",
                "coordinates": {"latitude": lat, "longitude": lon},
                "temperature_celsius": current.get("temperature_2m"),
                "relative_humidity_percent": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
            }
        except Exception as err:
            last_err = err
            time.sleep(1)

    return {"status": "error", "message": f"Failed to retrieve weather data: {last_err}"}


def get_agricultural_helpline() -> dict:
    """Get official Sri Lanka Department of Agriculture (DOA) advisory service and hotline details.

    Returns:
        Hotline numbers, official web portal, and Agrarian Service Center advice.
    """
    return {
        "status": "success",
        "hotline": "1920 (Free Agricultural Advisory Service / Krushi Upades)",
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
   - 📞 **Official Contact**: Include the DOA 1920 hotline details via `get_agricultural_helpline`.

Always remain encouraging, polite, and practical. Ground every diagnosis strictly in tool outputs.
""",
    tools=[
        search_crop_diseases,
        get_live_weather,
        get_agricultural_helpline,
    ],
)
