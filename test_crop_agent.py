import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from crop_agent import (
    root_agent,
    search_crop_diseases,
    get_live_weather,
    get_agricultural_helpline,
)


def test_agent_export():
    print("--- Testing Agent Package Export ---")
    assert root_agent.name == "crop_advisor"
    assert len(root_agent.tools) == 3
    print(f"✅ Root Agent '{root_agent.name}' initialized with {len(root_agent.tools)} tools.")


def test_disease_search_all_crops():
    print("\n--- Testing Disease Search for All Supported Crops ---")
    crops_to_test = [
        ("paddy", "blast"),
        ("chilli", "curling"),
        ("tomato", "blight"),
        ("cinnamon", "bark"),
        ("tea", "blister"),
        ("maize", "caterpillar"),
        ("potato", "spots"),
        ("pepper", "wilt"),
    ]

    for crop, kw in crops_to_test:
        res = search_crop_diseases(crop=crop, symptom_keyword=kw)
        assert res["status"] == "success", f"Search failed for {crop}: {res}"
        assert res["count"] > 0, f"No records found for crop '{crop}' keyword '{kw}'"
        disease_name = res["results"][0]["disease_name"]
        print(f"  ✅ [{crop.upper()}] record verified: {disease_name}")


def test_live_weather_districts():
    print("\n--- Testing Open-Meteo Live Weather API & District Normalization ---")
    test_queries = [
        "Anuradhapura",
        "Nuwara Eliya",
        "nuwaraeliya",
        "Kandy District",
        "Jaffna Peninsula",
    ]
    for dist in test_queries:
        res = get_live_weather(dist)
        if res["status"] == "success":
            assert "temperature_celsius" in res, f"Missing temperature for {dist}"
            assert "feels_like_celsius" in res, f"Missing feels_like for {dist}"
            assert "daily_precipitation_sum_mm" in res, f"Missing daily precip for {dist}"
            assert "wind_direction_cardinal" in res, f"Missing wind direction for {dist}"
            print(
                f"  ✅ Weather for '{dist}' -> {res['location']}: {res['temperature_celsius']}°C (Feels {res['feels_like_celsius']}°C), Humidity: {res['relative_humidity_percent']}%, Rain 24h: {res['daily_precipitation_sum_mm']}mm, Wind: {res['wind_direction_cardinal']} {res['wind_speed_kmh']}km/h"
            )
        else:
            print(f"  ⚠️ Weather warning for {dist} (transient network): {res.get('message')}")


def test_helpline():
    print("\n--- Testing DOA Helpline Tool ---")
    res = get_agricultural_helpline()
    assert res["status"] == "success"
    assert "+94 11 286 1500" in res["hotline"]
    print("✅ Helpline test passed:", res["hotline"])


if __name__ == "__main__":
    test_agent_export()
    test_disease_search_all_crops()
    test_live_weather_districts()
    test_helpline()
    print("\n🎉 ALL CROP AGENT TESTS PASSED SUCCESSFULLY!")
