import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_get_crops():
    print("--- Testing GET /api/crops ---")
    response = client.get("/api/crops")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "paddy" in data["crops"]
    assert "chilli" in data["crops"]
    print("✅ Supported crops endpoint verified:", data["crops"])


def test_get_diseases():
    print("\n--- Testing GET /api/diseases ---")
    # 1. All diseases
    res_all = client.get("/api/diseases")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["count"] > 0
    print(f"✅ Retrieved {data_all['count']} disease records in total.")

    # 2. Filter by crop
    res_paddy = client.get("/api/diseases?crop=paddy")
    assert res_paddy.status_code == 200
    data_paddy = res_paddy.json()
    assert data_paddy["count"] > 0
    for r in data_paddy["results"]:
        assert r["crop"] == "paddy"
    print("✅ Paddy filter test passed:", [r["disease_name"] for r in data_paddy["results"]])

    # 3. Search symptom keyword
    res_blast = client.get("/api/diseases?search=blast")
    assert res_blast.status_code == 200
    data_blast = res_blast.json()
    assert data_blast["count"] > 0
    print("✅ Symptom search test passed:", [r["disease_name"] for r in data_blast["results"]])


def test_get_weather():
    print("\n--- Testing GET /api/weather ---")
    res = client.get("/api/weather?location=Anuradhapura")
    assert res.status_code == 200
    data = res.json()
    if data["status"] == "success":
        assert "temperature_celsius" in data
        assert "disease_risk" in data
        print(f"✅ Weather endpoint passed for {data['location']}: Risk = {data['disease_risk']['level']}")
    else:
        print(f"⚠️ Weather endpoint returned warning (transient network): {data.get('message')}")


def test_get_helpline():
    print("\n--- Testing GET /api/helpline ---")
    res = client.get("/api/helpline")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "1920" in data["hotline"]
    print("✅ Helpline endpoint verified:", data["hotline"])


def test_get_dashboard_html():
    print("\n--- Testing GET / (HTML Dashboard) ---")
    res = client.get("/")
    assert res.status_code == 200
    assert "Sri Lanka CropAdvisor" in res.text
    assert "text/html" in res.headers["content-type"]
    print("✅ HTML Dashboard served successfully.")


if __name__ == "__main__":
    test_get_crops()
    test_get_diseases()
    test_get_weather()
    test_get_helpline()
    test_get_dashboard_html()
    print("\n🎉 ALL SERVER API TESTS PASSED SUCCESSFULLY!")
