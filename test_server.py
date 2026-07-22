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
    print("\n--- Testing GET /api/weather & Multi-Factor Risk Assessment ---")
    for loc in ["Anuradhapura", "Nuwara Eliya", "Jaffna"]:
        res = client.get(f"/api/weather?location={loc}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] == "success":
            assert "temperature_celsius" in data
            assert "disease_risk" in data
            risk = data["disease_risk"]
            assert "level" in risk
            assert "color" in risk
            assert "advice" in risk
            assert "risk_factors" in risk
            print(f"  ✅ Weather endpoint passed for {data['location']}: Risk Level = '{risk['level']}', Factors = {risk['risk_factors']}")
        else:
            print(f"  ⚠️ Weather endpoint returned warning for {loc}: {data.get('message')}")


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
    assert "theme-toggle-btn" in res.text
    assert "toggleTheme()" in res.text
    assert "text/html" in res.headers["content-type"]
    print("✅ HTML Dashboard & Dark Theme Toggle served successfully.")


def test_post_chat():
    print("\n--- Testing POST /api/chat ---")
    # 1. Initial message without session_id
    res1 = client.post("/api/chat", json={"message": "Hello"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "success"
    assert "session_id" in data1
    session_id = data1["session_id"]
    print("✅ Chat endpoint turn 1 passed, session_id:", session_id)

    # 2. Multi-turn message with existing session_id
    res2 = client.post("/api/chat", json={"message": "What is Paddy Blast?", "session_id": session_id})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "success"
    print("✅ Chat endpoint multi-turn passed.")

    # 3. Message with stale/bogus session_id (should auto-create without failing)
    res3 = client.post("/api/chat", json={"message": "Hi again", "session_id": "session_stale_xyz123"})
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["status"] == "success"
    print("✅ Chat endpoint stale session recovery passed.")


if __name__ == "__main__":
    test_get_crops()
    test_get_diseases()
    test_get_weather()
    test_get_helpline()
    test_get_dashboard_html()
    test_post_chat()
    print("\n🎉 ALL SERVER API TESTS PASSED SUCCESSFULLY!")
