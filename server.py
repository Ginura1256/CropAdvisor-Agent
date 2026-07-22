from __future__ import annotations

import os
import uuid
import asyncio
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from crop_agent.agent import (
    root_agent,
    search_crop_diseases,
    get_live_weather,
    get_agricultural_helpline,
    _load_db,
)

app = FastAPI(
    title="Sri Lanka CropAdvisor Agricultural Advisory Portal",
    description="REST API and Interactive Web Dashboard with AI Chat for Crop Diagnostics.",
    version="1.0.0",
)

SUPPORTED_CROPS = ["paddy", "chilli", "tomato", "cinnamon", "tea", "maize", "potato", "pepper"]

# ADK Session Service & Runner Initialization
session_service = InMemorySessionService()
runner = Runner(
    app_name="crop_agent",
    agent=root_agent,
    session_service=session_service,
    auto_create_session=True,
)



class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str = "web_user"


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.get("/api/crops")
def get_supported_crops():
    """Retrieve list of supported crops."""
    return {"status": "success", "crops": SUPPORTED_CROPS}


@app.get("/api/diseases")
def get_crop_diseases(
    crop: str | None = None,
    search: str | None = None,
):
    """Search crop disease records with optional crop and symptom filtering."""
    db = _load_db()
    results = db

    if crop and crop.lower() != "all":
        results = [r for r in results if r.get("crop", "").lower() == crop.lower()]

    if search:
        s = search.lower().strip()
        filtered = []
        for record in results:
            symptoms_str = " ".join(record.get("symptoms", [])).lower()
            name_str = record.get("disease_name", "").lower()
            cat_str = record.get("category", "").lower()
            if s in symptoms_str or s in name_str or s in cat_str or s in record.get("crop", "").lower():
                filtered.append(record)
        results = filtered

    return {
        "status": "success",
        "count": len(results),
        "total_count": len(db),
        "results": results,
    }


def evaluate_disease_risk(weather_data: dict) -> dict:
    """Evaluate multi-factor agricultural disease & pest risk based on live weather telemetry."""
    temp = weather_data.get("temperature_celsius") or 25.0
    humidity = weather_data.get("relative_humidity_percent") or 50
    daily_precip = weather_data.get("daily_precipitation_sum_mm") or 0.0
    current_precip = weather_data.get("precipitation_mm") or 0.0
    wind_speed = weather_data.get("wind_speed_kmh") or 0.0
    rain_prob = weather_data.get("rain_probability_percent") or 0

    risk_factors = []
    level = "Low Risk"
    color = "#10b981"
    advice = "Favorable agricultural conditions. Maintain standard irrigation and routine field monitoring."

    # Priority 1: Heavy Rainfall / Waterlogging Alert (Requires actual heavy rainfall >= 25mm daily or >= 10mm current)
    if daily_precip >= 25.0 or current_precip >= 10.0:
        risk_factors.append("heavy_rainfall")
        level = "High Flood & Bacterial Risk"
        color = "#ef4444"
        if current_precip >= 10.0 and daily_precip < 25.0:
            advice = f"High-intensity rain burst ({current_precip} mm/hr) detected! Ensure immediate field drainage to prevent root rot and bacterial wilt."
        else:
            advice = f"Heavy daily rainfall ({daily_precip} mm) detected! Ensure immediate field drainage to prevent root rot, bacterial wilt, and soil nutrient leaching."

    # Priority 2: Moderate Rain & Waterlogging Risk
    elif daily_precip >= 10.0 or (rain_prob >= 85 and daily_precip >= 5.0):
        risk_factors.append("moderate_rainfall")
        level = "Moderate Rain & Waterlogging Risk"
        color = "#f59e0b"
        advice = f"Moderate rainfall ({daily_precip} mm) detected or high rain probability ({rain_prob}%). Ensure field drainage channels are clear."

    # Priority 3: High Fungal Blast / Blight Risk (High Humidity > 85%)
    elif humidity >= 85:
        risk_factors.append("high_humidity")
        if temp < 20.0:
            level = "Critical Blight Risk (Cold Damp)"
            color = "#ef4444"
            advice = "High humidity (>85%) and cool temperatures (<20°C) strongly favor Potato/Tomato Late Blight and Tea Blister Blight. Apply preventive copper fungicide."
        else:
            level = "High Fungal Risk"
            color = "#ef4444"
            advice = "Relative humidity exceeds 85%. Fungal blast, downy mildew, and sheath blight threats are severe. Avoid evening overhead watering."

    # Priority 4: Moderate Fungal/Disease Risk (Humidity 70-84%)
    elif humidity >= 70:
        risk_factors.append("moderate_humidity")
        level = "Moderate Disease Risk"
        color = "#f59e0b"
        advice = "Moderate relative humidity. Maintain field sanitation, monitor leaf undersides, and avoid excessive nitrogen/Urea applications."

    # Priority 5: Heat Stress & Thrips/Mites Pest Risk (Temp > 32°C and Low Humidity < 55%)
    elif temp >= 32.0 and humidity <= 55:
        risk_factors.append("heat_pest_stress")
        level = "High Pest & Drought Stress"
        color = "#f59e0b"
        advice = "High temperatures (>32°C) and low humidity trigger Chilli Thrips, Spider Mites, and crop wilt. Maintain adequate field moisture and shade nurseries."

    # Secondary Factor: Wind Spore Dispersal Warning
    if wind_speed >= 25.0:
        risk_factors.append("high_wind")
        advice += f" ⚠️ Strong winds ({wind_speed} km/h) accelerate fungal spore dispersal across adjacent fields."

    return {
        "level": level,
        "color": color,
        "advice": advice,
        "risk_factors": risk_factors,
    }


@app.get("/api/weather")
def fetch_weather(location: str = Query("Anuradhapura", description="District or city name")):
    """Fetch live weather conditions and risk analysis for agricultural districts."""
    res = get_live_weather(location)
    if res.get("status") == "error":
        return res

    res["disease_risk"] = evaluate_disease_risk(res)
    return res


@app.get("/api/helpline")
def fetch_helpline():
    """Retrieve official Sri Lanka Department of Agriculture helpline details."""
    return get_agricultural_helpline()


def generate_fallback_response(user_msg: str) -> str:
    """Generate structured agricultural response when AI model quota rate limit (429) occurs."""
    msg_lower = user_msg.lower()
    
    detected_crop = None
    for crop in SUPPORTED_CROPS:
        if crop in msg_lower:
            detected_crop = crop
            break

    keywords = ["blight", "blast", "curl", "canker", "wilt", "spots", "caterpillar", "armyworm", "yellowing", "rot", "blister", "burn"]
    detected_kw = None
    for kw in keywords:
        if kw in msg_lower:
            detected_kw = kw
            break

    locations = ["anuradhapura", "nuwara eliya", "jaffna", "kandy", "colombo", "kurunegala", "badulla", "matale", "gampaha", "galle", "hambantota", "polonnaruwa", "trincomalee", "batticaloa", "ampara", "puttalam", "kegalle", "ratnapura", "monaragala", "vavuniya", "mannar", "kilinochchi", "mullaitivu"]
    detected_loc = "Anuradhapura"
    for loc in locations:
        if loc in msg_lower:
            detected_loc = loc.title()
            break

    if detected_crop:
        res = search_crop_diseases(crop=detected_crop, symptom_keyword=detected_kw)
    else:
        db = _load_db()
        matched = []
        for r in db:
            if detected_kw and (detected_kw in " ".join(r.get("symptoms", [])).lower() or detected_kw in r.get("disease_name", "").lower()):
                matched.append(r)
        res = {"status": "success", "results": matched} if matched else {"status": "error"}

    weather = get_live_weather(detected_loc)
    helpline = get_agricultural_helpline()

    lines = []
    lines.append("> 💡 *Notice: Google Gemini API quota/rate limit is active (429). Served seamlessly via CropAdvisor Offline Advisory Database.*\n")

    if res.get("status") == "success" and res.get("results"):
        rec = res["results"][0]
        lines.append(f"🔍 **Diagnostic Identification**: {rec.get('disease_name')} ({rec.get('crop').title()})")
        lines.append(f"• **Symptoms**: {', '.join(rec.get('symptoms', []))}")
        lines.append(f"• **Favorable Conditions**: {rec.get('favorable_conditions')}\n")

        if weather.get("status") == "success":
            lines.append(f"⛅ **Environmental & Weather Context ({weather.get('location')})**:")
            lines.append(f"• Temperature: {weather.get('temperature_celsius')}°C | Relative Humidity: {weather.get('relative_humidity_percent')}%")
            if (weather.get("relative_humidity_percent") or 0) > 80:
                lines.append("• ⚠️ *High humidity (>80%) exacerbates fungal blast and blight spore spread. Ensure field drainage immediately.*")
            lines.append("")

        lines.append(f"🌿 **Organic & Cultural Management**:")
        for org in rec.get("organic_treatments", []):
            lines.append(f"• {org}")
        lines.append("")

        lines.append(f"🧪 **Recommended Treatment (DOA Standards)**:")
        for chem in rec.get("chemical_treatments", []):
            lines.append(f"• {chem}")
        lines.append("")

        lines.append(f"📞 **Official Contact**:\n• {helpline.get('hotline')}\n• {helpline.get('agrarian_centers')}")
    else:
        lines.append(f"🌱 **CropAdvisor Agricultural Advisory**\n")
        lines.append(f"For crop diagnostics regarding '{user_msg}', please mention one of our supported crops: **Paddy, Chilli, Tomato, Cinnamon, Tea, Maize, Potato, Pepper**.\n")
        if weather.get("status") == "success":
            lines.append(f"⛅ **Live Weather for {weather.get('location')}**: {weather.get('temperature_celsius')}°C, Humidity: {weather.get('relative_humidity_percent')}%\n")
        lines.append(f"📞 **DOA Hotline**: {helpline.get('hotline')}")

    return "\n".join(lines)


@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest):
    """Interact with the CropAdvisor AI Agent directly via REST API."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    sess_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"

    # Ensure session exists in ADK session service
    sess = await session_service.get_session(app_name="crop_agent", user_id=req.user_id, session_id=sess_id)
    if sess is None:
        await session_service.create_session(app_name="crop_agent", user_id=req.user_id, session_id=sess_id)

    user_content = Content(role="user", parts=[Part.from_text(text=req.message.strip())])

    async def _execute_runner(target_sess_id: str):
        ai_responses = []
        async for event in runner.run_async(user_id=req.user_id, session_id=target_sess_id, new_message=user_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        ai_responses.append(part.text)
        return "\n".join(ai_responses).strip() if ai_responses else "No response generated by agent."

    try:
        full_text = await _execute_runner(sess_id)
    except Exception as err:
        err_str = str(err)
        if "Session not found" in err_str or "SessionNotFoundError" in type(err).__name__:
            new_sess_id = f"session_{uuid.uuid4().hex[:8]}"
            await session_service.create_session(app_name="crop_agent", user_id=req.user_id, session_id=new_sess_id)
            try:
                full_text = await _execute_runner(new_sess_id)
                sess_id = new_sess_id
            except Exception as retry_err:
                full_text = generate_fallback_response(req.message)
        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            full_text = generate_fallback_response(req.message)
        elif "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
            return {
                "status": "error",
                "message": "Invalid or placeholder Google API Key detected. Please open your `.env` file in the project folder and replace `GOOGLE_API_KEY` with a valid key from Google AI Studio (https://aistudio.google.com/).",
                "session_id": sess_id,
            }
        else:
            full_text = generate_fallback_response(req.message)

    return {
        "status": "success",
        "response": full_text,
        "session_id": sess_id,
    }


# ── Web Dashboard UI HTML ──────────────────────────────────────────────────────

CROP_ADVISOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sri Lanka CropAdvisor — Agricultural Advisory & AI Diagnostics Portal</title>
    <meta name="description" content="Official AI-powered agricultural advisory portal for Sri Lankan farmers. Diagnose crop diseases, chat with the AI Crop Advisor, monitor district live weather risks, and access DOA helpline support.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-canvas: #f8fafc;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
            --card-border-emerald: rgba(16, 185, 129, 0.3);
            --primary-emerald: #059669;
            --primary-dark: #047857;
            --accent-teal: #0d9488;
            --mint-light: #ecfdf5;
            --mint-border: #a7f3d0;
            --text-heading: #0f172a;
            --text-body: #334155;
            --text-muted: #64748b;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            --shadow-lg: 0 10px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
            --nav-bg: #f1f5f9;
        }

        [data-theme="dark"] {
            --bg-canvas: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --card-border-emerald: rgba(52, 211, 153, 0.3);
            --primary-emerald: #10b981;
            --primary-dark: #34d399;
            --accent-teal: #14b8a6;
            --mint-light: rgba(16, 185, 129, 0.15);
            --mint-border: rgba(16, 185, 129, 0.3);
            --text-heading: #f9fafb;
            --text-body: #d1d5db;
            --text-muted: #9ca3af;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 12px 30px rgba(0, 0, 0, 0.5);
            --nav-bg: #1f2937;
        }

        /* ── Dark Theme Element Overrides ── */
        [data-theme="dark"] body {
            background-image: 
                radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(20, 184, 166, 0.1) 0px, transparent 50%);
        }

        [data-theme="dark"] .nav-tabs {
            background: var(--nav-bg);
            border-color: #1f2937;
        }

        [data-theme="dark"] .nav-tab.active {
            background: #111827;
            color: #34d399;
            border-color: rgba(16, 185, 129, 0.4);
        }

        [data-theme="dark"] .helpline-badge {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(20, 184, 166, 0.1));
            border-color: rgba(16, 185, 129, 0.3);
        }

        [data-theme="dark"] .helpline-badge .number {
            color: #34d399;
        }

        [data-theme="dark"] .msg-ai {
            background: #1f2937;
            border-color: #374151;
            color: #f3f4f6;
        }

        [data-theme="dark"] .btn-reset {
            background: #1f2937;
            border-color: #374151;
            color: #e5e7eb;
        }

        [data-theme="dark"] .chat-input-row input {
            background: #1f2937;
            border-color: #374151;
            color: #f9fafb;
        }

        [data-theme="dark"] .prompt-btn {
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.25);
            color: #34d399;
        }

        [data-theme="dark"] .prompt-btn:hover {
            background: rgba(16, 185, 129, 0.25);
        }

        [data-theme="dark"] .district-select {
            background: #1f2937;
            border-color: #374151;
            color: #f9fafb;
        }

        [data-theme="dark"] .temp-box {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.25) 0%, rgba(154, 52, 18, 0.2) 100%);
            border-color: rgba(249, 115, 22, 0.4);
        }
        [data-theme="dark"] .humidity-box {
            background: linear-gradient(135deg, rgba(12, 74, 110, 0.3) 0%, rgba(3, 105, 161, 0.2) 100%);
            border-color: rgba(56, 189, 248, 0.4);
        }
        [data-theme="dark"] .rain-box {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(4, 120, 87, 0.2) 100%);
            border-color: rgba(52, 211, 153, 0.4);
        }
        [data-theme="dark"] .wind-box {
            background: linear-gradient(135deg, rgba(49, 46, 129, 0.3) 0%, rgba(67, 56, 202, 0.2) 100%);
            border-color: rgba(167, 139, 250, 0.4);
        }

        [data-theme="dark"] .risk-alert-box.risk-high {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(153, 27, 27, 0.2) 100%);
            border-color: rgba(239, 68, 68, 0.4);
        }
        [data-theme="dark"] .risk-alert-box.risk-moderate {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(146, 64, 14, 0.2) 100%);
            border-color: rgba(245, 158, 11, 0.4);
        }
        [data-theme="dark"] .risk-alert-box.risk-low {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(4, 120, 87, 0.2) 100%);
            border-color: rgba(16, 185, 129, 0.4);
        }

        [data-theme="dark"] .risk-advice-text {
            color: #e2e8f0;
        }

        [data-theme="dark"] .db-search-input {
            background: #1f2937;
            border-color: #374151;
            color: #f9fafb;
        }

        [data-theme="dark"] .disease-card {
            background: #111827;
            border-color: #1f2937;
        }

        [data-theme="dark"] .symptoms-ul li {
            color: #e5e7eb;
        }

        [data-theme="dark"] .disease-subtitle {
            color: #9ca3af;
        }

        [data-theme="dark"] .sec-organic {
            background: rgba(77, 124, 15, 0.15);
            border-color: rgba(163, 230, 53, 0.3);
            color: #ecfccb;
        }
        [data-theme="dark"] .sec-chemical {
            background: rgba(15, 118, 110, 0.15);
            border-color: rgba(45, 212, 191, 0.3);
            color: #ccfbf1;
        }
        [data-theme="dark"] .sec-prevention {
            background: rgba(194, 65, 12, 0.15);
            border-color: rgba(251, 146, 60, 0.3);
            color: #ffedd5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-canvas);
            background-image: 
                radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.07) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(13, 148, 136, 0.06) 0px, transparent 50%);
            color: var(--text-body);
            min-height: 100vh;
            padding: 1.5rem 1rem;
            line-height: 1.6;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
        }

        /* ── Header Navbar ── */
        header {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.1rem 1.75rem;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1.25rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-md);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-icon {
            font-size: 2.2rem;
            background: var(--mint-light);
            border: 1px solid var(--mint-border);
            width: 52px;
            height: 52px;
            border-radius: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .brand h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.7rem;
            font-weight: 800;
            color: var(--text-heading);
            letter-spacing: -0.02em;
        }

        .brand h1 span {
            color: var(--primary-emerald);
        }

        .brand p {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 500;
        }

        /* ── Navigation Tabs ── */
        .nav-tabs {
            display: flex;
            background: #f1f5f9;
            padding: 0.35rem;
            border-radius: 2rem;
            border: 1px solid var(--card-border);
            gap: 0.35rem;
        }

        .nav-tab {
            background: transparent;
            border: none;
            border-radius: 1.75rem;
            padding: 0.6rem 1.3rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .nav-tab:hover {
            color: var(--primary-dark);
        }

        .nav-tab.active {
            background: #ffffff;
            color: var(--primary-dark);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--mint-border);
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .theme-toggle-btn {
            background: var(--card-bg);
            border: 1.5px solid var(--card-border);
            border-radius: 0.75rem;
            height: 48px;
            padding: 0 1.15rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-heading);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-sm);
            box-sizing: border-box;
        }

        .theme-toggle-btn:hover {
            transform: translateY(-2px);
            border-color: var(--primary-emerald);
            box-shadow: var(--shadow-md);
        }

        [data-theme="dark"] .theme-toggle-btn {
            background: #1f2937;
            border-color: #374151;
            color: #fbbf24;
        }

        .helpline-badge {
            background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
            border: 1.5px solid var(--mint-border);
            border-radius: 0.75rem;
            height: 48px;
            padding: 0 1.15rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            box-shadow: var(--shadow-sm);
            box-sizing: border-box;
        }

        .helpline-badge .phone-icon {
            font-size: 1.3rem;
            display: flex;
            align-items: center;
        }

        .helpline-badge .number {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--primary-dark);
            line-height: 1.1;
        }

        .helpline-badge .label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            line-height: 1.1;
        }

        /* ── Grid Layout for Chat & Weather ── */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 410px;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 980px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        /* ── AI Agent Chat Card ── */
        .chat-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.5rem;
            box-shadow: var(--shadow-lg);
            display: flex;
            flex-direction: column;
            height: 650px;
        }

        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 1rem;
        }

        .chat-header h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .chat-header h2 span {
            color: var(--primary-emerald);
        }

        .chat-status-pill {
            background: var(--mint-light);
            color: var(--primary-dark);
            border: 1px solid var(--mint-border);
            border-radius: 2rem;
            padding: 0.3rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .chat-status-pill::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--primary-emerald);
            box-shadow: 0 0 6px var(--primary-emerald);
        }

        .btn-reset {
            background: #f1f5f9;
            border: 1px solid var(--card-border);
            border-radius: 0.6rem;
            color: var(--text-body);
            padding: 0.35rem 0.85rem;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-reset:hover {
            border-color: var(--primary-emerald);
            color: var(--primary-emerald);
            background: var(--mint-light);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.25rem 0.85rem 0.85rem 0.65rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            margin-bottom: 1rem;
        }

        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 10px;
        }

        .msg {
            max-width: 88%;
            padding: 1.1rem 1.35rem;
            border-radius: 0.75rem;
            font-size: 0.94rem;
            line-height: 1.6;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary-emerald), var(--accent-teal));
            color: #ffffff;
            border-bottom-right-radius: 0.2rem;
            font-weight: 500;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.2);
        }

        .msg-ai {
            align-self: flex-start;
            background: #ffffff;
            border: 1px solid var(--card-border);
            color: #1e293b;
            border-bottom-left-radius: 0.2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        }

        .msg-ai p { margin-bottom: 0.65rem; }
        .msg-ai p:last-child { margin-bottom: 0; }
        .msg-ai ul, .msg-ai ol { margin-left: 1.25rem; margin-bottom: 0.65rem; }
        .msg-ai h3, .msg-ai h4 { font-family: 'Outfit', sans-serif; color: var(--primary-dark); margin: 0.75rem 0 0.35rem 0; font-weight: 700; }
        .msg-ai blockquote { background: var(--mint-light); border-left: 4px solid var(--primary-emerald); padding: 0.6rem 0.85rem; border-radius: 0.4rem; margin: 0.5rem 0; color: var(--primary-dark); font-size: 0.88rem; }

        .chat-prompts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem 1.15rem;
            margin-top: 0.85rem;
            margin-bottom: 1.75rem;
        }

        .prompt-btn {
            background: var(--mint-light);
            border: 1px solid var(--mint-border);
            border-radius: 2rem;
            padding: 0.5rem 1.1rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--primary-dark);
            cursor: pointer;
            transition: all 0.2s;
        }

        .prompt-btn:hover {
            background: #dcfce7;
            border-color: var(--primary-emerald);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.15);
        }

        .chat-input-row {
            display: flex;
            gap: 0.75rem;
        }

        .chat-input-row input {
            flex: 1;
            background: #ffffff;
            border: 1.5px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 0.85rem 1.2rem;
            color: var(--text-heading);
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            transition: all 0.2s;
        }

        .chat-input-row input:focus {
            border-color: var(--primary-emerald);
            box-shadow: 0 0 0 3.5px rgba(5, 150, 105, 0.15);
        }

        .btn-send {
            background: linear-gradient(135deg, var(--primary-emerald), var(--accent-teal));
            color: #ffffff;
            border: none;
            border-radius: 0.75rem;
            height: 48px;
            padding: 0 1.6rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-send:hover {
            opacity: 0.94;
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(5, 150, 105, 0.35);
        }

        /* ── Sidebar Cards & Weather Telemetry ── */
        .side-col {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .weather-card {
            background: #ffffff;
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 1.25rem;
            padding: 1.5rem;
            box-shadow: 0 16px 36px -8px rgba(15, 23, 42, 0.08), 0 4px 12px rgba(15, 23, 42, 0.03);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .weather-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #059669, #0d9488, #0284c7);
        }

        .weather-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.15rem;
            padding-bottom: 0.95rem;
            border-bottom: 1px solid #f1f5f9;
        }

        .weather-title-wrap {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .weather-title-wrap h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0;
            letter-spacing: -0.01em;
        }

        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.68rem;
            font-weight: 800;
            color: #059669;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .live-badge .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulseDot 1.8s infinite;
        }

        @keyframes pulseDot {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }

        .weather-condition-sub {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .district-select-wrapper {
            position: relative;
            display: inline-block;
        }

        .district-select {
            appearance: none;
            -webkit-appearance: none;
            background: #ffffff;
            border: 1.5px solid #cbd5e1;
            border-radius: 0.75rem;
            padding: 0.48rem 2.2rem 0.48rem 0.95rem;
            color: #0f172a;
            font-size: 0.88rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            outline: none;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        }

        .district-select-wrapper::after {
            content: '▾';
            position: absolute;
            right: 0.85rem;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            font-size: 0.85rem;
            color: #64748b;
            font-weight: bold;
        }

        .district-select:hover, .district-select:focus {
            border-color: var(--primary-emerald);
            background: #ffffff;
            box-shadow: 0 0 0 3.5px rgba(5, 150, 105, 0.12);
        }

        .weather-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.15rem;
        }

        .stat-box {
            border-radius: 0.75rem;
            padding: 0.75rem 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        }

        .stat-box:hover {
            transform: translateY(-3px);
        }

        /* Temp Box Styling */
        .temp-box {
            background: linear-gradient(135deg, #fffcf5 0%, #fff7ed 100%);
            border: 1px solid #fed7aa;
        }
        .temp-box:hover {
            box-shadow: 0 8px 20px rgba(217, 119, 6, 0.14);
            border-color: #f97316;
        }
        .temp-box .stat-icon {
            background: linear-gradient(135deg, #ffedd5, #fde68a);
            border: 1px solid #fcd34d;
            color: #c2410c;
        }
        .temp-box .val { color: #ea580c; }

        /* Humidity Box Styling */
        .humidity-box {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border: 1px solid #bae6fd;
        }
        .humidity-box:hover {
            box-shadow: 0 8px 20px rgba(2, 132, 199, 0.14);
            border-color: #38bdf8;
        }
        .humidity-box .stat-icon {
            background: linear-gradient(135deg, #e0f2fe, #bae6fd);
            border: 1px solid #7dd3fc;
            color: #0284c7;
        }
        .humidity-box .val { color: #0284c7; }

        /* Rain Box Styling */
        .rain-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #bbf7d0;
        }
        .rain-box:hover {
            box-shadow: 0 8px 20px rgba(5, 150, 105, 0.14);
            border-color: #34d399;
        }
        .rain-box .stat-icon {
            background: linear-gradient(135deg, #dcfce7, #a7f3d0);
            border: 1px solid #6ee7b7;
            color: #059669;
        }
        .rain-box .val { color: #059669; }

        /* Wind Box Styling */
        .wind-box {
            background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
            border: 1px solid #ddd6fe;
        }
        .wind-box:hover {
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.14);
            border-color: #a78bfa;
        }
        .wind-box .stat-icon {
            background: linear-gradient(135deg, #ede9fe, #ddd6fe);
            border: 1px solid #c4b5fd;
            color: #6366f1;
        }
        .wind-box .val { color: #4f46e5; }

        .stat-icon {
            font-size: 1.1rem;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 0.65rem;
            flex-shrink: 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
        }

        .stat-info {
            flex: 1;
            min-width: 0;
            text-align: left;
        }

        .stat-box .val {
            font-family: 'Outfit', sans-serif;
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            letter-spacing: -0.02em;
        }

        .wind-box .val {
            font-size: 0.88rem;
            font-weight: 700;
        }

        .stat-box .lbl {
            font-size: 0.64rem;
            color: #334155;
            margin-top: 0.2rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.02em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        [data-theme="dark"] .stat-box .lbl {
            color: #cbd5e1;
        }

        /* Risk Alert Box */
        .risk-alert-box {
            border-radius: 0.75rem;
            padding: 1.15rem 1.25rem 1.15rem 1.55rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
            text-align: left;
            position: relative;
            overflow: hidden;
        }

        .risk-alert-box::before {
            content: '';
            position: absolute;
            top: -1.5px;
            left: -1.5px;
            bottom: -1.5px;
            width: 6px;
            background: #059669;
            border-top-left-radius: 0.75rem;
            border-bottom-left-radius: 0.75rem;
        }

        .risk-alert-box.risk-high {
            background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
            border: 1.5px solid #fecaca;
        }
        .risk-alert-box.risk-high::before {
            background: #dc2626;
        }

        .risk-alert-box.risk-moderate {
            background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
            border: 1.5px solid #fde68a;
        }
        .risk-alert-box.risk-moderate::before {
            background: #d97706;
        }

        .risk-alert-box.risk-low {
            background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
            border: 1.5px solid #a7f3d0;
        }
        .risk-alert-box.risk-low::before {
            background: #059669;
        }

        .risk-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.55rem;
        }

        .risk-icon {
            font-size: 1.85rem;
            line-height: 1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin: 0;
            align-self: center;
            filter: drop-shadow(0 2px 6px rgba(217, 119, 6, 0.3));
        }

        .risk-header strong {
            font-family: 'Outfit', sans-serif;
            font-size: 1.22rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            line-height: 1.2;
            margin: 0;
            display: inline-flex;
            align-items: center;
            align-self: center;
        }

        .risk-advice-text {
            font-size: 0.86rem;
            line-height: 1.5;
            color: #334155;
            margin: 0;
            font-weight: 500;
        }

        /* ── Page View 2: Dedicated Disease Database ── */
        .db-page-header {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-md);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1.5rem;
        }

        .db-page-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text-heading);
            margin-bottom: 0.35rem;
        }

        .db-page-subtitle {
            color: var(--text-muted);
            font-size: 0.92rem;
            font-weight: 500;
        }

        .db-search-input {
            width: 360px;
            background: #f8fafc;
            border: 1.5px solid var(--card-border);
            border-radius: 0.85rem;
            padding: 0.75rem 1.2rem;
            font-size: 0.92rem;
            font-family: inherit;
            outline: none;
            transition: all 0.2s;
        }

        .db-search-input:focus {
            border-color: var(--primary-emerald);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.15);
        }

        .controls-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.6rem 1.75rem;
            box-shadow: var(--shadow-md);
            margin-bottom: 1.5rem;
        }

        .crop-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .chip {
            background: #f1f5f9;
            border: 1px solid var(--card-border);
            border-radius: 2rem;
            padding: 0.5rem 1.1rem;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-body);
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
        }

        .chip:hover {
            border-color: var(--primary-emerald);
            color: var(--primary-emerald);
            background: var(--mint-light);
        }

        .chip.active {
            background: linear-gradient(135deg, var(--primary-emerald), var(--accent-teal));
            color: #ffffff;
            font-weight: 700;
            border-color: transparent;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.2);
        }

        /* ── Diagnostics Cards Grid ── */
        .diseases-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 1.5rem;
        }

        .disease-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 1.5rem;
            box-shadow: var(--shadow-md);
            display: flex;
            flex-direction: column;
            height: 100%;
            transition: all 0.2s;
        }

        .disease-card-body {
            display: flex;
            flex-direction: column;
            height: 100%;
            flex: 1;
        }

        .disease-card .sec-prevention {
            margin-top: auto;
            margin-bottom: 0;
        }

        .disease-card:hover {
            box-shadow: var(--shadow-lg);
            border-color: var(--card-border-emerald);
            transform: translateY(-2px);
        }

        .disease-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.85rem;
            gap: 0.75rem;
        }

        .disease-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.18rem;
            font-weight: 800;
            color: var(--text-heading);
            line-height: 1.35;
        }

        .disease-subtitle {
            font-weight: 400;
            color: #6b7280;
            font-size: 0.93em;
        }

        .crop-badge {
            background: var(--mint-light);
            color: var(--primary-dark);
            border: 1px solid var(--mint-border);
            border-radius: 0.4rem;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .category-tag {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.7rem;
            border-radius: 1rem;
            margin-bottom: 1rem;
        }

        .cat-Fungal { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .cat-Pest { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .cat-Viral { background: #faf5ff; color: #9333ea; border: 1px solid #e9d5ff; }

        .symptoms-list { margin-bottom: 1.1rem; }
        .symptoms-list h4 { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 700; letter-spacing: 0.05em; }
        .symptoms-ul {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        .symptoms-ul li {
            position: relative;
            padding-left: 1.1rem;
            font-size: 0.85rem;
            color: var(--text-heading);
            line-height: 1.45;
            font-weight: 500;
        }
        .symptoms-ul li::before {
            content: '';
            position: absolute;
            left: 0.2rem;
            top: 0.55em;
            width: 6px;
            height: 6px;
            background-color: var(--primary-emerald);
            border-radius: 50%;
        }

        .section-box { border-radius: 0.85rem; padding: 0.9rem 1rem; margin-bottom: 0.85rem; font-size: 0.88rem; line-height: 1.5; }
        .section-box h5 { font-size: 0.78rem; text-transform: uppercase; margin-bottom: 0.45rem; font-weight: 800; letter-spacing: 0.04em; line-height: 1.3; }
        .section-box > div, .section-box > p { margin: 0; }

        .sec-organic { background: #f7fee7; border: 1px solid #d9f99d; }
        .sec-organic h5 { color: #4d7c0f; }

        .sec-chemical { background: #f0fdfa; border: 1px solid #99f6e4; }
        .sec-chemical h5 { color: #0f766e; }

        .sec-prevention { background: #fff8f1; border: 1px solid #ffedd5; }
        .sec-prevention h5 { color: #c2410c; }

        .empty-state { grid-column: 1 / -1; padding: 3rem; text-align: center; color: var(--text-muted); background: var(--card-bg); border: 1px dashed var(--card-border); border-radius: 1.25rem; font-weight: 500; }

        footer { margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid var(--card-border); text-align: center; font-size: 0.88rem; color: var(--text-muted); font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Bar with Navigation Tabs -->
        <header>
            <div class="brand">
                <div class="brand-icon">🌾</div>
                <div>
                    <h1>Sri Lanka <span>CropAdvisor</span></h1>
                    <p>Department of Agriculture Standard Advisory Portal</p>
                </div>
            </div>

            <nav class="nav-tabs">
                <button class="nav-tab active" id="tab-btn-chat" onclick="switchPage('chat')">
                    🤖 AI Advisor & Weather
                </button>
                <button class="nav-tab" id="tab-btn-database" onclick="switchPage('database')">
                    📚 Disease Database
                </button>
            </nav>

            <div class="header-actions">
                <button class="theme-toggle-btn" id="theme-toggle-btn" onclick="toggleTheme()" title="Toggle Dark/Light Theme">
                    <span id="theme-toggle-icon">🌙</span>
                    <span id="theme-toggle-text">Dark Mode</span>
                </button>
                <div class="helpline-badge">
                    <div class="phone-icon">📞</div>
                    <div>
                        <div class="number">+94 11 286 1500</div>
                        <div class="label">DOA Krushi Upades Hotline</div>
                    </div>
                </div>
            </div>
        </header>

        <!-- PAGE 1: AI Chat Assistant & Live District Weather -->
        <div id="page-chat" class="main-grid">
            <!-- Left AI Chat Section -->
            <div class="chat-card">
                <div class="chat-header">
                    <h2>🤖 Chat with <span>CropAdvisor AI</span></h2>
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <button class="btn-reset" onclick="resetChat()" title="Start a fresh conversation">🔄 New Chat</button>
                        <div class="chat-status-pill">Agent Online</div>
                    </div>
                </div>

                <div class="chat-messages" id="chat-messages">
                    <div class="msg msg-ai">
                        <p>👋 <strong>Ayubowan! I am your CropAdvisor virtual assistant.</strong></p>
                        <p>Ask me any question about crop symptoms, pest control, chemical/organic treatments, or weather risks in Sri Lanka!</p>
                    </div>
                </div>

                <div class="chat-prompts">
                    <button class="prompt-btn" onclick="sendPrompt('My paddy leaves have brown spindle spots in Anuradhapura')">🌾 Paddy Blast in Anuradhapura</button>
                    <button class="prompt-btn" onclick="sendPrompt('Chilli leaves curling upwards in Jaffna')">🌶️ Chilli Leaf Curl in Jaffna</button>
                    <button class="prompt-btn" onclick="sendPrompt('Potato late blight symptoms and treatment in Nuwara Eliya')">🥔 Potato Blight in Nuwara Eliya</button>
                    <button class="prompt-btn" onclick="sendPrompt('What is the official Department of Agriculture helpline number?')">📞 DOA Hotline Info</button>
                </div>

                <div class="chat-input-row">
                    <input type="text" id="chat-input" placeholder="Type your agricultural question here..." onkeypress="handleKeyPress(event)">
                    <button class="btn-send" onclick="sendMessage()" id="send-btn">Send</button>
                </div>
            </div>

            <!-- Right Column: Live District Weather Risk -->
            <div class="side-col">
                <div class="weather-card">
                    <div class="weather-header">
                        <div class="weather-title-wrap">
                            <h2>
                                ⛅ Live Weather
                                <span class="live-badge"><span class="pulse-dot"></span>LIVE</span>
                            </h2>
                            <div class="weather-condition-sub" id="weather-condition">Loading telemetry...</div>
                        </div>
                        <div class="district-select-wrapper">
                            <select id="district-select" class="district-select" onchange="fetchWeather()">
                                <option value="Ampara">Ampara</option>
                                <option value="Anuradhapura" selected>Anuradhapura</option>
                                <option value="Badulla">Badulla</option>
                                <option value="Batticaloa">Batticaloa</option>
                                <option value="Colombo">Colombo</option>
                                <option value="Galle">Galle</option>
                                <option value="Gampaha">Gampaha</option>
                                <option value="Hambantota">Hambantota</option>
                                <option value="Jaffna">Jaffna</option>
                                <option value="Kalutara">Kalutara</option>
                                <option value="Kandy">Kandy</option>
                                <option value="Kegalle">Kegalle</option>
                                <option value="Kilinochchi">Kilinochchi</option>
                                <option value="Kurunegala">Kurunegala</option>
                                <option value="Mannar">Mannar</option>
                                <option value="Matale">Matale</option>
                                <option value="Matara">Matara</option>
                                <option value="Monaragala">Monaragala</option>
                                <option value="Mullaitivu">Mullaitivu</option>
                                <option value="Nuwara Eliya">Nuwara Eliya</option>
                                <option value="Polonnaruwa">Polonnaruwa</option>
                                <option value="Puttalam">Puttalam</option>
                                <option value="Ratnapura">Ratnapura</option>
                                <option value="Trincomalee">Trincomalee</option>
                                <option value="Vavuniya">Vavuniya</option>
                            </select>
                        </div>
                    </div>

                    <div class="weather-stats">
                        <div class="stat-box temp-box">
                            <div class="stat-icon">🌡️</div>
                            <div class="stat-info">
                                <div class="val" id="weather-temp">--°C</div>
                                <div class="lbl">Temperature</div>
                            </div>
                        </div>
                        <div class="stat-box humidity-box">
                            <div class="stat-icon">💧</div>
                            <div class="stat-info">
                                <div class="val" id="weather-humidity">--%</div>
                                <div class="lbl">Humidity</div>
                            </div>
                        </div>
                        <div class="stat-box rain-box">
                            <div class="stat-icon">🌧️</div>
                            <div class="stat-info">
                                <div class="val" id="weather-rain">-- mm</div>
                                <div class="lbl">Rain (24h)</div>
                            </div>
                        </div>
                        <div class="stat-box wind-box">
                            <div class="stat-icon">💨</div>
                            <div class="stat-info">
                                <div class="val" id="weather-wind">--</div>
                                <div class="lbl">Wind</div>
                            </div>
                        </div>
                    </div>

                    <div class="risk-alert-box risk-low" id="risk-box">
                        <div class="risk-header">
                            <span class="risk-icon" id="risk-icon">🛡️</span>
                            <strong id="risk-level" style="color: var(--primary-emerald);">Evaluating Risk...</strong>
                        </div>
                        <p id="risk-advice" class="risk-advice-text">Loading district weather telemetry...</p>
                    </div>
                </div>

                <div class="controls-card" style="background: linear-gradient(135deg, #ecfdf5, #ffffff); border-color: var(--mint-border);">
                    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--primary-dark); margin-bottom: 0.5rem;">
                        📚 Need Database Records?
                    </h3>
                    <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.85rem;">
                        Browse the complete Department of Agriculture disease database on a dedicated page.
                    </p>
                    <button class="btn-send" style="width: 100%; justify-content: center;" onclick="switchPage('database')">
                        Open Disease Database →
                    </button>
                </div>
            </div>
        </div>

        <!-- PAGE 2: Dedicated Department of Agriculture Disease Database Page -->
        <div id="page-database" style="display: none;">
            <div class="db-page-header">
                <div>
                    <h2 class="db-page-title">📚 Department of Agriculture Disease Database</h2>
                    <p class="db-page-subtitle">Search and explore verified Sri Lankan crop pathology records, organic remedies, and chemical standards.</p>
                </div>
                <div>
                    <input type="text" id="db-search-input" class="db-search-input" placeholder="🔍 Search disease name, symptoms, or remedies..." oninput="handleSearchInput()">
                </div>
            </div>

            <div class="controls-card">
                <div class="crop-chips" id="crop-chips">
                    <div class="chip active" onclick="selectCrop('all', this)">All Crops</div>
                    <div class="chip" onclick="selectCrop('paddy', this)">🌾 Paddy</div>
                    <div class="chip" onclick="selectCrop('chilli', this)">🌶️ Chilli</div>
                    <div class="chip" onclick="selectCrop('tomato', this)">🍅 Tomato</div>
                    <div class="chip" onclick="selectCrop('cinnamon', this)">🪵 Cinnamon</div>
                    <div class="chip" onclick="selectCrop('tea', this)">🍃 Tea</div>
                    <div class="chip" onclick="selectCrop('maize', this)">🌽 Maize</div>
                    <div class="chip" onclick="selectCrop('potato', this)">🥔 Potato</div>
                    <div class="chip" onclick="selectCrop('pepper', this)">🫑 Black Pepper</div>
                </div>
            </div>

            <div class="diseases-grid" id="diseases-grid">
                <div class="empty-state">Loading crop disease records...</div>
            </div>
        </div>

        <footer>
            Official Sri Lanka Department of Agriculture Advisory Standard • Powered by Google ADK Agent
        </footer>
    </div>

    <script>
        let currentCrop = 'all';
        let searchQuery = '';
        let sessionId = null;

        function switchPage(page) {
            const chatPage = document.getElementById('page-chat');
            const dbPage = document.getElementById('page-database');
            const btnChat = document.getElementById('tab-btn-chat');
            const btnDb = document.getElementById('tab-btn-database');

            if (page === 'database') {
                chatPage.style.display = 'none';
                dbPage.style.display = 'block';
                btnChat.classList.remove('active');
                btnDb.classList.add('active');
                loadDiseases();
            } else {
                dbPage.style.display = 'none';
                chatPage.style.display = 'grid';
                btnDb.classList.remove('active');
                btnChat.classList.add('active');
            }
        }

        function resetChat() {
            sessionId = null;
            const container = document.getElementById('chat-messages');
            container.innerHTML = `
                <div class="msg msg-ai">
                    <p>👋 <strong>Ayubowan! Conversation reset.</strong></p>
                    <p>Ask me any question about crop symptoms, pest control, chemical/organic treatments, or weather risks in Sri Lanka!</p>
                </div>
            `;
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            appendMessage(message, 'user');

            const sendBtn = document.getElementById('send-btn');
            sendBtn.disabled = true;
            sendBtn.innerText = 'Thinking...';

            const loadingMsgId = appendLoadingMessage();

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, session_id: sessionId })
                });

                const data = await res.json();
                removeLoadingMessage(loadingMsgId);

                if (data.status === 'success') {
                    sessionId = data.session_id;
                    appendMessage(data.response, 'ai');
                } else {
                    if (data.message && data.message.includes('Session not found')) {
                        sessionId = null;
                    }
                    appendMessage(`⚠️ Error: ${data.message || 'Failed to get response.'}`, 'ai');
                }
            } catch (err) {
                removeLoadingMessage(loadingMsgId);
                appendMessage('⚠️ Network error. Please check your connection.', 'ai');
            } finally {
                sendBtn.disabled = false;
                sendBtn.innerText = 'Send';
            }
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

        function sendPrompt(text) {
            document.getElementById('chat-input').value = text;
            sendMessage();
        }

        function appendMessage(text, role) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = `msg msg-${role}`;
            
            if (role === 'ai') {
                div.innerHTML = marked.parse(text);
            } else {
                div.innerText = text;
            }

            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function appendLoadingMessage() {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'msg msg-ai';
            div.id = 'loading-' + Date.now();
            div.innerHTML = '<em>🌱 CropAdvisor AI is searching diagnostic tools & live weather...</em>';
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div.id;
        }

        function removeLoadingMessage(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        function handleSearchInput() {
            searchQuery = document.getElementById('db-search-input').value.trim();
            loadDiseases();
        }

        async function loadDiseases() {
            const params = new URLSearchParams();
            if (currentCrop !== 'all') params.append('crop', currentCrop);
            if (searchQuery) params.append('search', searchQuery);

            try {
                const res = await fetch(`/api/diseases?${params.toString()}`);
                const data = await res.json();
                renderDiseases(data.results);
            } catch (err) {
                console.error("Failed to load diseases", err);
            }
        }

        function selectCrop(crop, el) {
            currentCrop = crop;
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            el.classList.add('active');
            loadDiseases();
        }

        function formatDiseaseTitle(name) {
            if (!name) return '';
            const idx = name.indexOf('(');
            if (idx !== -1 && name.endsWith(')')) {
                const primary = name.substring(0, idx).trim();
                const alt = name.substring(idx);
                return `${escapeHtml(primary)} <span class="disease-subtitle">${escapeHtml(alt)}</span>`;
            }
            return escapeHtml(name);
        }

        function renderDiseases(records) {
            const container = document.getElementById('diseases-grid');
            if (!records || records.length === 0) {
                container.innerHTML = `<div class="empty-state">🌾 No disease records matching criteria.</div>`;
                return;
            }

            container.innerHTML = records.map(r => {
                const catClass = `cat-${r.category.split(' ')[0]}`;
                const symptomsHtml = (r.symptoms || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');
                return `
                    <div class="disease-card">
                        <div class="disease-card-body">
                            <div class="disease-header">
                                <div class="disease-title">${formatDiseaseTitle(r.disease_name)}</div>
                                <span class="crop-badge">${escapeHtml(r.crop)}</span>
                            </div>
                            <span class="category-tag ${catClass}">${escapeHtml(r.category)}</span>
                            
                            <div class="symptoms-list">
                                <h4>Key Symptoms</h4>
                                <ul class="symptoms-ul">
                                    ${symptomsHtml}
                                </ul>
                            </div>

                            <div class="section-box sec-organic">
                                <h5>🌿 Organic & Cultural Remedy</h5>
                                <div>${escapeHtml(r.organic_treatment)}</div>
                            </div>

                            <div class="section-box sec-chemical">
                                <h5>🧪 DOA Chemical Standard</h5>
                                <div>${escapeHtml(r.chemical_treatment)}</div>
                            </div>

                            <div class="section-box sec-prevention">
                                <h5>🛡️ Prevention Strategy</h5>
                                <div>${escapeHtml(r.prevention)}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function fetchWeather() {
            const district = document.getElementById('district-select').value;
            const riskBox = document.getElementById('risk-box');
            const riskLevel = document.getElementById('risk-level');
            const riskAdvice = document.getElementById('risk-advice');
            const riskIcon = document.getElementById('risk-icon');
            const weatherCondition = document.getElementById('weather-condition');

            try {
                const res = await fetch(`/api/weather?location=${encodeURIComponent(district)}`);
                const data = await res.json();

                if (data.status === 'success') {
                    document.getElementById('weather-temp').innerText = `${data.temperature_celsius ?? '--'}°C`;
                    document.getElementById('weather-humidity').innerText = `${data.relative_humidity_percent ?? '--'}%`;
                    
                    const dailyPrecip = data.daily_precipitation_sum_mm ?? data.precipitation_mm ?? 0;
                    document.getElementById('weather-rain').innerText = `${dailyPrecip} mm`;

                    const windSpd = data.wind_speed_kmh != null ? Math.round(data.wind_speed_kmh) : '--';
                    const windCard = data.wind_direction_cardinal || '';
                    const windText = `${windCard} ${windSpd} km/h`.trim();
                    const windElem = document.getElementById('weather-wind');
                    windElem.innerText = windText;
                    windElem.title = windText;

                    if (weatherCondition) {
                        weatherCondition.innerText = data.condition || 'Live Telemetry';
                    }

                    if (data.disease_risk) {
                        riskLevel.innerText = data.disease_risk.level;
                        riskAdvice.innerText = data.disease_risk.advice;

                        // Reset dynamic risk state classes
                        riskBox.classList.remove('risk-high', 'risk-moderate', 'risk-low');

                        const lvlLower = data.disease_risk.level.toLowerCase();
                        if (lvlLower.includes('high') || lvlLower.includes('critical') || lvlLower.includes('flood')) {
                            riskBox.classList.add('risk-high');
                            riskLevel.style.color = '#dc2626';
                            if (riskIcon) riskIcon.innerText = '🚨';
                        } else if (lvlLower.includes('moderate') || lvlLower.includes('warn')) {
                            riskBox.classList.add('risk-moderate');
                            riskLevel.style.color = '#d97706';
                            if (riskIcon) riskIcon.innerText = '⚠️';
                        } else {
                            riskBox.classList.add('risk-low');
                            riskLevel.style.color = '#059669';
                            if (riskIcon) riskIcon.innerText = '🛡️';
                        }
                    }
                } else {
                    riskBox.classList.remove('risk-high', 'risk-moderate', 'risk-low');
                    riskBox.classList.add('risk-high');
                    riskLevel.innerText = "Weather Sensor Offline";
                    riskLevel.style.color = "#dc2626";
                    if (riskIcon) riskIcon.innerText = '📡';
                    riskAdvice.innerText = data.message || "Failed to fetch live weather.";
                }
            } catch (err) {
                console.error("Failed to load weather", err);
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
        }

        function setTheme(theme) {
            const iconEl = document.getElementById('theme-toggle-icon');
            const textEl = document.getElementById('theme-toggle-text');
            if (theme === 'dark') {
                document.documentElement.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                if (iconEl) iconEl.innerText = '☀️';
                if (textEl) textEl.innerText = 'Light Mode';
            } else {
                document.documentElement.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
                if (iconEl) iconEl.innerText = '🌙';
                if (textEl) textEl.innerText = 'Dark Mode';
            }
        }

        function initTheme() {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                setTheme('dark');
            } else {
                setTheme('light');
            }
        }

        // Initial Load
        initTheme();
        loadDiseases();
        fetchWeather();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def get_crop_advisor_dashboard():
    """Render the main CropAdvisor Web Dashboard."""
    return HTMLResponse(content=CROP_ADVISOR_HTML)


if __name__ == "__main__":
    print("Starting CropAdvisor Web Application on http://localhost:8080 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
