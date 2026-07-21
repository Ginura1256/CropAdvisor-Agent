# Sri Lankan CropAdvisor ADK Agent & Web Application

[![Google ADK](https://img.shields.io/badge/Google%20ADK-Agent%20Development%20Kit-emerald)](https://google.github.io/adk-docs/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

An official AI-powered agricultural advisory portal and REST API for Sri Lankan farmers and extension officers. Built using **Google Agent Development Kit (ADK)**, **FastAPI**, **Open-Meteo Live Weather API**, and grounded in the **Sri Lanka Department of Agriculture (DOA)** disease database.

---

## 🌟 Key Features

- 🤖 **ADK AI Agent Architecture**: Powered by `google-adk` with function tools (`search_crop_diseases`, `get_live_weather`, `get_agricultural_helpline`).
- 🎨 **Modern Interactive Dashboard**: Dark glassmorphic dashboard with live weather telemetry, disease filters, and AI chat.
- 🌾 **Grounded Crop Disease Database**: Grounded diagnoses for Paddy, Chilli, Tomato, Cinnamon, Tea, Maize, Potato, and Black Pepper.
- ⛅ **Real-Time Weather Risk Telemetry**: Fetches live district weather (temperature, relative humidity, rainfall) to assess fungal & disease threat levels.
- 🔄 **Resilient Multi-Turn Sessions**: Robust session recovery with auto-creation (`auto_create_session=True`) and interactive session reset button (`New Chat`).
- 🧪 **Comprehensive Automated Testing**: Unit and integration test suites covering agent tools, session handling, and API endpoints.

---

## 📁 Project Structure

```text
.
├── crop_agent/
│   ├── __init__.py           # Package exports for root_agent & tools
│   ├── agent.py              # Root ADK agent definition & function tools
│   └── crop_diseases.json    # DOA grounded disease & pest record database
├── server.py                 # FastAPI REST API & Web Dashboard server
├── test_crop_agent.py        # Automated test suite for crop agent & tools
├── test_server.py           # Automated test suite for FastAPI REST endpoints & chat
├── .env.example              # Template for API keys
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Google AI Studio API Key** (or Google Cloud Vertex AI credentials)
- **Git**

### 2. Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Ginura1256/CropAdvisor-Agent.git
cd CropAdvisor-Agent
```

Create and activate a virtual environment:

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
# macOS/Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` and add your Google API Key:

```env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

---

## 💻 Running the Application

### Option A: Launch Web Dashboard & REST API (Recommended)

Run the server with Python:

```bash
python server.py
```

Open your browser and navigate to:
👉 **`http://localhost:8080`**

### Option B: Launch ADK Developer Chat Interface

To test the agent inside the Google ADK Developer Web UI:

```bash
adk web
```

Select `crop_advisor` from the agent list.

---

## 💬 Example Queries for CropAdvisor

- **Paddy Diagnostics**: *"My paddy leaves have brown spindle-shaped spots in Anuradhapura."*
- **Chilli Leaf Curl**: *"Chilli leaves curling upwards with stunted growth in Jaffna."*
- **Potato Blight**: *"Potato leaves have dark water-soaked spots in Nuwara Eliya. What treatment should I use?"*
- **General Advisory**: *"What is the official Department of Agriculture helpline number?"*

---

## 🧪 Running Automated Tests

Run the crop agent tool tests:

```bash
python test_crop_agent.py
```

Run the server and API test suite:

```bash
python test_server.py
```

---

## 🛡️ License & Standards

Grounding guidelines adhere to the official standards of the **Department of Agriculture Sri Lanka (DOA)**. Hotline: **1920** (Krushi Upades).
