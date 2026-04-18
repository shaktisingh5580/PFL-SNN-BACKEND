<![CDATA[<div align="center">

# 🛰️ GeoGuard AI — Backend Engine

### The World's First SNN-Based Agentic Geospatial Compliance System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-purple)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A fully autonomous AI pipeline that replaces traditional deep learning with bio-inspired Siamese Spiking Neural Networks (SNNs) to analyze satellite imagery, detect unauthorized construction, and generate tamper-proof compliance evidence — all orchestrated by an intelligent LangGraph agent.**

[Getting Started](#-getting-started) · [Architecture](#-system-architecture) · [API Reference](#-api-reference) · [Frontend](#-frontend-setup) · [How It Works](#-how-it-works)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone & Install](#1-clone--install)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Start the Backend Server](#3-start-the-backend-server)
- [Frontend Setup](#-frontend-setup)
- [How It Works](#-how-it-works)
  - [The 9-Step Pipeline](#the-9-step-pipeline)
  - [Siamese-SNN Model Architecture](#siamese-snn-model-architecture)
  - [Spectral Index Classification](#spectral-index-classification)
  - [Compliance Rule Engine](#compliance-rule-engine)
  - [Agentic Chatbot (LangGraph)](#agentic-chatbot-langgraph)
  - [Tamper-Proof Evidence Chain](#tamper-proof-evidence-chain)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Training the Model](#-training-the-model)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)

---

## 🌍 Overview

GeoGuard AI is an **end-to-end geospatial compliance engine** that continuously monitors satellite imagery to detect unauthorized land-use changes — construction in protected zones, deforestation, water body encroachment, and more.

What makes this system unique is its use of **Spiking Neural Networks (SNNs)** — the third generation of neural networks that process information using temporal spike dynamics rather than static activations. This bio-inspired approach provides:

- 🧠 **Natural noise resilience** — atmospheric haze and seasonal variation are filtered automatically by the temporal dynamics
- ⚡ **Energy-efficient inference** — spike-based computation is fundamentally event-driven
- 🎯 **Better temporal reasoning** — the network processes change over time-steps, not as a single static comparison

> **This is the world's first fully operational system that uses Spiking Neural Networks for real-world satellite imagery change detection and autonomous compliance monitoring.**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **🧬 Siamese-SNN Model** | Hybrid Siamese U-Net encoder + snntorch Leaky Integrate-and-Fire decoder with Poisson rate coding |
| **🛰️ Live Satellite Data** | Real-time Sentinel-2 L2A imagery via Copernicus CDSE (10m resolution, 5 spectral bands + cloud mask) |
| **📊 Spectral Classification** | NDVI / NDBI / MNDWI index analysis classifies changes into Construction, Vegetation Loss, Water Change, and Other |
| **⚖️ Compliance Rule Engine** | JSON-configurable rules with spatial analysis (buffer zones, zone exclusions, percentage thresholds) |
| **🤖 Agentic AI Chatbot** | LangGraph-powered agent with 8 tools — queries real scan data, cites legal sections, dispatches WhatsApp alerts |
| **🔗 Tamper-Proof Evidence** | SHA-256 evidence hashing + Merkle tree for cryptographic integrity of scan reports |
| **📄 Auto-Generated PDFs** | Professional compliance reports with satellite imagery, classifications, violations, and legal citations |
| **🗺️ Supabase Integration** | PostGIS spatial database + object storage for scan results, images, and conversation history |
| **📡 SSE Live Streaming** | Real-time Server-Sent Events stream the 9-step pipeline progress to the frontend dashboard |
| **📱 WhatsApp Alerts** | WAHA-powered automated dispatches for critical violations (Risk Score > 80) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND DASHBOARD                           │
│   (Next.js / React — Satellite Map + Analytics + AI Chat)           │
│         Connects via HTTP/SSE/WebSocket to Backend API              │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   FastAPI Server   │
                    │   (Port 8000)      │
                    │   ├─ /api/scan/stream  (SSE Pipeline)
                    │   ├─ /api/chat         (Agentic AI)
                    │   ├─ /api/scans        (Database)
                    │   └─ /ws/chat          (WebSocket)
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼─────┐ ┌──────▼────────┐
    │  Scan Pipeline  │ │ Chatbot  │ │   Supabase    │
    │  (Orchestrator) │ │ (Agent)  │ │  (PostGIS +   │
    │                 │ │          │ │   Storage)    │
    │ 1. CDSE Auth    │ │ LangGraph│ │               │
    │ 2. Download     │ │ + 8 Tools│ │  Scans Table  │
    │ 3. Normalize    │ │ + RAG    │ │  Messages     │
    │ 4. SNN Infer    │ │          │ │  Images       │
    │ 5. Classify     │ └──────────┘ └───────────────┘
    │ 6. Compliance   │
    │ 7. PDF + Save   │
    │ 8. Upload       │
    │ 9. Visualize    │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Siamese-SNN    │
    │  ┌────────────┐ │
    │  │ Shared     │ │     ┌──────────────────────┐
    │  │ Encoder    │ │     │  Spectral Classifier  │
    │  │ (U-Net)    │ │     │  NDVI / NDBI / MNDWI  │
    │  └─────┬──────┘ │     └──────────────────────┘
    │  ┌─────▼──────┐ │
    │  │ SNN Decoder│ │     ┌──────────────────────┐
    │  │ (snntorch  │ │     │  Compliance Engine    │
    │  │  LIF)      │ │     │  5 Rules (JSON)       │
    │  └────────────┘ │     │  Shapely Spatial Ops  │
    └─────────────────┘     └──────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (3.10 also works)
- **Git**
- A free **Copernicus Data Space** account (for satellite imagery)
- A free **Supabase** project (for database + storage)
- *(Optional)* A free **NVIDIA NIM** or **OpenRouter** API key (for the AI chatbot)

### 1. Clone & Install

```bash
# Clone the backend repository
git clone https://github.com/shaktisingh5580/PFL-SNN-BACKEND.git
cd PFL-SNN-BACKEND

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note on PyTorch:** The `requirements.txt` installs the default PyTorch build. If you have a CUDA GPU and want GPU-accelerated inference, install the CUDA version manually first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Now edit `.env` with your actual keys:

```env
# ===== Sentinel Hub (REQUIRED — for satellite imagery) =====
# Get these from: https://dataspace.copernicus.eu/
# Register → Dashboard → OAuth Clients → Create New
SH_CLIENT_ID=your_copernicus_client_id
SH_CLIENT_SECRET=your_copernicus_client_secret

# ===== Supabase (REQUIRED — for database & image storage) =====
# Get these from: https://supabase.com → Your Project → Settings → API
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# ===== LLM — for the AI Chatbot (OPTIONAL — system works without it) =====
# Option A: NVIDIA NIM (recommended, generous free tier)
# Get from: https://build.nvidia.com/
NVIDIA_API_KEY=nvapi-your_nvidia_key

# Option B: OpenRouter (free Llama 3.3 70B)
# Get from: https://openrouter.ai/
OPENROUTER_API_KEY=your_openrouter_key

# ===== WhatsApp via WAHA (OPTIONAL — can be skipped entirely) =====
# Only needed if you want automated WhatsApp alerts
# Requires running the WAHA Docker container locally
# WHATSAPP_TARGET=919876543210
# WAHA_API_KEY=your_waha_key

# ===== App Config =====
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
```

#### What Can You Skip?

| Service | Required? | What happens if skipped |
|---|---|---|
| **Sentinel Hub (CDSE)** | ✅ Required | Scans won't work — this is the satellite data source |
| **Supabase** | ✅ Required | Scans run locally but results aren't persisted or uploaded |
| **NVIDIA / OpenRouter** | ❌ Optional | Chatbot runs in fallback mode (no LLM, only keyword-based responses) |
| **WhatsApp (WAHA)** | ❌ Optional | No WhatsApp alerts — everything else works normally |
| **PostGIS (Docker)** | ❌ Optional | Only needed if using the self-hosted PostGIS instead of Supabase |

### 3. Start the Backend Server

```bash
python start_server.py
```

You should see:

```
============================================================
  Satellite Compliance Engine - API Server
============================================================
  Local:    http://localhost:8000
  Network:  http://192.168.x.x:8000
  API Docs: http://192.168.x.x:8000/docs
  Scan SSE: POST http://192.168.x.x:8000/api/scan/stream
============================================================
```

The server is now running. Open `http://localhost:8000/docs` to see the interactive Swagger API documentation.

> **Network Access:** The server binds to `0.0.0.0` which means other devices on your local network (including the frontend running on another laptop) can connect using the **Network** URL shown above.

---

## 🖥️ Frontend Setup

The frontend is a separate repository — a Next.js dashboard with an interactive satellite map, real-time pipeline monitoring, analytics, and an AI chat interface.

### 1. Clone the Frontend

```bash
git clone https://github.com/shaktisingh5580/PFL-SNN-FRONTEND.git
cd PFL-SNN-FRONTEND
npm install
```

### 2. Configure Frontend Environment

Create a `.env.local` file in the frontend project root:

```env
# Point this to your backend server
# If running on the SAME machine:
NEXT_PUBLIC_API_URL=http://localhost:8000

# If running on a DIFFERENT machine on the same network:
# Find the backend's IP from the "Network" line in start_server.py output
NEXT_PUBLIC_API_URL=http://192.168.x.x:8000
```

### 3. Start the Frontend

```bash
npm run dev
```

The dashboard will open at `http://localhost:3000`.

### Connecting Frontend ↔ Backend on Network

If the frontend and backend are on **different laptops** on the same Wi-Fi/LAN:

1. **Backend laptop:** Run `python start_server.py` — note the `Network: http://192.168.x.x:8000` URL
2. **Frontend laptop:** Set `NEXT_PUBLIC_API_URL=http://192.168.x.x:8000` in `.env.local`
3. **Frontend laptop:** Run `npm run dev`
4. Both machines can now communicate — the frontend sends scan requests and receives SSE events from the backend in real-time

> **CORS:** The backend already allows all origins (`allow_origins=["*"]`), so cross-machine connections work out of the box.

---

## 🔬 How It Works

### The 9-Step Pipeline

When you draw a bounding box on the map and click **"Initialize Scan"**, the backend executes a fully autonomous 9-step pipeline, streaming real-time progress events via SSE:

| Step | Name | What Happens |
|------|------|-------------|
| 1 | **CDSE Authentication** | OAuth2 token exchange with Copernicus Data Space Ecosystem |
| 2 | **Download Imagery** | Fetches "Before" and "After" Sentinel-2 L2A patches (5 bands: B02, B03, B04, B08, B11 + SCL cloud mask) |
| 3 | **Normalize** | Per-channel min-max normalization of spectral bands |
| 4 | **Siamese-SNN Inference** | The core neural network processes 128×128 patches through the Siamese encoder → Rate Coding → SNN Decoder pipeline. Cloud pixels from SCL are masked out |
| 5 | **Spectral Classification** | NDVI/NDBI/MNDWI index deltas classify each changed pixel into: Construction, Vegetation Loss, Water Change, or Other |
| 6 | **Compliance Rules** | The JSON-configurable rule engine evaluates spatial violations using Shapely geometry (buffer zones, zone exclusions, percentage thresholds) |
| 7 | **Report Generation** | Creates PNG visualizations (Before, After, Change Mask, Classification Overlay) + a professional PDF compliance report + JSON report with evidence hash |
| 8 | **Supabase Upload** | Uploads all images to Supabase Storage and saves scan metadata, classification, and violations to the PostGIS database |
| 9 | **Visualization** | Generates a combined 4-panel matplotlib figure for quick visual inspection |

If the risk score exceeds 80, an optional **Step 10** dispatches a WhatsApp alert to the field officer.

### Siamese-SNN Model Architecture

The model (`src/model/siamese_snn.py`) is a hybrid architecture:

```
            Before Image                    After Image
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │               │               │               │
        │   Shared       │               │   Shared       │
        │   U-Net        │◄──── Same ───►│   U-Net        │
        │   Encoder      │   Weights     │   Encoder      │
        │               │               │               │
        │  64→128→256   │               │  64→128→256   │
        │     →512       │               │     →512       │
        └───┬───────────┘               └───┬───────────┘
            │ Features_A + Skips_A          │ Features_B + Skips_B
            │                               │
            └───────────┬───────────────────┘
                        │
                   |F_A − F_B|          ← Absolute Feature Difference
                        │
                   ┌────▼────┐
                   │  Rate   │
                   │ Coding  │          ← sigmoid(diff) → Poisson spikes
                   │ (T=10)  │
                   └────┬────┘
                        │
                   (T, B, C, H, W) spike trains
                        │
                   ┌────▼────┐
                   │  SNN    │
                   │ Decoder │          ← ConvTranspose2d + snn.Leaky LIF neurons
                   │ (4 lvl) │             with skip connection diffs
                   └────┬────┘
                        │
                   Spike Accumulation    ← Mean firing rate over T steps
                        │
                   ┌────▼────┐
                   │ Softmax │          ← Class 0: No Change
                   │  → Map  │            Class 1: Change
                   └─────────┘
```

**Key SNN Components:**
- **Rate Coding:** Feature differences are converted to Poisson spike trains via `snntorch.spikegen.rate()` over `T=10` time-steps
- **LIF Neurons:** `snn.Leaky` (Leaky Integrate-and-Fire) neurons in the decoder with `β=0.9` membrane decay and `fast_sigmoid` surrogate gradient
- **Temporal Accumulation:** The final change map is the mean firing rate over all time-steps — pixels that spike frequently = high confidence change

### Spectral Index Classification

After the SNN produces a binary change mask, the spectral classifier (`src/compliance/classifier.py`) categorizes **what type** of change occurred:

| Index | Formula | High Value Means |
|-------|---------|-----------------|
| **NDVI** | (NIR − Red) / (NIR + Red) | Dense vegetation |
| **NDBI** | (SWIR − NIR) / (SWIR + NIR) | Built-up / concrete surfaces |
| **MNDWI** | (Green − SWIR) / (Green + SWIR) | Water bodies |

**Classification Rules:**
- NDVI drops + NDBI rises → 🏗️ **Construction / Urban Sprawl**
- NDVI drops alone → 🌳 **Vegetation Clearance / Deforestation**
- MNDWI shifts → 🌊 **Water Body Change / Sand Mining**
- Any remaining → 🟡 **Other Land Alteration**

### Compliance Rule Engine

The rule engine (`src/compliance/rule_engine.py`) evaluates detected changes against configurable regulations loaded from `src/compliance/rules.json`:

| Rule ID | Name | Type | Severity | Legal Reference |
|---------|------|------|----------|----------------|
| R001 | Water Body Buffer Zone | buffer_exclusion (500m) | CRITICAL | Gujarat GDCR 2017 § 12.3 |
| R002 | Green Belt Protection | zone_exclusion | HIGH | Gujarat GDCR 2017 § 15.1 |
| R003 | Minimum Vegetation Cover | percentage_threshold (30%) | MEDIUM | SUDA Dev Plan 2035 § 8.2 |
| R004 | Tapi Riverfront Buffer | buffer_exclusion (500m) | CRITICAL | Surat GDCR Special Reg. |
| R005 | Industrial Zone Restriction | zone_exclusion | HIGH | Gujarat TP & UD Act 1976 § 22 |

Rules are **JSON-configurable** — add new rules or adapt to any city without changing code. When scanning locations outside Gujarat, legal references are automatically localized to the scanned city's municipal code.

### Agentic Chatbot (LangGraph)

The AI chatbot (`src/chatbot/agent.py`) is a full LangGraph state machine with 8 integrated tools:

| Tool | Function |
|------|----------|
| `get_all_scans` | Queries all scan history from Supabase |
| `get_scan_details` | Gets full breakdown for a specific scan |
| `get_violations_summary` | Aggregates violation data, filterable by severity |
| `get_compliance_rules` | Returns all configured rules from `rules.json` |
| `search_regulations` | RAG search through legal regulation documents |
| `check_zone_at_location` | Spatial zone classification for any coordinate |
| `get_scan_statistics` | Aggregate stats across all scans |
| `send_whatsapp_dispatch` | Dispatches alert to field officer via WAHA |

The agent uses **NVIDIA NIM** (Llama 3.3 70B) or **OpenRouter** as the LLM backbone. It pre-fetches live Supabase data and the latest PDF report content into the system prompt for contextual grounding, and enforces strict topic guardrails (geospatial/compliance topics only).

### Tamper-Proof Evidence Chain

Every scan generates a **SHA-256 evidence hash** from the canonical JSON of the scan results:

```python
evidence = json.dumps({"city": city, "bbox": list(bbox), "changed": changed_px}, sort_keys=True)
evidence_hash = "0x" + hashlib.sha256(evidence.encode()).hexdigest()
```

The `src/blockchain/` module provides:
- **`EvidenceHasher`** — Converts detection results into cryptographically sealed `EvidenceRecord` objects
- **`MerkleTree`** — Append-only Merkle tree for batch evidence integrity verification
- **`ProofVerifier`** — Verifies individual evidence records against the Merkle root

This ensures scan reports are **tamper-proof** and admissible as digital evidence.

---

## 📡 API Reference

The backend exposes a comprehensive REST API. Full interactive docs are available at `/docs` when the server is running.

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend (if present) |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/scan/stream` | **Main endpoint** — Streams a live satellite scan via SSE |
| `POST` | `/api/scan/quick` | Non-streaming scan (returns final result only) |
| `POST` | `/api/chat` | Send a message to the AI chatbot |
| `WS` | `/ws/chat` | WebSocket chat with streaming responses |
| `GET` | `/api/scans` | Get recent scans from database |
| `GET` | `/api/scans/violations` | Get scans with violations |
| `GET` | `/api/zones/check?lat=X&lon=Y` | Check zone classification for coordinates |

### Scan Request Body

```json
{
  "bbox": [72.78, 21.14, 72.82, 21.18],
  "city": "Vesu, Surat",
  "date_before": ["2025-01-01", "2025-03-31"],
  "date_after": ["2025-10-01", "2026-03-31"],
  "resolution": 10
}
```

### Chat Request Body

```json
{
  "message": "Show me all scans with critical violations",
  "session_id": "user123",
  "conversation_id": "",
  "language": "en",
  "scan_context": ""
}
```

### SSE Event Format

Each line streamed from `/api/scan/stream` is a JSON object:

```json
{
  "step": 4,
  "total_steps": 9,
  "status": "running",
  "message": "Running Siamese-SNN inference (1,234,567 params)...",
  "progress": 42,
  "timestamp": "2026-04-18T10:30:00.000000",
  "data": {
    "changed_pixels": 15420,
    "total_pixels": 262144,
    "change_percent": 5.88,
    "inference_time": 3.2
  }
}
```

---

## 📂 Project Structure

```
PFL-SNN-BACKEND/
├── start_server.py              # Entry point — starts FastAPI on 0.0.0.0:8000
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for environment variables
├── Dockerfile                   # Docker deployment (CPU inference)
├── docker-compose.yml           # PostGIS + ChromaDB services
├── railway.json                 # Railway PaaS deployment config
│
├── config/
│   ├── settings.py              # Global config (model hyperparams, paths, API keys)
│   ├── chatbot_prompts.py       # System prompts for the AI agent
│   └── compliance_rules.json    # Compliance rule definitions
│
├── src/
│   ├── model/                   # 🧬 Siamese-SNN Neural Network
│   │   ├── siamese_snn.py       #    Full model assembly + predict() + get_confidence_map()
│   │   ├── encoder.py           #    Siamese U-Net encoder with shared weights
│   │   ├── snn_decoder.py       #    SNN decoder with snntorch LIF neurons
│   │   ├── spike_utils.py       #    Poisson rate coding + spike map utilities
│   │   └── losses.py            #    Combined loss (CE rate loss + Weighted BCE)
│   │
│   ├── pipeline/                # 🔄 Scan Pipeline
│   │   ├── orchestrator.py      #    9-step SSE streaming pipeline generator
│   │   ├── fetcher.py           #    Sentinel Hub API data ingestion
│   │   ├── preprocessor.py      #    Band normalization and preprocessing
│   │   ├── cloud_mask.py        #    SCL-based cloud masking
│   │   └── spectral.py          #    Spectral index computation
│   │
│   ├── compliance/              # ⚖️ Compliance Engine
│   │   ├── rule_engine.py       #    Spatial rule evaluation (buffer/zone/threshold)
│   │   ├── classifier.py        #    NDVI/NDBI/MNDWI change type classification
│   │   ├── rules.json           #    JSON-configurable compliance rules
│   │   ├── geojson_export.py    #    Export violations as GeoJSON
│   │   └── postgis_client.py    #    PostGIS database client
│   │
│   ├── chatbot/                 # 🤖 Agentic AI
│   │   ├── agent.py             #    LangGraph state machine with 8 tools
│   │   ├── tools/               #    Individual tool implementations
│   │   ├── rag/                  #    RAG retriever for legal documents
│   │   ├── alerts/              #    Real-time alert notification system
│   │   └── i18n/                #    Multilingual support
│   │
│   ├── api/                     # 🌐 FastAPI Server
│   │   ├── server.py            #    All REST + WebSocket + SSE endpoints
│   │   ├── supabase_client.py   #    Supabase SDK wrapper (storage + PostGIS)
│   │   ├── waha_client.py       #    WhatsApp HTTP API client
│   │   └── ws_handler.py        #    WebSocket connection manager
│   │
│   ├── blockchain/              # 🔗 Evidence Integrity
│   │   ├── evidence_hasher.py   #    SHA-256 evidence record hashing
│   │   ├── merkle_tree.py       #    Append-only Merkle tree
│   │   ├── proof_verifier.py    #    Proof verification against Merkle root
│   │   └── audit_store.py       #    Persistent audit log store
│   │
│   ├── reporting/               # 📄 Report Generation
│   │   ├── pdf_generator.py     #    Professional compliance PDF reports
│   │   ├── notice_generator.py  #    Enforcement notice generation
│   │   └── templates/           #    Report templates
│   │
│   ├── foundation/              # 🧱 Foundation Model (MAE Pre-training)
│   │   ├── mae_model.py         #    Masked Autoencoder for satellite imagery
│   │   ├── feature_extractor.py #    ViT feature extraction
│   │   └── tile_dataset.py      #    Tile dataset for pre-training
│   │
│   ├── training/                # 🏋️ Model Training
│   │   ├── trainer.py           #    Training loop with BPTT for SNN
│   │   ├── dataset.py           #    OSCD dataset loader
│   │   └── evaluate.py          #    Evaluation metrics (F1, IoU, Accuracy)
│   │
│   ├── optimization/            # ⚡ Inference Optimization
│   │   ├── trt_exporter.py      #    TensorRT export for edge deployment
│   │   ├── fp16_converter.py    #    FP16 mixed precision conversion
│   │   ├── pruner.py            #    Network pruning (30% weight removal)
│   │   ├── city_scanner.py      #    City-scale batch scanning
│   │   └── benchmark.py         #    Inference benchmarking suite
│   │
│   ├── integration/             # 🔌 External Integrations
│   │   ├── whatsapp.py          #    WhatsApp notification service
│   │   └── active_learning.py   #    HITL feedback loop for model improvement
│   │
│   └── utils/                   # 🛠️ Utilities
│       └── url_shortener.py     #    URL shortening for WhatsApp messages
│
├── outputs/
│   └── models/
│       └── best_model.pt        # Pre-trained model weights (included in repo)
│
├── scripts/                     # 📜 Standalone Scripts
│   ├── scan_vesu_surat.py       #    Demo scan of Vesu, Surat
│   └── scan_city.py             #    Generic city scanning script
│
├── Train_SiameseSNN_Colab.ipynb # 📓 Google Colab training notebooks (v1-v3)
└── presentation.html            # 🎤 Project presentation (self-contained HTML)
```

---

## 🏋️ Training the Model

The model was trained on the **OSCD (Onera Satellite Change Detection)** dataset. Three Google Colab notebooks are included for training:

```bash
# The pre-trained weights ship with the repo at:
outputs/models/best_model.pt
```

If you want to retrain:

1. Download the OSCD dataset
2. Open `Train_SiameseSNN_Colab_v3.ipynb` in Google Colab
3. Upload the dataset and run all cells
4. Download the resulting `best_model.pt` to `outputs/models/`

**Training Configuration:**
- Encoder: `[32, 64, 128, 256]` channels
- SNN Time-steps: `T=10`
- Beta (membrane decay): `β=0.9`
- Loss: Combined CE Rate Loss + Weighted BCE (`α=0.7`)
- Change class weight: `5.0× oversampling` (handles class imbalance)
- Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-4`)

---

## 🚢 Deployment

### Docker (Local / Self-Hosted)

```bash
# Build the Docker image
docker build -t geoguard-backend .

# Run it
docker run -p 8000:8000 --env-file .env geoguard-backend
```

### Docker Compose (with PostGIS + ChromaDB)

```bash
# Start all services (PostGIS database + ChromaDB vector store)
docker-compose up -d

# Then start the app
python start_server.py
```

### Railway (Cloud PaaS)

The repo includes `railway.json` for one-click Railway deployment:

1. Push to GitHub
2. Connect repo to [Railway](https://railway.app)
3. Add environment variables in Railway dashboard
4. Deploy — Railway reads `Dockerfile` automatically

> **Note:** Railway deploys use CPU inference. SNN inference for small patches takes ~2-5 seconds on CPU, which is acceptable for the demo workflow.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Neural Network** | PyTorch 2.0+ · snntorch · timm · einops |
| **Geospatial** | Sentinel Hub SDK · Rasterio · Shapely · GeoPandas · s2cloudless |
| **API Server** | FastAPI · Uvicorn · WebSockets · Server-Sent Events |
| **AI Agent** | LangChain · LangGraph · NVIDIA NIM / OpenRouter (Llama 3.3 70B) |
| **RAG** | ChromaDB · Sentence-Transformers (all-MiniLM-L6-v2) |
| **Database** | Supabase (PostGIS + Object Storage) · SQLAlchemy · GeoAlchemy2 |
| **Reporting** | ReportLab · Pillow · Matplotlib |
| **Deployment** | Docker · Railway · Docker Compose |
| **Evidence** | SHA-256 · Merkle Trees |

---

<div align="center">

**Built with 🧠 Spiking Neural Networks and 🛰️ Real Satellite Data**

*If you found this useful, consider giving it a ⭐*

</div>
]]>
