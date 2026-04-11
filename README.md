# PFL-SNN-BACKEND

Agentic Geospatial Compliance System for tracking urban sprawl and environmental compliance violations using a hybrid Siamese-Spiking Neural Network (SNN) and Spectral indices (NDVI/NDBI/MNDWI), powered by LangGraph.

## Quick Start (Demo Mode)

For hackathon demonstrations, the system includes a pre-configured live scan of Vesu, Surat. This scans a 3.1km \u00d7 2.7km area for new construction, vegetation loss, and water body alterations.

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure you have your `.env` copied with the SentinelHub, Supabase, and OpenRouter API keys)

2. **Run the Vesu Demo Scan**:
   ```bash
   # This runs the backend pipeline directly on the Vesu region
   python scripts/scan_vesu_surat.py
   ```
   *This will fetch live Copernicus CDSE satellite imagery, run the Siamese-SNN inference, generate a compliance classification map, identify rule violations, and generate a tamper-proof PDF with an Evidence Hash.*

3. **Run the Interactive Dashboard (Frontend + Backend)**:
   - To connect the frontend UI to this backend, run the API server:
     ```bash
     python start_server.py
     ```
   - The server will boot up and host the real-time SSE event pipeline on `http://0.0.0.0:8000`. You can then trigger scans and multi-modal chats directly from your frontend dashboard on the same local network.

## System Architecture

- **`src/pipeline/orchestrator.py`**: The SSE event generator that controls the scan pipeline.
- **`src/model/`**: The PyTorch Siamese Spiking Neural Network (SNN).
- **`src/compliance/`**: The GDCR Rule Engine and Change Type Spectral Classifier.
- **`src/chatbot/`**: LangGraph integration using OpenRouter/Gemini to interrogate compliance PDF reports.
