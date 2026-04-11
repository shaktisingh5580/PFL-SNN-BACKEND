"""
FastAPI backend server with chat, WebSocket, detection, compliance, and reporting endpoints.
"""
import logging
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.error("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")

from config.settings import APP_HOST, APP_PORT, DEBUG, REPORTS_DIR, NOTICES_DIR

# ---- Pydantic Models ----

if HAS_FASTAPI:
    class ChatRequest(BaseModel):
        message: str
        session_id: str = "default"
        language: str = "en"
        scan_context: str = ""  # JSON of the current scan the user is viewing

    class ChatResponse(BaseModel):
        response: str
        language: str

    class FeedbackRequest(BaseModel):
        detection_id: str
        feedback_type: str  # "approve", "reject", "edit_bounds"
        reviewed_by: str = "dashboard_user"
        notes: str = ""
        edited_geojson: Optional[dict] = None

    class DispatchRequest(BaseModel):
        detection_id: str
        officer_phone: str
        language: str = "gu"


# ---- App Factory ----

def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn[standard]")

    app = FastAPI(
        title="Surat Satellite Compliance Engine",
        description="AI-powered satellite imagery change detection and urban compliance monitoring",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files (frontend)
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    # ---- Lazy-loaded components ----
    _components = {}

    def get_agent():
        if "agent" not in _components:
            from src.chatbot.agent import GeospatialAgent
            _components["agent"] = GeospatialAgent()
        return _components["agent"]

    def get_notifier():
        if "notifier" not in _components:
            from src.chatbot.alerts.notifier import notifier
            _components["notifier"] = notifier
        return _components["notifier"]

    def get_feedback():
        if "feedback" not in _components:
            from src.integration.active_learning import ActiveLearningManager
            _components["feedback"] = ActiveLearningManager()
        return _components["feedback"]

    def get_whatsapp():
        if "whatsapp" not in _components:
            from src.integration.whatsapp import WhatsAppNotifier
            _components["whatsapp"] = WhatsAppNotifier()
        return _components["whatsapp"]

    # ---- Routes ----

    @app.get("/")
    async def root():
        """Serve the frontend."""
        index = frontend_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Surat Satellite Compliance Engine API", "docs": "/docs"}

    @app.get("/api/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    # ===== CHAT ENDPOINTS =====

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_http(request: ChatRequest):
        """HTTP chat endpoint. Passes scan_context so the agent knows what the user is viewing."""
        agent = get_agent()
        result = agent.chat_sync(request.message, request.session_id, scan_context=request.scan_context)
        return ChatResponse(**result)

    @app.websocket("/ws/chat")
    async def chat_websocket(websocket: WebSocket):
        """Real-time chat via WebSocket with streaming responses."""
        from src.api.ws_handler import manager

        session_id = str(uuid.uuid4())
        await manager.connect(websocket, session_id)
        notifier = get_notifier()
        notifier.register_client(websocket)

        try:
            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")
                lang = data.get("language", "en")

                # Process with agent
                agent = get_agent()
                result = agent.chat_sync(message, session_id)

                # Send response
                await manager.stream_chat_response(session_id, {
                    "role": "assistant",
                    "content": result["response"],
                    "highlighted_areas": result.get("highlighted_areas", []),
                    "actions": result.get("pending_actions", []),
                    "language": result.get("language", "en"),
                })

                # Send map highlights if any
                for area in result.get("highlighted_areas", []):
                    await manager.send_highlight_area(session_id, area)

        except WebSocketDisconnect:
            manager.disconnect(session_id)
            notifier.unregister_client(websocket)

    # ===== DETECTION ENDPOINTS =====

    @app.get("/api/detections")
    async def get_detections(ward: str = None, days: int = 90):
        """Get latest detections."""
        from src.compliance.postgis_client import PostGISClient
        client = PostGISClient()
        detections = client.get_latest_detections(ward=ward, days=days)
        client.close()
        return {"detections": detections, "count": len(detections)}

    @app.get("/api/detections/{detection_id}")
    async def get_detection(detection_id: str):
        """Get detection details."""
        from src.chatbot.tools.detection_tools import get_detection_detail
        result = get_detection_detail.invoke(detection_id)
        return JSONResponse(content={"detail": result})

    # ===== COMPLIANCE ENDPOINTS =====

    @app.post("/api/compliance/check/{detection_id}")
    async def check_compliance(detection_id: str):
        """Run compliance check on a detection."""
        from src.chatbot.tools.compliance_tools import run_compliance_check
        result = run_compliance_check.invoke(detection_id)
        return {"detection_id": detection_id, "result": result}

    @app.get("/api/violations")
    async def get_violations(ward: str = None, severity: str = None):
        """List active violations."""
        from src.chatbot.tools.compliance_tools import list_violations
        result = list_violations.invoke({"ward": ward, "severity": severity})
        return {"violations": result}

    # ===== REPORT ENDPOINTS =====

    @app.post("/api/reports/{detection_id}")
    async def generate_report(detection_id: str):
        """Generate compliance report PDF."""
        from src.chatbot.tools.report_tools import generate_compliance_report
        result = generate_compliance_report.invoke(detection_id)
        return {"detection_id": detection_id, "result": result}

    @app.post("/api/notices/{detection_id}/draft")
    async def draft_notice(detection_id: str, notice_type: str = "show_cause"):
        """Generate enforcement notice PDF."""
        from src.chatbot.tools.report_tools import draft_enforcement_notice
        result = draft_enforcement_notice.invoke({
            "detection_id": detection_id,
            "notice_type": notice_type,
        })
        return {"detection_id": detection_id, "notice_type": notice_type, "result": result}

    # ===== ALERTS =====

    @app.get("/api/alerts")
    async def get_alerts(severity: str = None):
        """Get pending alerts."""
        notifier = get_notifier()
        alerts = notifier.get_pending_alerts(severity=severity)
        return {"alerts": alerts, "stats": notifier.get_alert_stats()}

    @app.post("/api/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str):
        """Mark an alert as reviewed."""
        notifier = get_notifier()
        notifier.acknowledge_alert(alert_id)
        return {"alert_id": alert_id, "status": "acknowledged"}

    # ===== HITL FEEDBACK =====

    @app.post("/api/feedback")
    async def submit_feedback(request: FeedbackRequest):
        """Submit human feedback on a detection."""
        from src.integration.active_learning import FeedbackRecord
        fm = get_feedback()
        record = FeedbackRecord(
            detection_id=request.detection_id,
            feedback_type=request.feedback_type,
            reviewed_by=request.reviewed_by,
            timestamp=datetime.utcnow().isoformat(),
            original_confidence=0.0,
            change_type="unknown",
            lat=0.0, lon=0.0,
            notes=request.notes,
            edited_geojson=request.edited_geojson,
        )
        result = fm.submit_feedback(record)
        return result

    @app.get("/api/feedback/stats")
    async def feedback_stats():
        """Get HITL feedback statistics."""
        return get_feedback().get_feedback_stats()

    # ===== WHATSAPP DISPATCH =====

    @app.post("/api/dispatch")
    async def dispatch_officer(request: DispatchRequest):
        """Send WhatsApp alert to field officer."""
        wa = get_whatsapp()
        detection = {"id": request.detection_id, "lat": 21.17, "lon": 72.83,
                      "change_type": "new_construction", "confidence": 0.87,
                      "area_hectares": 0.45}
        result = wa.dispatch_officer(request.officer_phone, detection, request.language)
        return result

    # ===== RAG / REGULATIONS SEARCH =====

    @app.get("/api/regulations/search")
    async def search_regulations(query: str, top_k: int = 5):
        """Direct RAG search endpoint."""
        from src.chatbot.rag.retriever import RegulatoryRetriever
        retriever = RegulatoryRetriever()
        results = retriever.retrieve(query, top_k=top_k)
        return {"query": query, "results": results, "count": len(results)}

    # ===== LIVE SCAN (SSE Streaming) =====

    class ScanRequest(BaseModel):
        bbox: list  # [lon_min, lat_min, lon_max, lat_max]
        city: str = "Custom Scan"
        date_before: list = ["2025-01-01", "2025-03-31"]  # [start, end]
        date_after: list = ["2025-10-01", "2026-03-31"]
        resolution: int = 10

    @app.post("/api/scan/stream")
    async def scan_stream(request: ScanRequest):
        """
        Stream a live satellite scan via Server-Sent Events (SSE).

        The frontend connects to this endpoint with a POST containing the
        bounding box drawn on the map. Progress events are streamed back
        in real-time as the pipeline runs each step.

        Each line is a JSON object with:
          step, total_steps, status, message, progress (0-100), data
        """
        from fastapi.responses import StreamingResponse
        from src.pipeline.orchestrator import run_pipeline

        bbox_tuple = tuple(request.bbox)

        def event_stream():
            for event in run_pipeline(
                bbox=bbox_tuple,
                city=request.city,
                date_before=tuple(request.date_before),
                date_after=tuple(request.date_after),
                resolution=request.resolution,
            ):
                yield f"data: {event}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/scan/quick")
    async def scan_quick(request: ScanRequest):
        """
        Non-streaming scan endpoint. Runs the full pipeline and
        returns the final result JSON. Useful for testing.
        """
        from src.pipeline.orchestrator import run_pipeline

        bbox_tuple = tuple(request.bbox)
        last_event = None
        for event in run_pipeline(
            bbox=bbox_tuple,
            city=request.city,
            date_before=tuple(request.date_before),
            date_after=tuple(request.date_after),
            resolution=request.resolution,
        ):
            last_event = json.loads(event.strip())

        return JSONResponse(content=last_event or {"error": "Pipeline returned no events"})

    # ===== SUPABASE DATA ENDPOINTS (for dashboard) =====

    @app.get("/api/scans")
    async def get_scans(city: str = None, limit: int = 20):
        """Get recent scans from Supabase."""
        from src.api.supabase_client import SupabaseClient
        sb = SupabaseClient()
        if city:
            scans = sb.get_scans_by_city(city)
        else:
            scans = sb.get_recent_scans(limit=limit)
        return {"scans": scans, "count": len(scans)}

    @app.get("/api/scans/violations")
    async def get_scan_violations(severity: str = None):
        """Get scans with violations."""
        from src.api.supabase_client import SupabaseClient
        sb = SupabaseClient()
        scans = sb.get_violations(severity=severity)
        return {"scans": scans, "count": len(scans)}

    # ===== ZONE QUERY =====

    @app.get("/api/zones/check")
    async def check_zone(lat: float, lon: float):
        """Check zone classification for coordinates."""
        from src.compliance.rule_engine import generate_mock_zones
        from shapely.geometry import Point
        bbox_approx = (lon - 0.015, lat - 0.0125, lon + 0.015, lat + 0.0125)
        zones = generate_mock_zones(bbox_approx)
        pt = Point(lon, lat)
        results = {}
        for name, geom in zones.items():
            results[name] = pt.within(geom)
        return {"lat": lat, "lon": lon, "zones": results}

    return app


# ---- Entry Point ----

app = create_app() if HAS_FASTAPI else None


def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
    )


if __name__ == "__main__":
    main()
