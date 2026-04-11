"""
SNN change detection tools for the LangGraph agent.
Query the latest detections, details, and spike statistics.
"""
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    def tool(func):
        func.is_tool = True
        return func

from src.compliance.postgis_client import PostGISClient

_client = PostGISClient()


@tool
def get_latest_detections(ward: str = None, days: int = 90) -> str:
    """Get latest SNN change detections, optionally filtered by ward.
    Returns list of detections with coordinates, confidence, change type."""
    detections = _client.get_latest_detections(ward=ward, days=days)

    if not detections:
        ward_str = f" in {ward}" if ward else ""
        return f"No change detections found{ward_str} in the last {days} days."

    lines = [f"📡 Found {len(detections)} detections (last {days} days):\n"]
    for i, d in enumerate(detections, 1):
        confidence = d.get("confidence", 0) * 100
        lines.append(
            f"  {i}. **{d.get('change_type', 'unknown').replace('_', ' ').title()}** "
            f"(Confidence: {confidence:.0f}%)\n"
            f"     📍 Location: {d.get('lat', 'N/A')}°N, {d.get('lon', 'N/A')}°E\n"
            f"     📐 Area: {d.get('area_hectares', 0):.2f} hectares\n"
            f"     📅 Date: {d.get('detection_date', 'N/A')}\n"
            f"     🏷️ Status: {d.get('status', 'pending')}\n"
        )

    return "\n".join(lines)


@tool
def get_detection_detail(detection_id: str) -> str:
    """Get full detail for a specific detection: spectral indices,
    SNN confidence score, affected zone, and any triggered compliance rules."""
    if _client._mock_mode:
        return json.dumps({
            "detection_id": detection_id,
            "change_type": "new_construction",
            "confidence": 0.87,
            "area_hectares": 0.45,
            "lat": 21.1756,
            "lon": 72.8312,
            "spectral_indices": {
                "ndvi_before": 0.62, "ndvi_after": 0.18, "ndvi_delta": -0.44,
                "ndbi_before": 0.15, "ndbi_after": 0.67, "ndbi_delta": 0.52,
                "mndwi_before": -0.30, "mndwi_after": -0.45, "mndwi_delta": -0.15,
            },
            "spike_statistics": {
                "mean_spike_rate": 0.73,
                "peak_spike_density": 0.91,
                "avg_spike_latency_ms": 3.2,
                "total_active_neurons": 18432,
            },
            "zone": "Residential Zone R1",
            "violated_rules": ["R001 - Water body buffer", "R002 - Green cover minimum"],
        }, indent=2)

    with _client.get_cursor() as cur:
        cur.execute("""
            SELECT d.*, z.zone_name, z.zone_type,
                   ST_Y(d.coordinates) as lat, ST_X(d.coordinates) as lon
            FROM detections d
            LEFT JOIN zone_boundaries z ON d.zone_id = z.id
            WHERE d.id = %s::uuid
        """, (detection_id,))
        row = cur.fetchone()
        if row:
            return json.dumps(dict(row), default=str, indent=2)
    return f"Detection {detection_id} not found."


@tool
def get_spike_statistics(ward: str, months: int = 3) -> str:
    """Get aggregate spike statistics for a ward over time.
    E.g., '70% increase in Built-up spikes in Ward 12 over 3 months'."""
    # Mock aggregation for demo
    stats = {
        "ward": ward,
        "period_months": months,
        "total_detections": 12,
        "by_type": {
            "new_construction": {"count": 5, "trend": "+70%", "avg_confidence": 0.82},
            "vegetation_loss": {"count": 4, "trend": "+25%", "avg_confidence": 0.88},
            "road_expansion": {"count": 2, "trend": "+10%", "avg_confidence": 0.71},
            "water_change": {"count": 1, "trend": "0%", "avg_confidence": 0.65},
        },
        "hotspots": [
            {"area": f"{ward} Sector A", "detections": 3, "severity": "HIGH"},
            {"area": f"{ward} Sector B", "detections": 2, "severity": "MEDIUM"},
        ],
    }

    lines = [
        f"📊 Spike Statistics for {ward} (last {months} months):\n",
        f"Total detections: {stats['total_detections']}\n",
    ]
    for ctype, data in stats["by_type"].items():
        emoji = "🏗️" if "construction" in ctype else "🌿" if "vegetation" in ctype else "🛣️" if "road" in ctype else "💧"
        lines.append(
            f"  {emoji} {ctype.replace('_', ' ').title()}: {data['count']} detections "
            f"(Trend: {data['trend']}, Avg Confidence: {data['avg_confidence']:.0%})"
        )

    lines.append(f"\n🔴 Hotspots:")
    for hs in stats["hotspots"]:
        lines.append(f"  - {hs['area']}: {hs['detections']} detections (Severity: {hs['severity']})")

    return "\n".join(lines)
