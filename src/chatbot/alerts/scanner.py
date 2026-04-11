"""
Proactive scan scheduler.
Periodically scans for new violations and pushes alerts to connected clients.
"""
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)

from config.settings import SCAN_INTERVAL_MINUTES, SEVERITY_THRESHOLDS


class ProactiveScanner:
    """
    Background scan scheduler that periodically checks for new violations
    and pushes alerts to the dashboard / chatbot.
    """

    def __init__(
        self,
        interval_minutes: int = None,
        on_alert: Optional[Callable] = None,
    ):
        self.interval = (interval_minutes or SCAN_INTERVAL_MINUTES) * 60  # seconds
        self.on_alert = on_alert  # Callback for new alerts
        self._running = False
        self._task = None
        self._scan_count = 0
        self._alerts_generated = 0

    async def start(self):
        """Start the background scan loop."""
        self._running = True
        logger.info(f"Proactive scanner started (interval: {self.interval}s)")
        self._task = asyncio.create_task(self._scan_loop())

    async def stop(self):
        """Stop the background scan loop."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info(f"Scanner stopped. Total: {self._scan_count} scans, {self._alerts_generated} alerts")

    async def _scan_loop(self):
        """Main scan loop."""
        while self._running:
            try:
                alerts = await self._run_scan()
                for alert in alerts:
                    self._alerts_generated += 1
                    if self.on_alert:
                        await self.on_alert(alert)
                self._scan_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan failed: {e}")

            await asyncio.sleep(self.interval)

    async def _run_scan(self) -> List[Dict]:
        """
        Run a single scan cycle:
        1. Fetch latest unreviewed detections
        2. Run compliance checks
        3. Generate alerts for violations
        """
        from src.compliance.postgis_client import PostGISClient
        client = PostGISClient()

        # Get unreviewed detections
        detections = client.get_latest_detections(days=7)
        pending = [d for d in detections if d.get("status") == "pending"]

        if not pending:
            logger.debug(f"Scan {self._scan_count}: No pending detections")
            return []

        alerts = []
        for detection in pending:
            confidence = detection.get("confidence", 0)
            severity = self._classify_severity(confidence)

            if severity in ("HIGH", "CRITICAL"):
                alert = {
                    "alert_id": f"ALT-{self._alerts_generated + len(alerts) + 1:04d}",
                    "detection_id": detection.get("id", "unknown"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": severity,
                    "change_type": detection.get("change_type", "unknown"),
                    "confidence": confidence,
                    "lat": detection.get("lat", 0),
                    "lon": detection.get("lon", 0),
                    "area_hectares": detection.get("area_hectares", 0),
                    "message": self._generate_alert_message(detection, severity),
                    "geojson": self._detection_to_geojson(detection),
                    "actions": ["generate_report", "draft_notice", "show_on_map", "dispatch_officer"],
                }
                alerts.append(alert)
                logger.info(f"Alert generated: {alert['alert_id']} ({severity})")

        client.close()
        return alerts

    def _classify_severity(self, confidence: float) -> str:
        """Classify alert severity based on SNN confidence."""
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if confidence >= SEVERITY_THRESHOLDS.get(level, 0):
                return level
        return "LOW"

    def _generate_alert_message(self, detection: Dict, severity: str) -> str:
        """Generate human-readable alert message."""
        emoji = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "⛔"}.get(severity, "ℹ️")
        ct = detection.get("change_type", "change").replace("_", " ")
        conf = detection.get("confidence", 0) * 100

        return (
            f"{emoji} **NEW ALERT**: Detected {ct} "
            f"(Confidence: {conf:.0f}%) at "
            f"{detection.get('lat', 0):.4f}°N, {detection.get('lon', 0):.4f}°E | "
            f"Area: {detection.get('area_hectares', 0):.2f} hectares. "
            f"Would you like me to generate a report or draft a notice?"
        )

    def _detection_to_geojson(self, detection: Dict) -> Dict:
        """Convert detection to GeoJSON point for map highlighting."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [detection.get("lon", 0), detection.get("lat", 0)],
            },
            "properties": {
                "detection_id": detection.get("id", ""),
                "severity": self._classify_severity(detection.get("confidence", 0)),
                "change_type": detection.get("change_type", ""),
            },
        }

    async def trigger_manual_scan(self) -> List[Dict]:
        """Trigger an immediate scan (e.g., from chatbot or API)."""
        logger.info("Manual scan triggered")
        return await self._run_scan()
