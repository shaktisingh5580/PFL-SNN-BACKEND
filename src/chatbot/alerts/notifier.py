"""
Push alert notification system.
Generates alert messages and broadcasts to connected WebSocket clients.
"""
import logging
import json
from datetime import datetime
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)


class AlertNotifier:
    """
    Manages alert broadcasting to connected WebSocket dashboard clients.
    """

    def __init__(self):
        self._connected_clients: Set = set()
        self._alert_history: List[Dict] = []
        self._acknowledged: Set[str] = set()

    def register_client(self, websocket):
        """Register a new WebSocket client."""
        self._connected_clients.add(websocket)
        logger.info(f"Client registered. Total: {len(self._connected_clients)}")

    def unregister_client(self, websocket):
        """Remove a disconnected client."""
        self._connected_clients.discard(websocket)
        logger.info(f"Client unregistered. Total: {len(self._connected_clients)}")

    async def broadcast_alert(self, alert: Dict):
        """
        Broadcast an alert to all connected WebSocket clients.

        Alert format:
        {
            "type": "alert",
            "data": {
                "alert_id": "ALT-0001",
                "severity": "HIGH",
                "message": "...",
                "geojson": { ... },
                "actions": [...]
            }
        }
        """
        self._alert_history.append(alert)

        message = json.dumps({
            "type": "alert",
            "data": alert,
            "timestamp": datetime.utcnow().isoformat(),
        })

        disconnected = set()
        for client in self._connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)

        for client in disconnected:
            self.unregister_client(client)

        logger.info(f"Alert {alert.get('alert_id', '?')} broadcast to "
                     f"{len(self._connected_clients)} clients")

    async def send_chat_alert(self, websocket, alert: Dict):
        """Send an alert as a proactive chat message to a specific client."""
        message = json.dumps({
            "type": "chat_response",
            "data": {
                "role": "assistant",
                "content": alert["message"],
                "is_alert": True,
                "alert_id": alert.get("alert_id"),
                "severity": alert.get("severity"),
                "geojson": alert.get("geojson"),
                "actions": alert.get("actions", []),
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send chat alert: {e}")

    def acknowledge_alert(self, alert_id: str, reviewed_by: str = "system"):
        """Mark an alert as acknowledged/reviewed."""
        self._acknowledged.add(alert_id)
        logger.info(f"Alert {alert_id} acknowledged by {reviewed_by}")

    def get_pending_alerts(
        self, severity: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Get unacknowledged alerts."""
        pending = [
            a for a in self._alert_history
            if a.get("alert_id") not in self._acknowledged
        ]
        if severity:
            pending = [a for a in pending if a.get("severity") == severity.upper()]
        return pending[:limit]

    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        total = len(self._alert_history)
        acked = len(self._acknowledged)
        severity_counts = {}
        for a in self._alert_history:
            s = a.get("severity", "UNKNOWN")
            severity_counts[s] = severity_counts.get(s, 0) + 1

        return {
            "total_alerts": total,
            "acknowledged": acked,
            "pending": total - acked,
            "by_severity": severity_counts,
            "connected_clients": len(self._connected_clients),
        }


# Global notifier instance
notifier = AlertNotifier()
