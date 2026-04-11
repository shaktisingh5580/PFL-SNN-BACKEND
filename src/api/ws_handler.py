"""
WebSocket handler for real-time chat and alert streaming.
Manages client connections, message routing, and live scan updates.
"""
import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Set, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import WebSocket, WebSocketDisconnect
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class ConnectionManager:
    """Manages WebSocket connections for chat and scan streaming."""

    def __init__(self):
        self._active_connections: Dict[str, WebSocket] = {}  # session_id -> websocket
        self._chat_sessions: Dict[str, list] = {}  # session_id -> message history

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self._active_connections[session_id] = websocket
        self._chat_sessions.setdefault(session_id, [])
        logger.info(f"WebSocket connected: {session_id} (total: {len(self._active_connections)})")

    def disconnect(self, session_id: str):
        """Remove a disconnected client."""
        self._active_connections.pop(session_id, None)
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        """Send a message to a specific client."""
        ws = self._active_connections.get(session_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Send failed for {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast(self, message: dict, exclude: str = None):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for sid, ws in self._active_connections.items():
            if sid == exclude:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(sid)

        for sid in disconnected:
            self.disconnect(sid)

    async def stream_chat_response(
        self, session_id: str, response: dict
    ):
        """
        Stream a chat response to the client.

        Message format:
        {
            "type": "chat_response",
            "data": {
                "role": "assistant",
                "content": "...",
                "highlighted_areas": [...],
                "actions": [...],
            },
            "timestamp": "..."
        }
        """
        await self.send_message(session_id, {
            "type": "chat_response",
            "data": response,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def stream_scan_progress(
        self, completed: int, total: int, result: dict
    ):
        """
        Stream city scan progress to all clients.

        Message format:
        {
            "type": "scan_progress",
            "data": {
                "completed": 847,
                "total": 2400,
                "percent": 35.3,
                "latest_tile": { ... },
                "eta_seconds": 28,
            }
        }
        """
        eta = ((total - completed) / max(completed, 1)) * (result.get("processing_time_ms", 200) / 1000)
        await self.broadcast({
            "type": "scan_progress",
            "data": {
                "completed": completed,
                "total": total,
                "percent": round(completed / total * 100, 1) if total > 0 else 0,
                "latest_tile": result,
                "eta_seconds": round(eta, 1),
            },
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def send_highlight_area(self, session_id: str, geojson: dict):
        """Send a map highlight event to the client."""
        await self.send_message(session_id, {
            "type": "highlight_area",
            "data": {"geojson": geojson},
            "timestamp": datetime.utcnow().isoformat(),
        })

    @property
    def active_count(self) -> int:
        return len(self._active_connections)


# Global connection manager
manager = ConnectionManager()
