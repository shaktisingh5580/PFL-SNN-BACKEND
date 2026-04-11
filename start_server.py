"""
Start the Satellite Compliance API Server.

Binds to 0.0.0.0 so other devices on your network can connect.
Find your IP with: ipconfig (Windows) / ifconfig (Mac/Linux)

Other laptops can connect to: http://<YOUR_IP>:8000
API docs available at:        http://<YOUR_IP>:8000/docs
"""
import os
import sys
import socket
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


def get_local_ip():
    """Get this machine's local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    HOST = "0.0.0.0"
    PORT = int(os.getenv("APP_PORT", 8000))
    local_ip = get_local_ip()

    print("=" * 60)
    print("  Satellite Compliance Engine - API Server")
    print("=" * 60)
    print(f"  Local:    http://localhost:{PORT}")
    print(f"  Network:  http://{local_ip}:{PORT}")
    print(f"  API Docs: http://{local_ip}:{PORT}/docs")
    print(f"  Scan SSE: POST http://{local_ip}:{PORT}/api/scan/stream")
    print("=" * 60)
    print()

    uvicorn.run(
        "src.api.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
