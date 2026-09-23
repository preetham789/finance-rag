# scripts/start.py
"""
Start the Finance RAG system.

Usage:
  python scripts\start.py          # starts both API and UI
  python scripts\start.py --api    # API only
  python scripts\start.py --ui     # UI only (API must be running)
"""
import sys
import subprocess
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path


def wait_for_api(url: str = "http://127.0.0.1:8000/health",
                 timeout: int = 120) -> bool:
    """
    N4 FIX: Poll /health until the API is ready instead of sleeping a fixed 3s.
    On first run the SentenceTransformer model downloads ~90MB — that takes
    far longer than 3 seconds, causing Streamlit to show 'API offline'.
    """
    print("  Waiting for API to become ready", end="", flush=True)
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            print(" ✓")
            return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print(" TIMEOUT")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--ui",  action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    api_cmd = [sys.executable, "-m", "uvicorn",
               "src.api.main:app",
               "--host", "127.0.0.1",   # N3 FIX: was 0.0.0.0 — exposed API to LAN
               "--port", "8000",
               "--reload"]

    ui_cmd  = [sys.executable, "-m", "streamlit",
               "run", str(root / "src" / "api" / "streamlit_app.py"),
               "--server.port", "8501"]

    if args.api:
        print("Starting API at http://127.0.0.1:8000")
        print("API docs at  http://127.0.0.1:8000/docs")
        subprocess.run(api_cmd, cwd=root)

    elif args.ui:
        print("Starting UI at http://localhost:8501")
        subprocess.run(ui_cmd, cwd=root)

    else:
        # Start API in background, wait until healthy, then start UI
        print("Starting Finance RAG System")
        print("  API → http://127.0.0.1:8000")
        print("  UI  → http://localhost:8501")
        print("  Docs→ http://127.0.0.1:8000/docs")
        print("\nPress Ctrl+C to stop\n")

        api_proc = subprocess.Popen(api_cmd, cwd=root)
        ready = wait_for_api()   # N4 FIX: health-check poll, not sleep(3)
        if not ready:
            print("ERROR: API did not start in time. Check logs above.")
            api_proc.terminate()
            sys.exit(1)
        try:
            subprocess.run(ui_cmd, cwd=root)
        finally:
            api_proc.terminate()


if __name__ == "__main__":
    main()