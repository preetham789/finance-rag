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
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--ui",  action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    api_cmd = [sys.executable, "-m", "uvicorn",
               "src.api.main:app",
               "--host", "0.0.0.0",
               "--port", "8000",
               "--reload"]

    ui_cmd  = [sys.executable, "-m", "streamlit",
               "run", str(root / "src" / "api" / "streamlit_app.py"),
               "--server.port", "8501"]

    if args.api:
        print("Starting API at http://localhost:8000")
        print("API docs at  http://localhost:8000/docs")
        subprocess.run(api_cmd, cwd=root)

    elif args.ui:
        print("Starting UI at http://localhost:8501")
        subprocess.run(ui_cmd, cwd=root)

    else:
        # Start API in background, UI in foreground
        print("Starting Finance RAG System")
        print("  API → http://localhost:8000")
        print("  UI  → http://localhost:8501")
        print("  Docs→ http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop\n")

        api_proc = subprocess.Popen(api_cmd, cwd=root)
        time.sleep(3)   # give API time to load models
        try:
            subprocess.run(ui_cmd, cwd=root)
        finally:
            api_proc.terminate()

if __name__ == "__main__":
    main()