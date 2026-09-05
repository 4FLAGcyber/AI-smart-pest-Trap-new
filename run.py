"""
One-command launcher: starts the web dashboard, then runs the detection
pipeline on the live feed — all in a single process so the camera is only
opened once (by the shared hub).

    python run.py                     # auto camera source, simulated trap
    python run.py --source webcam     # force webcam
    python run.py --source demo       # force bundled demo photos
    python run.py --once              # single cycle (demo rehearsal)

Open http://127.0.0.1:8000 for the live dashboard (live cam, upload, stats).
"""

import argparse
import json
import socket
import threading
import time
import urllib.request

import config


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _http_json(url, payload=None, timeout=2):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def start_dashboard() -> bool:
    """Start the FastAPI dashboard in a daemon thread. Returns True on success."""
    import uvicorn
    from dashboard import app

    if not _port_free(config.DASHBOARD_HOST, config.DASHBOARD_PORT):
        print(f"[dashboard] Port {config.DASHBOARD_PORT} already in use — "
              f"assuming a dashboard is already running.")
        return True

    cfg = uvicorn.Config(app, host=config.DASHBOARD_HOST,
                         port=config.DASHBOARD_PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()

    url = f"http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}/api/stats"
    for _ in range(50):
        try:
            _http_json(url)
            print(f"[dashboard] Live at http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
            return True
        except Exception:
            time.sleep(0.2)
    print("[dashboard] Failed to start.")
    return False


def main():
    parser = argparse.ArgumentParser(description="AI Smart Pest Trap — dashboard + pipeline")
    parser.add_argument("--source", choices=["auto", "webcam", "demo"], default=None)
    parser.add_argument("--simulate-trap", action="store_true", default=None)
    parser.add_argument("--real-trap", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-start", action="store_true",
                        help="start the detection pipeline immediately instead of waiting for the web UI")
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--no-cloud", action="store_true")
    args = parser.parse_args()

    if args.no_cloud:
        config.LOG_TO_CLOUD = False
    if args.interval is not None:
        config.CAPTURE_INTERVAL_SECONDS = args.interval
    if args.source is not None:
        config.CAMERA_SOURCE = args.source

    base = f"http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}"
    if not start_dashboard():
        return

    simulate = True if args.simulate_trap else (False if args.real_trap else None)
    if args.once or args.auto_start:
        try:
            _http_json(f"{base}/api/pipeline/start",
                       {"source": args.source, "simulate": simulate}, timeout=30)
        except Exception as e:
            print(f"[run] Could not start pipeline: {e}")
            return

    try:
        if args.once:
            for _ in range(120):
                if _http_json(f"{base}/api/stats")["total"] >= 1:
                    break
                time.sleep(0.5)
            _http_json(f"{base}/api/pipeline/stop", {})
            print("[run] Single cycle complete.")
        else:
            if args.auto_start:
                print("[run] Pipeline running. Press Ctrl+C to stop.")
            else:
                print("[run] Dashboard ready - press 'Start detection' in the web UI.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[run] Stopping...")
    finally:
        try:
            _http_json(f"{base}/api/pipeline/stop", {})
        except Exception:
            pass


if __name__ == "__main__":
    main()
