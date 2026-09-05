"""
Event logging for the AI Smart Pest Trap.

Always logs locally to a CSV file (plus an optional captured-frame snapshot).
Optionally POSTs each detection as JSON to the dashboard endpoint when
config.LOG_TO_CLOUD is True.
"""

import csv
import os
from datetime import datetime, timezone

import config

_CSV_HEADER = ["timestamp_utc", "label", "confidence", "category", "action_taken", "snapshot"]
_cloud_failed = False  # after the first failure we stop spamming the console


def _ensure_log_file():
    log_dir = os.path.dirname(config.LOCAL_LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(config.LOCAL_LOG_FILE):
        with open(config.LOCAL_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_HEADER)


def save_snapshot(image, timestamp: str) -> str:
    """Save a frame as JPEG; returns the file name ('' on failure)."""
    if image is None:
        return ""
    try:
        from PIL import Image

        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        name = timestamp.replace(":", "-").replace("+", "_") + ".jpg"
        img = Image.fromarray(image)
        img.thumbnail((640, 640))
        img.convert("RGB").save(os.path.join(config.SNAPSHOT_DIR, name), "JPEG", quality=82)
        _prune_snapshots()
        return name
    except Exception as e:
        print(f"[cloud_logger] Failed to save snapshot: {e}")
        return ""


def _prune_snapshots():
    try:
        files = sorted(
            (os.path.join(config.SNAPSHOT_DIR, f) for f in os.listdir(config.SNAPSHOT_DIR)),
            key=os.path.getmtime,
        )
        for path in files[: -config.MAX_SNAPSHOTS]:
            os.remove(path)
    except Exception:
        pass


def log_detection(label: str, confidence: float, category: str, action_taken: bool,
                  image=None, top=None, post=None):
    """Record one detection: snapshot -> CSV -> optional cloud POST.

    Returns (timestamp, snapshot) so callers can mirror the event elsewhere.
    """
    _ensure_log_file()
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot = save_snapshot(image, timestamp)

    with open(config.LOCAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, label, f"{confidence:.3f}", category, action_taken, snapshot])

    if (config.LOG_TO_CLOUD if post is None else post):
        _post_to_cloud(timestamp, label, confidence, category, action_taken, snapshot, top)

    return timestamp, snapshot


def _post_to_cloud(timestamp, label, confidence, category, action_taken, snapshot, top):
    global _cloud_failed
    if _cloud_failed:
        return
    try:
        import requests

        resp = requests.post(
            config.CLOUD_ENDPOINT_URL,
            json={
                "timestamp": timestamp,
                "label": label,
                "confidence": confidence,
                "category": category,
                "action_taken": action_taken,
                "snapshot": snapshot,
                "top": top or [],
            },
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as e:
        _cloud_failed = True
        print(f"[cloud_logger] Dashboard unreachable ({e}); continuing with CSV-only logging.")
