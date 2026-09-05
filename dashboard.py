"""
Cloud dashboard for the AI Smart Pest Trap.

Receives detection events (POST /api/detections — the target of
config.CLOUD_ENDPOINT_URL), persists them in SQLite, and serves a live web UI
with three interactive features:

  * Live cam   GET  /api/stream          MJPEG feed from the shared camera
  * Upload     POST /api/classify        analyse an uploaded photo
  * Detection  POST /api/pipeline/start  run detect->classify->decide->act
               POST /api/pipeline/stop   on the live feed
               GET  /api/pipeline/status

Run standalone:      python dashboard.py
Or via the launcher: python run.py
"""

import io
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config
import chatbot
import cloud_logger
import live_camera
from pest_categories import CATEGORY_HARMFUL
from subject_gate import evaluate_subject
from trap_controller import make_trap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "logs", "dashboard.db")
UI_PATH = os.path.join(BASE_DIR, "dashboard", "index.html")
SNAPSHOT_ROOT = os.path.abspath(config.SNAPSHOT_DIR)

app = FastAPI(title="AI Smart Pest Trap Dashboard", version="2.0")

hub = live_camera.CameraHub()

# ---------------------------------------------------------------- AI model --
_classifier = None
_classifier_lock = threading.Lock()


def get_classifier():
    """Lazy, thread-safe classifier singleton (model load is slow)."""
    global _classifier
    with _classifier_lock:
        if _classifier is None:
            from classifier import InsectClassifier
            print("[ai] Loading classifier model...")
            _classifier = InsectClassifier()
            print("[ai] Classifier ready.")
        return _classifier


# Warm the model in the background so the first upload/cycle is fast.
threading.Thread(target=get_classifier, daemon=True).start()


# ---------------------------------------------------------------- database --
def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                category TEXT NOT NULL,
                action_taken INTEGER NOT NULL,
                snapshot TEXT,
                top TEXT,
                source TEXT DEFAULT 'live'
            )
            """
        )
        # Upgrade DBs created before the source column existed.
        try:
            conn.execute("ALTER TABLE detections ADD COLUMN source TEXT DEFAULT 'live'")
        except sqlite3.OperationalError:
            pass


init_db()
_start_time = time.time()


def store_detection(event: dict):
    """Insert one detection event into SQLite (shared by POST + worker)."""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO detections (timestamp, label, confidence, category,"
            " action_taken, snapshot, top, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(event.get("timestamp", "") or ""),
                str(event.get("label", "unknown")),
                float(event.get("confidence", 0.0)),
                str(event.get("category", "Harmless")),
                int(bool(event.get("action_taken", False))),
                str(event.get("snapshot", "") or ""),
                json.dumps(event.get("top", []) or []),
                str(event.get("source", "live")),
            ),
        )


# ------------------------------------------------------- detection worker --
class PipelineWorker:
    """Runs detect -> classify -> decide -> act on the shared live feed."""

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self.source_kind = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source=None, simulate=None) -> bool:
        if self.running:
            return False
        requested = (source or config.CAMERA_SOURCE).lower()
        if hub.running and requested != "auto" and requested != hub.source_kind:
            hub.stop()  # explicit source switch: restart the capture thread
        self.source_kind = hub.start(source)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(simulate,), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        hub.stop()  # camera runs only while detection is running

    def _loop(self, simulate):
        from main import decide_and_act

        classifier = get_classifier()
        trap = make_trap(simulate)
        interval = config.CAPTURE_INTERVAL_SECONDS
        print(f"[pipeline] started on '{self.source_kind}' feed (interval {interval}s)")

        try:
            while not self._stop.is_set():
                frame = hub.get_frame()
                if frame is None:
                    time.sleep(0.2)
                    continue

                prediction = classifier.classify(frame)
                verdict = evaluate_subject(prediction)
                if not verdict.accepted:
                    print(
                        f"[pipeline] Rejected {verdict.subject} frame "
                        f"({verdict.reason})"
                    )
                    time.sleep(interval)
                    continue
                print(
                    f"[detect] label='{prediction.label}' "
                    f"confidence={prediction.confidence:.2f} "
                    f"category={prediction.category} (live: {self.source_kind})"
                )
                action_taken = decide_and_act(prediction, trap)

                # CSV + snapshot without POSTing back into ourselves;
                # the DB row is written directly.
                timestamp, snapshot = cloud_logger.log_detection(
                    label=prediction.label,
                    confidence=prediction.confidence,
                    category=prediction.category,
                    action_taken=action_taken,
                    image=frame,
                    top=prediction.top,
                    post=False,
                )
                store_detection({
                    "timestamp": timestamp,
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "category": prediction.category,
                    "action_taken": action_taken,
                    "snapshot": snapshot,
                    "top": prediction.top,
                    "source": "live",
                })
                time.sleep(interval)
        finally:
            trap.close()
            print("[pipeline] stopped")


worker = PipelineWorker()


# ---------------------------------------------------------------- endpoints --
@app.post("/api/detections")
async def ingest_detection(request: Request):
    try:
        event = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    store_detection(event)
    return {"ok": True}


@app.get("/api/detections")
def list_detections(limit: int = 30):
    limit = max(1, min(limit, 500))
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "label": r["label"],
            "confidence": r["confidence"],
            "category": r["category"],
            "action_taken": bool(r["action_taken"]),
            "snapshot": r["snapshot"],
            "top": json.loads(r["top"] or "[]"),
            "source": r["source"] or "live",
        }
        for r in rows
    ]


@app.delete("/api/detections/{det_id}")
def delete_detection(det_id: int):
    """Remove one detection record and its snapshot file."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT snapshot FROM detections WHERE id = ?", (det_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Detection not found")
        conn.execute("DELETE FROM detections WHERE id = ?", (det_id,))
    return {"ok": True, "removed_file": _remove_snapshot(row["snapshot"])}


@app.delete("/api/detections")
def clear_detections():
    """Remove every detection record and all snapshot files."""
    with get_db() as conn:
        rows = conn.execute("SELECT snapshot FROM detections").fetchall()
        conn.execute("DELETE FROM detections")
    removed = sum(1 for r in rows if _remove_snapshot(r["snapshot"]))
    return {"ok": True, "removed": len(rows), "removed_files": removed}


@app.post("/api/detections/delete-batch")
async def delete_detections_batch(request: Request):
    """Remove the selected detection records and their snapshot files."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    ids = body.get("ids") if isinstance(body, dict) else None
    if not isinstance(ids, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in ids
    ):
        raise HTTPException(status_code=400, detail='Body must be {"ids": [int, ...]}')
    if not ids:
        return {"ok": True, "removed": 0, "removed_files": 0}
    marks = ",".join("?" * len(ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT snapshot FROM detections WHERE id IN ({marks})", ids
        ).fetchall()
        conn.execute(f"DELETE FROM detections WHERE id IN ({marks})", ids)
    removed = sum(1 for r in rows if _remove_snapshot(r["snapshot"]))
    return {"ok": True, "removed": len(rows), "removed_files": removed}


def _remove_snapshot(name) -> bool:
    """Delete a snapshot file from SNAPSHOT_ROOT (path-traversal safe)."""
    if not name:
        return False
    safe_root = os.path.normpath(SNAPSHOT_ROOT)
    path = os.path.abspath(os.path.join(safe_root, str(name)))
    if not path.startswith(safe_root + os.sep) or not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


@app.get("/api/stats")
def stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
        by_cat = {
            r["category"]: r["c"]
            for r in conn.execute(
                "SELECT category, COUNT(*) c FROM detections GROUP BY category"
            )
        }
        activations = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE action_taken = 1"
        ).fetchone()["c"]
        last = conn.execute(
            "SELECT timestamp, label, category FROM detections ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "total": total,
        "by_category": {
            "Harmful": by_cat.get("Harmful", 0),
            "Beneficial": by_cat.get("Beneficial", 0),
            "Harmless": by_cat.get("Harmless", 0),
        },
        "trap_activations": activations,
        "last_detection": dict(last) if last else None,
        "dashboard_uptime_seconds": int(time.time() - _start_time),
    }


@app.get("/api/pipeline/status")
def pipeline_status():
    return {
        "running": worker.running,
        "camera": hub.source_kind if hub.running else None,
        "camera_running": hub.running,
        "requested_source": hub.requested_source,
        "camera_fell_back": hub.fell_back,
        "camera_error": hub.last_error,
        "interval_seconds": config.CAPTURE_INTERVAL_SECONDS,
        "threshold": config.CONFIDENCE_THRESHOLD,
    }


@app.post("/api/camera/start")
async def camera_start(request: Request):
    """Open the shared camera for preview without starting detection."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    source = body.get("source")
    if source is not None:
        if not isinstance(source, str) or source.lower() not in {"auto", "webcam", "demo"}:
            raise HTTPException(status_code=400, detail="Unknown camera source")
        source = source.lower()

    if source is not None and hub.running and source != hub.source_kind:
        if worker.running:
            raise HTTPException(status_code=409, detail="Stop detection before changing cameras")
        hub.stop()

    try:
        camera = hub.start(source)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Could not open camera: {e}")
    return {
        "ok": True,
        "camera": camera,
        "requested_source": hub.requested_source,
        "camera_fell_back": hub.fell_back,
    }


@app.post("/api/pipeline/start")
async def pipeline_start(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    source = body.get("source")
    if source is not None:
        if not isinstance(source, str) or source.lower() not in {"auto", "webcam", "demo"}:
            raise HTTPException(status_code=400, detail="Unknown camera source")
        source = source.lower()
    try:
        started = worker.start(source=source, simulate=body.get("simulate"))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Could not open camera: {e}")
    return {"ok": True, "started": started, "camera": hub.source_kind}


@app.post("/api/pipeline/stop")
def pipeline_stop():
    worker.stop()
    return {"ok": True, "running": worker.running}


@app.get("/api/stream")
def stream():
    """MJPEG live feed from the shared camera hub (~8 fps).

    The hub is NOT auto-started here: the camera comes alive only when
    detection is started via /api/pipeline/start (or an explicit capture).
    """
    def gen():
        while True:
            frame = hub.get_frame()
            if frame is not None:
                buf = io.BytesIO()
                Image.fromarray(frame).convert("RGB").save(buf, "JPEG", quality=80)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.getvalue() + b"\r\n")
            time.sleep(0.12)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/chat")
async def chat(request: Request):
    """On-device assistant grounded in live trap data + pest knowledge."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    message = str((body or {}).get("message", "") or "")
    lang = str((body or {}).get("lang", "") or "en")

    current = stats()
    ctx = {
        "stats": current,
        "last_detection": current["last_detection"],
        "pipeline": pipeline_status(),
        "threshold": config.CONFIDENCE_THRESHOLD,
    }
    return {"reply": chatbot.reply(message, ctx, lang)}


def _analyse_frame(frame: np.ndarray, source: str) -> dict:
    """Classify a supported frame, record it, and return the AI details."""
    prediction = get_classifier().classify(frame)
    verdict = evaluate_subject(prediction)
    if not verdict.accepted:
        print(f"[{source}] Rejected {verdict.subject} frame ({verdict.reason})")
        raise HTTPException(status_code=422, detail=verdict.detail())

    would_activate = (
        prediction.confidence >= config.CONFIDENCE_THRESHOLD
        and prediction.category == CATEGORY_HARMFUL
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot = cloud_logger.save_snapshot(frame, timestamp)
    store_detection({
        "timestamp": timestamp,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "category": prediction.category,
        "action_taken": False,
        "snapshot": snapshot,
        "top": prediction.top,
        "source": source,
    })

    print(f"[{source}] label='{prediction.label}' "
          f"confidence={prediction.confidence:.2f} category={prediction.category}")
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "category": prediction.category,
        "top": prediction.top,
        "snapshot": snapshot,
        "would_activate": would_activate,
        "subject": verdict.subject,
        "threshold": config.CONFIDENCE_THRESHOLD,
    }


@app.post("/api/classify")
async def classify_upload(file: UploadFile = File(...)):
    """Analyse an uploaded photo and return the AI details."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB)")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Not a readable image file")

    img.thumbnail((1280, 1280))
    return _analyse_frame(np.asarray(img), "upload")


@app.post("/api/capture")
def capture_frame():
    """Classify the current live-camera frame without activating the trap."""
    if not hub.running:
        try:
            hub.start()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"Could not open camera: {e}")

    deadline = time.monotonic() + 2
    frame = hub.get_frame()
    while frame is None and time.monotonic() < deadline:
        time.sleep(0.05)
        frame = hub.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera did not provide a frame")
    return _analyse_frame(frame, "capture")


@app.get("/snapshots/{name}")
def snapshot(name: str):
    safe_root = os.path.normpath(SNAPSHOT_ROOT)
    path = os.path.abspath(os.path.join(safe_root, name))
    if not path.startswith(safe_root + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()


if os.path.isdir(SNAPSHOT_ROOT):
    app.mount("/files", StaticFiles(directory=SNAPSHOT_ROOT), name="files")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Pest trap dashboard server")
    parser.add_argument("--host", default=config.DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
