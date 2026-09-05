"""
Shared camera access for the dashboard.

A single background thread owns the capture device (webcam, Pi camera, or
the demo-image folder as a simulated feed) and publishes the latest frame.
The MJPEG stream and the detection worker both read from the hub, so the
camera is only ever opened once per process.
"""

import threading
import time

import config
from camera_capture import DemoImageSource, OpenCVCamera, PiCamera, _on_raspberry_pi


class CameraHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._source = None
        self._thread = None
        self._stop = threading.Event()
        self.source_kind = None   # "webcam" | "picam" | "demo"
        self.requested_source = None
        self.fell_back = False
        self.last_error = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, source=None) -> str:
        """Start the capture thread; returns the active source kind."""
        with self._lock:
            if self.running:
                return self.source_kind
            source = (source or config.CAMERA_SOURCE).lower()
            self.requested_source = source
            self.fell_back = False
            self.last_error = None
            self._stop.clear()

            if source == "demo":
                self._source = DemoImageSource()
                self.source_kind = "demo"
            elif source == "webcam":
                try:
                    self._source = OpenCVCamera()
                except Exception as e:
                    self.last_error = str(e)
                    self.source_kind = None
                    raise
                self.source_kind = "webcam"
            elif source == "auto" and _on_raspberry_pi():
                try:
                    self._source = PiCamera()
                    self.source_kind = "picam"
                except Exception as e:
                    print(f"[camera] Pi camera unavailable ({e}); trying webcam.")
                    self._source = self._webcam_or_demo()
            elif source == "auto":
                self._source = self._webcam_or_demo()
            else:
                self.last_error = f"Unknown camera source: {source}"
                raise ValueError(self.last_error)

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return self.source_kind

    def _webcam_or_demo(self):
        try:
            source = OpenCVCamera()
        except Exception as e:
            self.last_error = str(e)
            self.fell_back = True
            print(f"[camera] No webcam available ({e}); using simulated demo feed.")
            source = DemoImageSource()
            self.source_kind = "demo"
            return source
        self.source_kind = "webcam"
        return source

    def _loop(self):
        # Real cameras are read as fast as they deliver; the demo source
        # changes scene every 1.5 s to behave like a live feed.
        pace = 1.5 if self.source_kind == "demo" else 0.05
        while not self._stop.is_set():
            try:
                frame = self._source.capture_frame()
            except Exception as e:
                print(f"[camera] capture error: {e}")
                time.sleep(1)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(pace)

    def get_frame(self):
        """Latest frame (HxWx3 RGB) or None before the first capture."""
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            if self._source is not None:
                try:
                    self._source.close()
                except Exception:
                    pass
            self._source = None
            self._frame = None
            self.source_kind = None
            self.requested_source = None
            self.fell_back = False
            self.last_error = None
        self._thread = None
