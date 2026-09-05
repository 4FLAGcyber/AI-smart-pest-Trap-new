"""
Capture sources for the AI Smart Pest Trap.

    PiCamera        Raspberry Pi camera module (picamera2), Pi only
    OpenCVCamera    USB webcam, works on any OS
    DemoImageSource cycles through bundled insect photos (demo_images/)

`get_camera(source)` picks among them; "auto" prefers a real camera and
falls back to the demo folder so the pipeline always runs.
"""

import os
import sys

import numpy as np
from PIL import Image

import config

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PiCamera:
    """Wraps picamera2 for simple 'grab one frame' usage."""

    def __init__(self, resolution=None):
        from picamera2 import Picamera2  # imported here so this file can
                                          # still be read/edited off-Pi
        import time

        self.resolution = resolution or config.CAMERA_RESOLUTION
        self.picam2 = Picamera2()
        cam_config = self.picam2.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self.picam2.configure(cam_config)
        self.picam2.start()
        time.sleep(2)  # let auto-exposure/white-balance settle

    def capture_frame(self) -> np.ndarray:
        """Returns an HxWx3 RGB numpy array."""
        return self.picam2.capture_array()

    def close(self):
        self.picam2.stop()


class OpenCVCamera:
    """USB webcam source."""

    def __init__(self, resolution=None, device_index=None):
        import cv2

        self.cv2 = cv2
        self.resolution = resolution or config.CAMERA_RESOLUTION
        indexes = (device_index,) if device_index is not None else range(3)
        backends = ((None, "default"), (cv2.CAP_DSHOW, "DirectShow"))
        failures = []

        for index in indexes:
            for backend, backend_name in backends:
                cap = (
                    cv2.VideoCapture(index)
                    if backend is None
                    else cv2.VideoCapture(index, backend)
                )
                if not cap.isOpened():
                    cap.release()
                    failures.append(f"index {index} ({backend_name}) could not be opened")
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                ok, _ = cap.read()
                if not ok:
                    cap.release()
                    failures.append(f"index {index} ({backend_name}) returned no frames")
                    continue
                self.cap = cap
                self.device_index = index
                return

        detail = "; ".join(failures)
        raise RuntimeError(f"No usable webcam found ({detail})")

    def capture_frame(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam")
        return self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class DemoImageSource:
    """Cycles through the insect photos in DEMO_IMAGE_DIR.

    Lets the full detect -> classify -> decide -> act pipeline run anywhere,
    even without a camera.
    """

    def __init__(self, image_dir=None):
        self.image_dir = image_dir or config.DEMO_IMAGE_DIR
        if not os.path.isdir(self.image_dir):
            raise RuntimeError(f"Demo image folder not found: {self.image_dir}")
        self.images = sorted(
            f for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self._index = 0
        self._current = self.images[0]

    @property
    def current_name(self) -> str:
        """Filename of the frame returned by the last capture_frame() call."""
        return self._current

    def capture_frame(self) -> np.ndarray:
        path = os.path.join(self.image_dir, self.images[self._index])
        self._current = self.images[self._index]
        self._index = (self._index + 1) % len(self.images)
        img = Image.open(path).convert("RGB")
        img.thumbnail((800, 800))
        return np.asarray(img)

    def close(self):
        pass


def _on_raspberry_pi() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def get_camera(source=None):
    """Factory for the configured capture source.

    source: "auto" | "webcam" | "demo" (default: config.CAMERA_SOURCE)
    """
    source = (source or config.CAMERA_SOURCE).lower()

    if source == "demo":
        return DemoImageSource()

    if source == "webcam":
        return OpenCVCamera()

    # auto: Pi camera -> webcam -> demo images
    if _on_raspberry_pi():
        try:
            return PiCamera()
        except Exception as e:
            print(f"[camera] Pi camera unavailable ({e}); trying webcam.")

    try:
        return OpenCVCamera()
    except Exception as e:
        print(f"[camera] No webcam available ({e}); using demo images.")
        return DemoImageSource()
