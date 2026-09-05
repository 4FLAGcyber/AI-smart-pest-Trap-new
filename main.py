"""
AI Smart Pest Trap — main control loop.

    01 DETECT   -> camera_capture.py
    02 CLASSIFY -> classifier.py
    03 DECIDE   -> this file
    04 ACT      -> trap_controller.py

Run with:  python main.py
Stop with: Ctrl+C
"""

import argparse
import sys
import time

# Windows consoles often default to a legacy code page; force UTF-8 so the
# log lines print cleanly everywhere.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import config
from camera_capture import get_camera, DemoImageSource
from classifier import InsectClassifier
from pest_categories import CATEGORY_HARMFUL
from subject_gate import evaluate_subject
from trap_controller import make_trap
import cloud_logger


def decide_and_act(prediction, trap) -> bool:
    """Applies the decision logic. Returns True if the trap was activated."""
    verdict = evaluate_subject(prediction)
    if not verdict.accepted:
        print(f"[decide] Rejected {verdict.subject} frame ({verdict.reason}) — no action")
        return False
    if prediction.confidence < config.CONFIDENCE_THRESHOLD:
        print(f"[decide] Low confidence ({prediction.confidence:.2f}) — no action")
        return False

    if prediction.category == CATEGORY_HARMFUL:
        trap.activate()
        return True

    print(f"[decide] {prediction.category} — no action")
    return False


def classify_single(image_path: str):
    """Classify one image file and print the result (handy for sanity checks)."""
    from PIL import Image
    import numpy as np

    frame = np.asarray(Image.open(image_path).convert("RGB"))
    classifier = InsectClassifier()
    prediction = classifier.classify(frame)

    print(f"[image] {image_path}")
    for label, score in prediction.top:
        marker = " <-" if label == prediction.label else ""
        print(f"    {score:.3f}  {label}{marker}")
    print(f"[detect] label='{prediction.label}' confidence={prediction.confidence:.2f} "
          f"category={prediction.category}")
    verdict = evaluate_subject(prediction)
    if not verdict.accepted:
        print(f"[subject] Rejected {verdict.subject} frame ({verdict.reason})")
    elif prediction.confidence >= config.CONFIDENCE_THRESHOLD and prediction.category == CATEGORY_HARMFUL:
        print("[trap] WOULD ACTIVATE — harmful pest detected")
    else:
        print("[decide] no trap action")


def run(source: str = None, simulate_trap: bool = None, once: bool = False,
        interval: float = None):
    print("Starting AI Smart Pest Trap...")
    print(f"[config] source={source or config.CAMERA_SOURCE} "
          f"threshold={config.CONFIDENCE_THRESHOLD} "
          f"cloud={'on' if config.LOG_TO_CLOUD else 'off'}")

    camera = get_camera(source)
    classifier = InsectClassifier()
    trap = make_trap(simulate_trap)

    interval = interval if interval is not None else config.CAPTURE_INTERVAL_SECONDS

    try:
        while True:
            frame = camera.capture_frame()
            prediction = classifier.classify(frame)

            origin = ""
            if isinstance(camera, DemoImageSource):
                origin = f" (demo: {camera.current_name})"
            print(
                f"[detect] label='{prediction.label}' "
                f"confidence={prediction.confidence:.2f} "
                f"category={prediction.category}{origin}"
            )

            verdict = evaluate_subject(prediction)
            if not verdict.accepted:
                print(f"[subject] Rejected {verdict.subject} frame ({verdict.reason})")
                if once:
                    break
                time.sleep(interval)
                continue

            action_taken = decide_and_act(prediction, trap)

            cloud_logger.log_detection(
                label=prediction.label,
                confidence=prediction.confidence,
                category=prediction.category,
                action_taken=action_taken,
                image=frame,
                top=prediction.top,
            )

            if once:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.close()
        trap.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Smart Pest Trap control loop")
    parser.add_argument(
        "--source", choices=["auto", "webcam", "demo"], default=None,
        help="Capture source (default: config.CAMERA_SOURCE = auto).",
    )
    parser.add_argument(
        "--image", metavar="PATH",
        help="Classify a single image file and exit.",
    )
    parser.add_argument(
        "--simulate-trap", action="store_true", default=None,
        help="Force simulated trap (print actions instead of moving the servo).",
    )
    parser.add_argument(
        "--real-trap", action="store_true",
        help="Force the real servo controller (Raspberry Pi).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single detect-classify-decide-act cycle then exit.",
    )
    parser.add_argument(
        "--interval", type=float, default=None,
        help="Seconds between captures (default: config.CAPTURE_INTERVAL_SECONDS).",
    )
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="Disable POSTing detections to the dashboard.",
    )
    args = parser.parse_args()

    if args.no_cloud:
        config.LOG_TO_CLOUD = False

    if args.image:
        classify_single(args.image)
    else:
        simulate = True if args.simulate_trap else (False if args.real_trap else None)
        run(source=args.source, simulate_trap=simulate, once=args.once,
            interval=args.interval)
