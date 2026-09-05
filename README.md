# AI Smart Pest Trap — Full App

Detect → Classify → Decide → Act. A camera captures an insect, a MobileNetV2
model predicts what it is, the controller maps that to Harmful / Beneficial /
Harmless, and the trap responds **only to harmful pests** — while a live web
dashboard records and visualizes every detection.

This build runs end-to-end on a normal PC (webcam or bundled demo images,
simulated servo) and on a Raspberry Pi (Pi camera + real servo) with no code
changes.

## Quick start (this PC)

```bash
pip install -r requirements.txt
python run.py
```

That single command:

1. starts the dashboard at **http://127.0.0.1:8000** (open it in a browser),
2. starts the detection pipeline on the live feed — webcam if present,
   otherwise the simulated demo feed,
3. feeds every detection (with a snapshot) into the dashboard and into
   `logs/detections.csv`.

The dashboard is fully interactive:

- **Live camera** — an MJPEG stream of the webcam (or simulated feed). Press
  **Start detection** to run detect→classify→decide→act on it, **Stop** to pause.
- **Analyse a photo** — drop any insect photo onto the panel; the AI returns
  the label, category, confidence and top-3 guesses, and tells you whether the
  trap would activate.
- **Stats & history** — live counters, category distribution, and a table of
  every detection (uploads are tagged `upload`).

Useful variations:

```bash
python run.py --source demo --once      # one rehearsed cycle
python run.py --source webcam           # force the USB webcam
python run.py --interval 2              # snappier live demo
python main.py --image path/to/photo.jpg  # classify a single photo
python main.py --no-cloud               # pipeline only, no dashboard POSTs
python dashboard.py                     # dashboard server only
```

On a Raspberry Pi with the servo wired up:

```bash
python main.py --real-trap
```

## Layout

```
├── config.py            # pins, thresholds, paths — edit this first
├── pest_categories.py   # ImageNet label -> Harmful/Beneficial/Harmless map
├── camera_capture.py    # Pi camera / USB webcam / demo image folder
├── live_camera.py       # shared camera hub (single owner) + MJPEG feed
├── classifier.py        # TFLite inference (auto-adapts to the model)
├── trap_controller.py   # servo via gpiozero + simulated version
├── cloud_logger.py      # CSV logging + snapshots + POST to dashboard
├── dashboard.py         # FastAPI "cloud" backend (SQLite) + UI at /
├── dashboard/index.html # live dashboard UI
├── run.py               # one-command launcher (dashboard + pipeline)
├── main.py              # pipeline entry point
├── models/              # mobilenet_v2.tflite + labels.txt
├── demo_images/         # real insect photos for demo mode
└── logs/                # detections.csv, snapshots/, dashboard.db
```

## The AI model

`models/mobilenet_v2.tflite` is a pretrained MobileNetV2 (ImageNet) used as a
stand-in classifier; `pest_categories.py` maps its ~30 insect labels onto the
three trap categories. The classifier auto-detects input layout (NHWC/NCHW),
dtype (uint8/float32) and output format (scores/logits), so when you train a
real pest model (e.g. Teachable Machine, export TFLite + labels.txt into
`models/`), it drops in with no code changes — update `MODEL_PATH` /
`LABELS_PATH` in `config.py`.

## Tuning

- `CONFIDENCE_THRESHOLD` (config.py): below this, no action. Raise to avoid
  false triggers, lower for a livelier demo.
- `CAPTURE_INTERVAL_SECONDS` / `--interval`: polling rate.
- `TRAP_ACTIVE_SECONDS`: how long the servo holds the open position.

## Wiring the real trap (Raspberry Pi)

Servo signal → BCM 17 (`SERVO_PIN`), power from an external 5V supply
(not the Pi's 5V pin), shared ground. Camera via CSI or USB.

```bash
pip install gpiozero picamera2   # or: sudo apt install python3-picamera2
python run.py --real-trap
```
