"""
Central configuration for the AI Smart Pest Trap app.
Edit these values to match your wiring, camera, and desired behaviour.
"""

# --- Capture source ---
# "auto"   -> webcam if one is available, otherwise demo image folder
# "webcam" -> force USB webcam (OpenCV)
# "demo"   -> cycle through the insect photos in DEMO_IMAGE_DIR
# On a Raspberry Pi the app additionally tries the Pi camera module first.
CAMERA_SOURCE = "auto"
DEMO_IMAGE_DIR = "demo_images"
CAMERA_RESOLUTION = (640, 480)
CAPTURE_INTERVAL_SECONDS = 4       # how often we grab a frame and classify it

# --- GPIO / Servo (Raspberry Pi only) ---
SERVO_PIN = 17            # BCM pin the servo signal wire is connected to
TRAP_OPEN_ANGLE = 90      # servo angle that "activates" the trap
TRAP_CLOSED_ANGLE = 0     # resting / idle angle
TRAP_ACTIVE_SECONDS = 2   # how long the trap stays activated before resetting

# --- AI model ---
MODEL_PATH = "models/mobilenet_v2.tflite"
LABELS_PATH = "models/labels.txt"
CONFIDENCE_THRESHOLD = 0.35      # below this -> treated as uncertain (no action)
TOP_K = 3
SUBJECT_TOP_K = 12
SUBJECT_MIN_TOP_CONFIDENCE = 0.20
SUBJECT_MIN_INSECT_MASS = 0.25
SUBJECT_MIN_ANIMAL_MASS = 0.50
SUBJECT_HUMAN_EVIDENCE = 0.12
SUBJECT_OBJECT_EVIDENCE = 0.40

# --- Logging / dashboard ---
LOG_TO_CLOUD = True              # POST every detection to the dashboard endpoint
CLOUD_ENDPOINT_URL = "http://127.0.0.1:8000/api/detections"
LOCAL_LOG_FILE = "logs/detections.csv"
SNAPSHOT_DIR = "logs/snapshots"  # captured frames kept next to the CSV log
MAX_SNAPSHOTS = 300              # oldest snapshots are pruned beyond this

# --- Dashboard server ---
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8000
