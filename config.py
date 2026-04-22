import os

from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")

if not RTSP_URL:
    raise ValueError("RTSP_URL not found in .env")

NTFY_BASE_URL = os.getenv("NTFY_URL", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.getenv("NTFY_TOPIC")
NTFY_TITLE = os.getenv("NTFY_TITLE", "Smart cam detection")
NTFY_PRIORITY = os.getenv("NTFY_PRIORITY", "4")
NTFY_TAGS = [
    tag.strip()
    for tag in os.getenv("NTFY_TAGS", "camera,warning").split(",")
    if tag.strip()
]
NTFY_TIMEOUT_SECONDS = float(os.getenv("NTFY_TIMEOUT_SECONDS", "10"))
NTFY_SEND_SCREENSHOT = os.getenv("NTFY_SEND_SCREENSHOT", "1").lower() not in {
    "0",
    "false",
    "no",
}
HEADLESS = os.getenv("HEADLESS", "0").strip() == "1"

if not NTFY_TOPIC:
    print("NTFY_TOPIC not set; ntfy notifications are disabled")

FRAME_SKIP = 1
DEBUG = True

CONF_THRESHOLDS = {
    "person": 0.35,
    "car": 0.45,
    "dog": 0.35,
    "cat": 0.35,
}

MIN_SIZE = {
    "person": (10, 20),
    "car": (20, 20),
    "dog": (10, 10),
    "cat": (10, 10),
}

TARGET_CLASSES = {"person", "car", "dog", "cat"}

CONFIRM_FRAMES = 2
