import cv2
import os
from collections import deque
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")

if not RTSP_URL:
    raise ValueError("RTSP_URL not found in .env")

# Better than yolov8n for your use case
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream")

# =========================
# SETTINGS
# =========================
FRAME_SKIP = 1
DEBUG = True

# Per-class thresholds
CONF_THRESHOLDS = {
    "person": 0.35,
    "car": 0.45,
    "dog": 0.35,
    "cat": 0.35,
}

# Small distant objects should still pass
MIN_SIZE = {
    "person": (10, 20),
    "car": (20, 20),
    "dog": (10, 10),
    "cat": (10, 10),
}

TARGET_CLASSES = {"person", "car", "dog", "cat"}

# Require detection in X of the last N processed frames
history = deque(maxlen=5)
CONFIRM_FRAMES = 2

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Optional resize for speed; comment this out if you want full-size inference
    # frame = cv2.resize(frame, None, fx=0.8, fy=0.8)

    results = model(frame, verbose=False)

    current_frame_hits = 0
    frame_height, frame_width = frame.shape[:2]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            # Print everything YOLO sees for the classes you care about
            if DEBUG and label in TARGET_CLASSES:
                print(f"{label:6} conf={conf:.2f} w={w:3} h={h:3} x1={x1:4} y1={y1:4}")

            # 1) keep only wanted classes
            if label not in TARGET_CLASSES:
                continue

            # 2) class-specific confidence threshold
            if conf < CONF_THRESHOLDS[label]:
                continue

            # 3) class-specific size threshold
            min_w, min_h = MIN_SIZE[label]
            if w < min_w or h < min_h:
                continue

            # 4) looser zone filter
            # Only ignore boxes that are extremely high in the frame.
            # Your previous 20% was too aggressive for distant people.
            if y1 < frame_height * 0.05:
                continue

            current_frame_hits += 1

            # Draw accepted detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Multi-frame confirmation
    history.append(current_frame_hits > 0)

    if sum(history) >= CONFIRM_FRAMES:
        cv2.putText(
            frame,
            "CONFIRMED DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )
        print("CONFIRMED detection")

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
