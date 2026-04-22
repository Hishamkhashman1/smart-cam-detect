from collections import deque

import cv2
from ultralytics import YOLO

from config import (
    CONFIRM_FRAMES,
    CONF_THRESHOLDS,
    DEBUG,
    FRAME_SKIP,
    HEADLESS,
    MIN_SIZE,
    RTSP_URL,
    TARGET_CLASSES,
)
from ntfy_client import send_ntfy_notification

# Better than yolov8n for your use case
model = YOLO("yolov8s.pt")


def build_detection_summary(labels):
    unique_labels = sorted(set(labels))
    if not unique_labels:
        return "Motion detected"
    return f"Detected: {', '.join(unique_labels)}"


def run_detection():
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream")

    history = deque(maxlen=5)
    alert_latched = False
    last_detection_frame = None
    last_detection_summary = "Motion detected"
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
        current_frame_labels = []
        frame_height = frame.shape[0]

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
                    print(
                        f"{label:6} conf={conf:.2f} w={w:3} h={h:3} x1={x1:4} y1={y1:4}"
                    )

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
                current_frame_labels.append(label)

                # Draw accepted detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if current_frame_hits > 0:
            last_detection_frame = frame.copy()
            last_detection_summary = build_detection_summary(current_frame_labels)

        # Multi-frame confirmation
        history.append(current_frame_hits > 0)
        confirmed = sum(history) >= CONFIRM_FRAMES

        if confirmed:
            cv2.putText(
                frame,
                "CONFIRMED DETECTION",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            if not alert_latched and last_detection_frame is not None:
                print(f"CONFIRMED detection: {last_detection_summary}")
                send_ntfy_notification(last_detection_summary, last_detection_frame)
                alert_latched = True
        else:
            alert_latched = False

        if not HEADLESS:
            cv2.imshow("Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
