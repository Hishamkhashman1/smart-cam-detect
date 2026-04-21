import cv2
import os
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")

if not RTSP_URL:
    raise ValueError("RTSP_URL not found in .env")

# Load model
model = YOLO("yolov8n.pt")

# Open stream
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        continue

    # Run detection
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Only keep relevant objects
            if label not in ["person", "car", "dog", "cat"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
