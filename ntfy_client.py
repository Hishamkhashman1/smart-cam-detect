import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2

from config import (
    NTFY_BASE_URL,
    NTFY_PRIORITY,
    NTFY_SEND_SCREENSHOT,
    NTFY_TAGS,
    NTFY_TIMEOUT_SECONDS,
    NTFY_TOPIC,
    NTFY_TITLE,
)


def parse_priority(priority_value):
    try:
        return int(priority_value)
    except ValueError:
        return priority_value


def send_ntfy_notification(message, frame=None):
    if not NTFY_TOPIC:
        return

    headers = {
        "Title": NTFY_TITLE,
        "Priority": NTFY_PRIORITY,
        "Message": message,
    }

    if NTFY_TAGS:
        headers["Tags"] = ",".join(NTFY_TAGS)

    if frame is not None and NTFY_SEND_SCREENSHOT:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if ok:
            headers["Content-Type"] = "image/jpeg"
            headers["Filename"] = "snapshot.jpg"
            request = Request(
                f"{NTFY_BASE_URL}/{NTFY_TOPIC}",
                data=encoded.tobytes(),
                headers=headers,
                method="PUT",
            )
            try:
                with urlopen(request, timeout=NTFY_TIMEOUT_SECONDS) as response:
                    response.read()
                print(f"ntfy alert sent: {message}")
                return
            except (HTTPError, URLError, OSError) as exc:
                print(
                    f"ntfy screenshot alert failed, falling back to text-only: {exc}"
                )

    payload = {
        "topic": NTFY_TOPIC,
        "message": message,
        "title": NTFY_TITLE,
        "priority": parse_priority(NTFY_PRIORITY),
    }
    if NTFY_TAGS:
        payload["tags"] = NTFY_TAGS

    request = Request(
        f"{NTFY_BASE_URL}/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=NTFY_TIMEOUT_SECONDS) as response:
            response.read()
        print(f"ntfy text alert sent: {message}")
    except (HTTPError, URLError, OSError) as exc:
        print(f"Failed to send ntfy alert: {exc}")
