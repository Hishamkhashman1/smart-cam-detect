# smart-cam-detect

Run the camera detector against an RTSP stream and send confirmed alerts to ntfy.

Environment variables:

- `RTSP_URL`: RTSP camera stream URL
- `NTFY_TOPIC`: ntfy topic to publish to
- `NTFY_URL`: ntfy server base URL, defaults to `https://ntfy.sh`
- `NTFY_TITLE`: notification title
- `NTFY_PRIORITY`: ntfy priority, defaults to `4`
- `NTFY_TAGS`: comma-separated ntfy tags
- `NTFY_SEND_SCREENSHOT`: set to `0` to disable screenshot attachments

The script confirms detections across multiple frames, draws boxes on the frame, and publishes the annotated screenshot to ntfy when the alert triggers.
