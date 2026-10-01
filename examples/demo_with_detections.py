import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from oc_sort import OCSort

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "yolo11s.pt"
VIDEO_PATH = SCRIPT_DIR / "dance_demo.mp4"
OUTPUT_PATH = SCRIPT_DIR / "dance_demo_tracked.mp4"

# 1. Load YOLO (weights download automatically on first run!)
print(f"Loading YOLO model: {MODEL_PATH.name}...")
model = YOLO(str(MODEL_PATH))

# 2. Initialize OC-SORT
tracker = OCSort(det_thresh=0.4, iou_threshold=0.3)

# 3. Process video
if not VIDEO_PATH.exists():
    sys.exit(f"Error: Video file not found at {VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    sys.exit(f"Error: Could not open video source at {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 4. Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

print(f"Processing video: {VIDEO_PATH.name} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
print(f"Saving output to: {OUTPUT_PATH}\n")

frame_idx = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Run detection (e.g. classes=[0] for person only)
    results = model(frame, classes=[0], verbose=False)

    # Extract boxes: shape (N, 6) -> [x1, y1, x2, y2, conf, class_id]
    boxes = results[0].boxes.data.cpu().numpy()

    if len(boxes) > 0:
        # OC-SORT expects (N, 5): [[x1, y1, x2, y2, score], ...]
        detections = boxes[:, :5]
    else:
        detections = np.empty((0, 5))

    # Update tracker
    tracks = tracker.update(detections)

    # Draw tracks on frame
    for trk in tracks:
        x1, y1, x2, y2, track_id = trk
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {int(track_id)}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Write frame to video
    out.write(frame)

    if frame_idx % 50 == 0 or frame_idx == total_frames:
        elapsed = time.time() - start_time
        fps_proc = frame_idx / elapsed if elapsed > 0 else 0
        pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        print(f"  Frame [{frame_idx}/{total_frames}] ({pct:5.1f}%) | Speed: {fps_proc:5.1f} FPS | Active Tracks: {len(tracks)}")

cap.release()
out.release()

total_time = time.time() - start_time
avg_fps = frame_idx / total_time if total_time > 0 else 0
print(f"\nProcessing complete in {total_time:.2f}s ({avg_fps:.1f} FPS average)!")
print(f"Tracked video saved successfully to: {OUTPUT_PATH}")
