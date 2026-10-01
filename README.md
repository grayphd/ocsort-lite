# OC-SORT (Observation-Centric SORT) - Lightweight & Standalone

This is a standalone, lightweight Python package for the **Observation-Centric SORT (OC-SORT)** multi-object tracking algorithm.

It has been decoupled from external detector code (YOLOX), heavy training pipelines, and CUDA C++ extensions. It can be installed as a pure tracking library in any computer vision pipeline.

## Features
- **Pure Tracking**: Only requires NumPy, SciPy, FilterPy, and LAPx.
- **Fast Linear Assignment**: Uses `lapx` (with automatic fallback to `scipy.optimize.linear_sum_assignment`).
- **Flexible Input**: Works with raw NumPy bounding boxes (`[x1, y1, x2, y2, score]`) or detector output tensors.
- **Full OC-SORT Functionality**: Momentum-aware Kalman filter (`KalmanFilterNew`), Observation-Centric Re-Update (ORU), and Observation-Centric Recovery (OCR).

## Installation

```bash
# Clone or navigate to the repository
cd ocsort-lite

# Install in editable mode
pip install -e .
# or using uv:
uv pip install -e .
```

## Quickstart

```python
import numpy as np
from oc_sort import OCSort

# Initialize tracker
tracker = OCSort(
    det_thresh=0.4,       # Confidence threshold for detections
    max_age=30,           # Maximum frames to keep a lost track
    min_hits=3,           # Minimum hits before confirming a track
    iou_threshold=0.3,    # IoU association threshold
    delta_t=3,            # Momentum estimation window
    asso_func="iou",      # Association metric: "iou", "giou", "diou", "ciou", or "ct_dist"
    inertia=0.2,          # Weight of velocity direction consistency
    use_byte=False        # Whether to enable BYTE secondary matching
)

# For each video frame, pass detections as a numpy array of shape (N, 5):
# [[x1, y1, x2, y2, confidence], ...]
detections = np.array([
    [100, 150, 200, 300, 0.92],
    [400, 200, 480, 350, 0.85]
])

# Update tracker
active_tracks = tracker.update(detections)

# Returns numpy array of shape (M, 5):
# [[x1, y1, x2, y2, track_id], ...]
for trk in active_tracks:
    x1, y1, x2, y2, track_id = trk
    print(f"Track {int(track_id)}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
```

## Running Tests

```bash
python -m unittest discover tests
```

## Running the Demo

```bash
python examples/standalone_demo.py
```
