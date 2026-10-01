#!/usr/bin/env python3
"""Standalone demonstration of OC-SORT tracking without any detector dependencies."""

import numpy as np

from oc_sort import OCSort, Timer


def main():
    print("=" * 60)
    print("  OC-SORT: Standalone Multi-Object Tracker Demo")
    print("=" * 60)

    # Initialize tracker
    tracker = OCSort(
        det_thresh=0.4,
        max_age=10,
        min_hits=2,
        iou_threshold=0.3,
        delta_t=3,
        inertia=0.2,
    )
    timer = Timer()

    # Simulate 6 frames of synthetic detections for 2 targets
    # Target 1 starts at (20, 20), Target 2 starts at (150, 150)
    simulated_frames = [
        # Frame 1: Initial appearance of both targets
        np.array([
            [20.0, 20.0, 60.0, 60.0, 0.95],
            [150.0, 150.0, 210.0, 210.0, 0.91],
        ]),
        # Frame 2: Smooth motion (+5px)
        np.array([
            [25.0, 24.0, 65.0, 64.0, 0.94],
            [155.0, 153.0, 215.0, 213.0, 0.89],
        ]),
        # Frame 3: Target 1 continues, Target 2 is occluded (missing detection)
        np.array([
            [30.0, 29.0, 70.0, 69.0, 0.96],
        ]),
        # Frame 4: Target 1 continues, Target 2 still occluded
        np.array([
            [35.0, 33.0, 75.0, 73.0, 0.93],
        ]),
        # Frame 5: Target 2 reappears (OCR recovers it)
        np.array([
            [40.0, 38.0, 80.0, 78.0, 0.95],
            [170.0, 163.0, 230.0, 223.0, 0.88],
        ]),
        # Frame 6: Both targets continue moving
        np.array([
            [45.0, 42.0, 85.0, 82.0, 0.92],
            [175.0, 168.0, 235.0, 228.0, 0.90],
        ]),
    ]

    for frame_id, detections in enumerate(simulated_frames, start=1):
        timer.tic()
        tracks = tracker.update(detections)
        elapsed_ms = timer.toc() * 1000

        print(f"\n--- Frame {frame_id} (Detections: {len(detections)}) [Latency: {elapsed_ms:.2f}ms] ---")
        if len(tracks) == 0:
            print("  No confirmed tracks yet (building streak...)")
        for trk in tracks:
            x1, y1, x2, y2, track_id = trk[:5]
            print(f"  [Track #{int(track_id)}] bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    print("\n" + "=" * 60)
    print("  Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
