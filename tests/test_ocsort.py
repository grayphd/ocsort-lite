import unittest

import numpy as np

from oc_sort import (
    OCSort,
    Timer,
    ciou_batch,
    ct_dist,
    diou_batch,
    giou_batch,
    iou_batch,
    linear_assignment,
)


class TestOCSort(unittest.TestCase):
    def test_iou_and_distance_metrics(self):
        b1 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
        b2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])

        iou = iou_batch(b1, b2)
        self.assertEqual(iou.shape, (2, 2))
        self.assertTrue(np.isclose(iou[0, 0], 1.0))

        giou = giou_batch(b1, b2)
        self.assertEqual(giou.shape, (2, 2))

        diou = diou_batch(b1, b2)
        self.assertEqual(diou.shape, (2, 2))

        ciou = ciou_batch(b1, b2)
        self.assertEqual(ciou.shape, (2, 2))

        dist = ct_dist(b1, b2)
        self.assertEqual(dist.shape, (2, 2))

    def test_linear_assignment(self):
        cost = np.array([[10, 2, 5], [3, 8, 9], [6, 7, 1]])
        matches = linear_assignment(cost)
        self.assertEqual(len(matches), 3)

        empty_matches = linear_assignment(np.empty((0, 0)))
        self.assertEqual(len(empty_matches), 0)

    def test_tracker_lifecycle_numpy(self):
        tracker = OCSort(det_thresh=0.4, max_age=5, min_hits=2)

        # Frame 1: 2 detections
        f1 = np.array([[10, 10, 50, 50, 0.9], [100, 100, 150, 150, 0.85]])
        res1 = tracker.update(f1)
        self.assertIsInstance(res1, np.ndarray)

        # Frame 2: same boxes moved slightly
        f2 = np.array([[11, 11, 51, 51, 0.92], [101, 101, 151, 151, 0.88]])
        res2 = tracker.update(f2)
        self.assertEqual(len(res2), 2)
        ids2 = sorted(res2[:, 4].tolist())
        self.assertEqual(len(ids2), 2)

        # Frame 3: empty detections (lost frame)
        res3 = tracker.update(np.empty((0, 5)))
        self.assertIsInstance(res3, np.ndarray)

        # Frame 4: object reappears (hit streak becomes 1)
        f4 = np.array([[13, 13, 53, 53, 0.9]])
        res4 = tracker.update(f4)
        self.assertIsInstance(res4, np.ndarray)

        # Frame 5: object continues (hit streak becomes 2 >= min_hits)
        f5 = np.array([[14, 14, 54, 54, 0.91]])
        res5 = tracker.update(f5)
        self.assertGreaterEqual(len(res5), 1)

    def test_tracker_with_optional_rescaling(self):
        tracker = OCSort(det_thresh=0.3)
        dets = np.array([
            [20.0, 30.0, 70.0, 80.0, 0.9],
            [120.0, 130.0, 170.0, 180.0, 0.8]
        ])
        res = tracker.update(dets, img_info=(640, 640), img_size=(640, 640))
        self.assertIsInstance(res, np.ndarray)

    def test_timer(self):
        t = Timer()
        t.tic()
        dur = t.toc()
        self.assertGreaterEqual(dur, 0)


if __name__ == "__main__":
    unittest.main()
