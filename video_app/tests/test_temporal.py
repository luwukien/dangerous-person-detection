import unittest

import numpy as np
from ultralytics.engine.results import Results

from video_app.temporal import TemporalLabels, VideoTracker


class TemporalTests(unittest.TestCase):
    def test_waits_for_evidence_and_rejects_one_frame_flip(self):
        labels = TemporalLabels(30)
        self.assertIsNone(labels.update(1, 0, .9, 1))
        for frame in range(2, 20):
            label = labels.update(1, 0, .9, frame)
        self.assertEqual(label, 0)
        self.assertEqual(labels.update(1, 1, .99, 20), 0)

    def test_sustained_changes_work_in_both_directions_at_multiple_fps(self):
        for fps in (10, 30, 60):
            with self.subTest(fps=fps):
                labels = TemporalLabels(fps)
                frame = 0
                for cls in (0, 1, 0):
                    for _ in range(fps):
                        frame += 1
                        label = labels.update(1, cls, .8, frame)
                    self.assertEqual(label, cls)

    def test_alternating_evidence_does_not_flip_confirmed_label(self):
        labels = TemporalLabels(30)
        for frame in range(1, 31):
            labels.update(1, 0, .8, frame)
        for frame in range(31, 100):
            self.assertEqual(labels.update(1, frame % 2, .8, frame), 0)

    def test_ids_have_independent_evidence(self):
        labels = TemporalLabels(30)
        for frame in range(1, 31):
            a = labels.update(1, 0, .8, frame)
            b = labels.update(2, 1, .8, frame)
        self.assertEqual((a, b), (0, 1))

    def test_long_absence_expires_label(self):
        labels = TemporalLabels(30)
        for frame in range(1, 31):
            labels.update(1, 1, .9, frame)
        labels.expire(46)
        self.assertNotIn(1, labels.states)
        self.assertIsNone(labels.update(1, 0, .9, 47))

    def test_short_gap_does_not_accumulate_stale_votes(self):
        labels = TemporalLabels(30)
        for frame in range(1, 6):
            labels.update(1, 1, .9, frame)
        self.assertIsNone(labels.update(1, 1, .9, 8))

    def test_invalid_parameters(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TemporalLabels(fps)

    def test_real_tracker_keeps_id_across_class_changes_and_handles_empty_frames(self):
        tracker = VideoTracker(30)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        names = {0: "normal_person", 1: "potentially_dangerous_person"}
        ids = set()
        for cls in [0] * 20 + [1] * 30 + [0] * 30:
            result = Results(frame, path="test", names=names,
                             boxes=np.array([[10, 10, 50, 90, .9, cls]], dtype=np.float32))
            original = result.boxes.data.copy()
            self.assertEqual(tracker.annotate(result).shape, frame.shape)
            np.testing.assert_array_equal(result.boxes.data, original)
            ids.add(tracker.observations[0][1])
        self.assertEqual(len(ids), 1)
        self.assertEqual(tracker.raw_switches, 2)
        self.assertEqual(tracker.stable_switches, 2)
        empty = Results(frame, path="test", names=names, boxes=np.empty((0, 6), dtype=np.float32))
        for _ in range(16):
            np.testing.assert_array_equal(tracker.annotate(empty), frame)
        self.assertEqual(tracker.labels.states, {})
        self.assertEqual(VideoTracker(30).labels.states, {})


if __name__ == "__main__":
    unittest.main()
