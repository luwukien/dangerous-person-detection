import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_processing as video


class FakeResult:
    def __init__(self, frame): self.frame = frame
    def plot(self): return self.frame


class FakeModel:
    def predict(self, frame, **kwargs):
        assert kwargs["device"] == "cpu"
        return [FakeResult(frame)]


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self): self.tmp.cleanup()

    def make_video(self, frames=3, fps=5):
        path = self.root / "input.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(frames): writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        return path

    def test_playable_output(self):
        source = self.make_video()
        output = self.root / "result.mp4"
        progress = []
        self.assertEqual(video.process_video(source, output, FakeModel(), "cpu", lambda n, **_: progress.append(n)), str(output))
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertTrue(capture.isOpened())
            self.assertTrue(capture.read()[0])
        finally: capture.release()
        self.assertTrue(progress)

    def test_bad_and_large_inputs(self):
        with self.assertRaises(ValueError): video.validate_video(None)
        bad = self.root / "bad.mp4"
        bad.write_bytes(b"invalid")
        with self.assertRaises(ValueError): video.validate_video(bad)
        with patch.object(video, "MAX_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "20 MB"): video.validate_video(bad)
        with self.assertRaises(ValueError): video.validate_video(self.root / "missing.mov")

    def test_long_video(self):
        source = self.make_video(frames=76)
        with self.assertRaisesRegex(ValueError, "15 giây"): video.validate_video(source)

    def test_duration_also_checked_while_decoding(self):
        source = self.make_video(frames=6)
        with patch.object(video, "MAX_SECONDS", 1), patch.object(video, "validate_video", return_value=(5, 5, 64, 64)):
            # Allowance of one frame guards against floating point drift.
            source = self.make_video(frames=7)
            with self.assertRaisesRegex(ValueError, "15 giây"):
                video.process_video(source, self.root / "out.mp4", FakeModel(), "cpu")


if __name__ == "__main__": unittest.main()
