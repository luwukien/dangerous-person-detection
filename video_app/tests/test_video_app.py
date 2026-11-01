import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import imageio_ffmpeg
import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from video_app import app as demo


class DeferredExecutor:
    def submit(self, fn, job):
        self.job = job


class FakeResult:
    def __init__(self, frame): self.frame = frame
    def plot(self): return self.frame


class FakeModel:
    task = 'detect'
    names = demo.EXPECTED
    def predict(self, frame, **kwargs): return [FakeResult(frame)]


class VideoAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(demo, 'workspace', self.tmp), patch.object(demo, 'executor', DeferredExecutor()), patch.object(demo, 'model', FakeModel())]
        for item in self.patches: item.start()
        demo.jobs.clear()

    def tearDown(self):
        demo.jobs.clear()
        for item in reversed(self.patches): item.stop()
        self.tmp.cleanup()

    def upload(self, name, data):
        return demo.create_job(UploadFile(filename=name, file=io.BytesIO(data)))['id']

    def test_formats_invalid_upload_and_missing_model(self):
        for ext in ('mp4', 'mov', 'avi', 'mkv'):
            self.assertEqual(demo.status(self.upload('clip.' + ext, b'x'))['status'], 'queued')
        with self.assertRaises(HTTPException) as exc:
            self.upload('x.txt', b'x')
        self.assertEqual(exc.exception.status_code, 400)
        with patch.object(demo, 'MAX_BYTES', 2):
            with self.assertRaises(HTTPException) as exc: self.upload('x.mp4', b'abc')
            self.assertEqual(exc.exception.status_code, 413)
        with patch.object(demo, 'model', None):
            with self.assertRaises(HTTPException) as exc: self.upload('x.mp4', b'x')
            self.assertEqual(exc.exception.status_code, 503)

    def test_invalid_video_and_unknown_job(self):
        job = self.upload('x.mp4', b'not video')
        demo.process(job)
        self.assertEqual(demo.status(job)['status'], 'failed')
        with self.assertRaises(HTTPException) as exc: demo.status('unknown')
        self.assertEqual(exc.exception.status_code, 404)

    def test_http_upload_status_and_video(self):
        import time
        clip = Path(self.tmp.name) / 'http.avi'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        with patch('ultralytics.YOLO', return_value=FakeModel()):
            with TestClient(demo.app) as client:
                self.assertEqual(client.get('/').status_code, 200)
                response = client.post('/api/jobs', files={'video': ('clip.avi', clip.read_bytes(), 'video/x-msvideo')})
                self.assertEqual(response.status_code, 202)
                job = response.json()['id']
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    state = client.get('/api/jobs/' + job).json()
                    if state['status'] in ('completed', 'failed'): break
                    time.sleep(.1)
                self.assertEqual(state['status'], 'completed', state)
                response = client.get('/api/jobs/' + job + '/video', headers={'range': 'bytes=0-15'})
                self.assertEqual(response.status_code, 206)
                self.assertEqual(len(response.content), 16)

    def test_processing_creates_playable_mp4_and_removes_input(self):
        clip = Path(self.tmp.name) / 'source.avi'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
        self.assertTrue(writer.isOpened())
        for i in range(3): writer.write(np.full((64, 64, 3), i * 50, dtype=np.uint8))
        writer.release()
        job = self.upload('x.avi', clip.read_bytes())
        demo.process(job)
        self.assertEqual(demo.status(job)['status'], 'completed')
        self.assertEqual(demo.status(job)['progress'], 100)
        path = Path(demo.video(job).path)
        self.assertTrue(path.exists())
        cap = cv2.VideoCapture(str(path))
        self.assertTrue(cap.read()[0])
        cap.release()
        self.assertFalse((path.parent / 'input').exists())

    def test_supported_video_containers_and_audio_removal(self):
        source = Path(self.tmp.name) / 'source.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(3): writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for ext in ('mp4', 'mov', 'avi', 'mkv'):
            with self.subTest(ext=ext):
                path = Path(self.tmp.name) / ('source.' + ext)
                if ext != 'avi':
                    subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(source), '-f', 'lavfi', '-i', 'sine=frequency=440', '-shortest', '-c:v', 'mpeg4', '-c:a', 'aac', str(path)], check=True)
                job = self.upload('clip.' + ext, path.read_bytes())
                demo.process(job)
                self.assertEqual(demo.status(job)['status'], 'completed')
                result = subprocess.run([ffmpeg, '-i', str(demo.video(job).path)], capture_output=True, text=True)
                self.assertIn('Video:', result.stderr)
                self.assertNotIn('Audio:', result.stderr)

    @unittest.skipUnless(os.getenv('RUN_MODEL_TESTS') == '1', 'requires the local YOLO checkpoint')
    def test_checkpoint_processes_video_with_person_detections(self):
        from ultralytics import YOLO
        model = YOLO(str(demo.MODEL))
        self.assertEqual(model.names, demo.EXPECTED)
        frame = cv2.imread(str(demo.ROOT.parent / 'test_0111.jpg'))
        self.assertIsNotNone(frame)
        frame = cv2.resize(frame, (640, 480))
        self.assertGreater(len(model.predict(frame, device='cpu', imgsz=640, verbose=False)[0].boxes), 0)
        clip = Path(self.tmp.name) / 'source.avi'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 5, (640, 480))
        self.assertTrue(writer.isOpened())
        for _ in range(2): writer.write(frame)
        writer.release()
        job = self.upload('real.avi', clip.read_bytes())
        with patch.object(demo, 'model', model): demo.process(job)
        self.assertEqual(demo.status(job)['status'], 'completed')
        cap = cv2.VideoCapture(str(demo.video(job).path))
        ok, annotated = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertGreater(np.mean(cv2.absdiff(frame, annotated)), 1.0)


if __name__ == '__main__': unittest.main()
