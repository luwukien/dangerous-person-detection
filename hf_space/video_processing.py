"""Video validation and annotated MP4 generation for the public demo."""
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg

MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 15
EXPECTED = {0: "normal_person", 1: "potentially_dangerous_person"}


def validate_video(path):
    if not path or not Path(path).is_file():
        raise ValueError("Please select a video to analyze.")
    if Path(path).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError("Only MP4, MOV, AVI, and MKV videos are supported.")
    size = os.path.getsize(path)
    if not size or size > MAX_BYTES:
        raise ValueError("The video must be non-empty and no larger than 20 MB.")
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not math.isfinite(fps) or not 0 < fps <= 120 or count <= 0 or not 0 < width <= 3840 or not 0 < height <= 2160:
            raise ValueError("Unable to read the video, or its format is unsupported.")
        if count / fps > MAX_SECONDS:
            raise ValueError("The video must be no longer than 15 seconds.")
        return fps, count, width, height
    finally:
        cap.release()


def process_video(path, output, model, device, progress=None):
    fps, count, width, height = validate_video(path)
    cap = cv2.VideoCapture(str(path))
    # The intermediate file is removed even if decoding, inference, or encoding fails.
    with tempfile.TemporaryDirectory(prefix="detection-frames-") as tmp:
        intermediate = str(Path(tmp) / "frames.mp4")
        writer = cv2.VideoWriter(intermediate, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise ValueError("Unable to initialize the video encoder.")
        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if n > MAX_SECONDS * fps + 1:
                    raise ValueError("The video must be no longer than 15 seconds.")
                result = model.predict(frame, device=device, imgsz=640, verbose=False)[0]
                writer.write(result.plot())
                if progress and (n == 1 or n % max(1, int(fps)) == 0):
                    progress(min(n / count * 0.95, 0.95), desc="Analyzing video")
        finally:
            cap.release()
            writer.release()
        if not n:
            raise ValueError("The video contains no readable frames.")
        if progress:
            progress(0.97, desc="Creating result video")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", intermediate,
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output)
