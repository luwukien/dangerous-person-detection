"""Single-user, local video review application."""
import logging
import math
import os
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import imageio_ffmpeg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parent
MODEL = Path(os.getenv("MODEL_PATH", str(ROOT.parent / "kaggle-output/model-comparison/yolo11s-fit-within-640/weights/best.pt")))
EXPECTED = {0: "normal_person", 1: "potentially_dangerous_person"}
MAX_BYTES = int(os.getenv("MAX_VIDEO_MB", "0")) * 1024 * 1024
MAX_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "0"))
logger = logging.getLogger(__name__)
jobs = {}
jobs_lock = threading.Lock()
model = None
model_error = "Model chưa được tải"
workspace = None
executor = None


def set_status(job, status, progress=0, message=""):
    with jobs_lock:
        jobs[job].update(status=status, progress=progress, message=message)


def check_job(job):
    with jobs_lock:
        if job not in jobs:
            raise HTTPException(404, "Không tìm thấy video")
        return dict(jobs[job])


def process(job):
    directory = Path(workspace.name) / job
    source = directory / "input"
    intermediate = directory / "frames.mp4"
    output = directory / "result.mp4"
    try:
        set_status(job, "processing", message="Đang phân tích video")
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not math.isfinite(fps) or not 0 < fps <= 120 or count <= 0 or not 0 < width <= 3840 or not 0 < height <= 2160:
            cap.release()
            raise ValueError("Không thể đọc video hoặc độ phân giải không được hỗ trợ (tối đa 3840×2160).")
        if MAX_SECONDS and count / fps > MAX_SECONDS:
            cap.release()
            raise ValueError(f"Video vượt quá giới hạn {MAX_SECONDS} giây.")
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError("Không thể khởi tạo bộ mã hóa video.")
        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if MAX_SECONDS and n > MAX_SECONDS * fps + 1:
                    raise ValueError(f"Video vượt quá giới hạn {MAX_SECONDS} giây.")
                result = model.predict(frame, device="cpu", imgsz=640, verbose=False)[0]
                writer.write(result.plot())
                if n == 1 or n % max(1, int(fps)) == 0:
                    set_status(job, "processing", min(95, int(n / count * 95)), "Đang phân tích video")
        finally:
            cap.release()
            writer.release()
        if n == 0:
            raise ValueError("Video không có khung hình đọc được.")
        set_status(job, "processing", 96, "Đang tạo video MP4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(intermediate), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        set_status(job, "completed", 100, "Xử lý hoàn tất")
    except Exception as exc:
        logger.exception("Video processing failed for job %s", job)
        output.unlink(missing_ok=True)
        message = str(exc) if isinstance(exc, ValueError) else "Không thể xử lý video. Hãy kiểm tra định dạng và thử lại."
        set_status(job, "failed", 0, message)
    finally:
        source.unlink(missing_ok=True)
        intermediate.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app):
    global workspace, executor, model, model_error
    workspace = tempfile.TemporaryDirectory(prefix="video-review-")
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL))
        if model.task != "detect" or model.names != EXPECTED:
            model = None
            raise ValueError("Sai loại model hoặc thứ tự lớp")
        model_error = ""
    except Exception as exc:
        model = None
        model_error = f"Không thể tải model: {type(exc).__name__}. Kiểm tra MODEL_PATH và checkpoint."
        logger.exception("Model loading failed")
    yield
    executor.shutdown(wait=True)
    workspace.cleanup()
    with jobs_lock:
        jobs.clear()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    return {"max_mb": MAX_BYTES // 1048576, "max_seconds": MAX_SECONDS, "model_error": model_error}


@app.post("/api/jobs", status_code=202)
def create_job(video: UploadFile = File(...)):
    if not model:
        raise HTTPException(503, model_error)
    if not video.filename or Path(video.filename).suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
        raise HTTPException(400, "Hãy chọn video MP4, MOV, AVI hoặc MKV.")
    job = uuid.uuid4().hex
    directory = Path(workspace.name) / job
    directory.mkdir()
    source = directory / "input"
    try:
        size = 0
        with source.open("wb") as out:
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if MAX_BYTES and size > MAX_BYTES:
                    raise HTTPException(413, "Video vượt quá giới hạn dung lượng.")
                out.write(chunk)
        if not size:
            raise HTTPException(400, "Video trống.")
    except Exception:
        source.unlink(missing_ok=True)
        directory.rmdir()
        raise
    finally:
        video.file.close()
    with jobs_lock:
        jobs[job] = {"id": job, "status": "queued", "progress": 0, "message": "Đang chờ xử lý"}
    executor.submit(process, job)
    return {"id": job}


@app.get("/api/jobs/{job}")
def status(job: str):
    return check_job(job)


@app.get("/api/jobs/{job}/video")
def video(job: str):
    if check_job(job)["status"] != "completed":
        raise HTTPException(409, "Video chưa sẵn sàng")
    return FileResponse(Path(workspace.name) / job / "result.mp4", media_type="video/mp4", headers={"Cache-Control": "no-store"})
