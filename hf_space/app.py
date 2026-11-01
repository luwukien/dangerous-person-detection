"""Gradio entry point for a Hugging Face ZeroGPU Space."""
import logging
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import spaces
from ultralytics import YOLO
from video_processing import EXPECTED, process_video, validate_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent / "best.pt"))
DEVICE = "cuda" if os.environ.get("SPACE_ID") else "cpu"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "dangerous-person-demo-results"
OUTPUT_DIR.mkdir(exist_ok=True)
model = None
model_error = ""
try:
    model = YOLO(str(MODEL_PATH))
    if model.task != "detect" or model.names != EXPECTED:
        raise ValueError("Unexpected model task or class mapping")
    if DEVICE == "cuda":
        model.to("cuda")
except Exception:
    logger.exception("Model loading failed")
    model = None
    model_error = "Không thể tải model. Người quản trị cần kiểm tra checkpoint và cấu hình Space."


@spaces.GPU(duration=120)
def run_on_gpu(path, output, progress=gr.Progress()):
    return process_video(path, output, model, DEVICE, progress)


def analyze(path, progress=gr.Progress()):
    try:
        validate_video(path)
        if model is None:
            raise ValueError(model_error)
        # Files are retained briefly for playback/download and cleaned on future requests.
        for old in OUTPUT_DIR.glob("*.mp4"):
            try:
                if old.stat().st_mtime < time.time() - 3600:
                    old.unlink()
            except OSError:
                logger.warning("Could not remove expired result", exc_info=True)
        fd, output = tempfile.mkstemp(suffix=".mp4", dir=OUTPUT_DIR)
        os.close(fd)
        try:
            result = run_on_gpu(path, output, progress)
        except Exception:
            Path(output).unlink(missing_ok=True)
            raise
        return result, result
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logger.exception("Video processing failed")
        raise gr.Error("Không thể xử lý video. GPU có thể đang bận hoặc đã hết quota; hãy thử lại sau.") from exc


with gr.Blocks(title="Phân tích video với YOLO", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("""# Phân tích video với YOLO
Tải video lên để xem người được đánh dấu trong từng khung hình. Tối đa **20 MB, 15 giây**.

**Lưu ý:** Đây là công cụ hỗ trợ con người xem xét; dự đoán có thể sai hoặc bỏ sót. Nhãn `potentially_dangerous_person` mô tả ngữ cảnh hình ảnh, không xác định ý định hay mức độ nguy hiểm thực tế. Không tải video nhạy cảm; tệp được xử lý tạm thời trên máy chủ. Video kết quả không có âm thanh.""")
    if model_error:
        gr.Markdown(f"**{model_error}**")
    video = gr.File(label="Video đầu vào", file_types=[".mp4", ".mov", ".avi", ".mkv"], type="filepath")
    submit = gr.Button("Bắt đầu phân tích", variant="primary")
    result = gr.Video(label="Video kết quả")
    download = gr.File(label="Tải video kết quả")
    submit.click(analyze, inputs=video, outputs=[result, download], concurrency_limit=1)

demo.queue(max_size=5, default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch(max_file_size="20mb", allowed_paths=[str(OUTPUT_DIR)])
