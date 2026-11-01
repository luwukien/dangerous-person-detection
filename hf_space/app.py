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
    model_error = "Unable to load the model. The administrator should check the checkpoint and Space configuration."


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
        raise gr.Error("Unable to process the video. The GPU may be busy or its quota may be exhausted. Please try again later.") from exc


with gr.Blocks(title="Video Analysis with YOLO", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("""# Video Analysis with YOLO
Upload a video to view person detections in each frame. Maximum **20 MB, 15 seconds**.

**Note:** This tool supports human review; predictions may be incorrect or miss detections. The `potentially_dangerous_person` label describes visual context and does not determine intent or actual threat level. Do not upload sensitive videos; files are processed temporarily on the server. Output videos have no audio.""")
    if model_error:
        gr.Markdown(f"**{model_error}**")
    video = gr.File(label="Input video", file_types=[".mp4", ".mov", ".avi", ".mkv"], type="filepath")
    submit = gr.Button("Analyze video", variant="primary")
    result = gr.Video(label="Result video")
    download = gr.File(label="Download result video")
    submit.click(analyze, inputs=video, outputs=[result, download], concurrency_limit=1)

demo.queue(max_size=5, default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch(max_file_size="20mb", allowed_paths=[str(OUTPUT_DIR)])
