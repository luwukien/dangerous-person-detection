"""Render a video with person tracking and stable, reversible class labels."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

from video_app.temporal import VideoTracker

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "test-video-img/Download - Trim.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "inference-results/Download_yolo11s_stable(2).mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "model-comparison/yolo11s-fit-within-640/weights/best.pt")
    parser.add_argument("--raw", action="store_true", help="Render original per-frame predictions without stabilization")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if any(args.output.with_suffix(suffix).exists() for suffix in (args.output.suffix, ".csv", ".json")):
        parser.error("Output or diagnostics already exist; choose a new --output path")
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or min(width, height, count) <= 0:
        cap.release()
        raise ValueError("Không mở được video đầu vào")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = None
    processed = 0
    try:
        model = YOLO(str(args.weights))
        expected = {0: "normal_person", 1: "potentially_dangerous_person"}
        if model.task != "detect" or model.names != expected:
            raise ValueError("Unexpected model task or class mapping")
        tracker = None if args.raw else VideoTracker(fps)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                   "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
        with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as trace:
            writer = csv.writer(trace)
            writer.writerow(["frame", "track_id", "raw_class", "raw_confidence", "stable_class"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(frame, device="cpu", imgsz=640, conf=0.5,
                                       agnostic_nms=not args.raw, verbose=False)[0]
                annotated = result.plot() if tracker is None else tracker.annotate(result)
                ffmpeg.stdin.write(annotated.tobytes())
                if tracker is not None:
                    writer.writerows(tracker.observations)
                processed += 1
                if processed == 1 or processed % 150 == 0:
                    print(f"Đã xử lý {processed}/{count} khung hình", flush=True)
    finally:
        cap.release()
        if ffmpeg is not None:
            ffmpeg.stdin.close()
            return_code = ffmpeg.wait()
    if return_code != 0 or processed != count:
        raise RuntimeError(f"Xử lý chưa hoàn tất: FFmpeg={return_code}, số khung hình={processed}/{count}")
    summary = {"frames": processed, "fps": fps, "mode": "raw" if args.raw else "stable"}
    if tracker is not None:
        summary.update(raw_tracked_class_switches=tracker.raw_switches,
                       stable_class_switches=tracker.stable_switches,
                       note="Counts compare labels on matched tracks after class-agnostic NMS; not accuracy metrics.")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
