from pathlib import Path
import argparse
import sys

# Tránh lỗi mã hóa khi đầu ra PowerShell dùng bảng mã cũ.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")


# Các đường dẫn mặc định tính từ thư mục gốc của dự án, không phụ thuộc
# vào thư mục hiện tại khi chạy lệnh.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = (
    PROJECT_ROOT / "kaggle-output" / "model-comparison"
    / "yolo11s-fit-within-640" / "weights" / "best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "inference-results"
RUN_NAME = "yolo11s-single-image"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25


def main():
    parser = argparse.ArgumentParser(description="Chạy YOLO trên một ảnh.")
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="Đường dẫn model .pt")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Thư mục lưu kết quả")
    args = parser.parse_args()

    weights_path = args.weights.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not weights_path.is_file():
        parser.error(f"Không tìm thấy model: {weights_path}")
    if not image_path.is_file():
        parser.error(f"Không tìm thấy ảnh: {image_path}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        parser.exit(1, f"Thiếu thư viện: {exc}. Hãy cài torch và ultralytics vào môi trường Python.\n")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")
    print(f"Model: {weights_path}")
    print(f"Ảnh đầu vào: {image_path}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
        device=device, save=True, project=str(args.output.expanduser().resolve()),
        name=RUN_NAME, exist_ok=False, verbose=False,
    )
    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("\nKhông phát hiện đối tượng nào với ngưỡng hiện tại.")
    else:
        print(f"\nPhát hiện {len(boxes)} đối tượng:\n")
        for index, box in enumerate(boxes, start=1):
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(
                f"{index}. Class: {class_name} | Confidence: {confidence:.3f} | "
                f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    saved_dir = Path(result.save_dir)
    print(f"\nThư mục kết quả: {saved_dir}")
    saved_image_path = saved_dir / image_path.name
    if saved_image_path.is_file():
        print(f"Ảnh kết quả: {saved_image_path}")
    else:
        print("Không tìm thấy ảnh theo tên gốc. Các file trong thư mục kết quả:")
        for file in saved_dir.iterdir():
            print(file)


if __name__ == "__main__":
    main()
