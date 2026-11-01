#!/usr/bin/env python3
"""Train YOLO11 variants on the same detection dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


# Model pretrained và patience tham khảo cho từng phiên bản.
VARIANTS = {
    "yolo11n": {"model": "yolo11n.pt", "patience": 35},
    "yolo11s": {"model": "yolo11s.pt", "patience": 35},
    "yolo11m": {"model": "yolo11m.pt", "patience": 40},
    "yolo11l": {"model": "yolo11l.pt", "patience": 40},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare YOLO11 model variants."
    )

    parser.add_argument(
        "--variant",
        choices=VARIANTS.keys(),
        required=True,
        help="Model cần train: yolo11n, yolo11s, yolo11m hoặc yolo11l.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Đường dẫn đến file data.yaml.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Số epoch tối đa. Mặc định: 150.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. Mặc định: 8.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước đầu vào. Mặc định: 640.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Đặt patience chung. Nếu bỏ qua, dùng cấu hình theo variant.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Thiết bị train, ví dụ "0" hoặc "cpu".',
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Thư mục lưu kết quả train.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Tên run. Mặc định: <variant>-fit-within-640.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data).expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file data.yaml: {data_path}"
        )

    variant_config = VARIANTS[args.variant]
    model_name = variant_config["model"]

    # Nếu truyền --patience thì dùng giá trị đó.
    # Nếu không, dùng giá trị tham khảo của từng variant.
    patience = (
        args.patience
        if args.patience is not None
        else variant_config["patience"]
    )

    run_name = args.name or f"{args.variant}-fit-within-{args.imgsz}"

    print(f"Model: {model_name}")
    print(f"Dataset: {data_path}")
    print(f"Epochs tối đa: {args.epochs}")
    print(f"Patience: {patience}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Run name: {run_name}")

    # Load pretrained weights.
    model = YOLO(model_name)

    # Fine-tune trên dataset.
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=2,
        optimizer="auto",

        # Giữ augmentation giống nhau giữa các model.
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,

        seed=42,
        deterministic=True,
        amp=True,

        project=args.project,
        name=run_name,
        exist_ok=False,
        save=True,
        save_period=10,
        plots=True,
    )


if __name__ == "__main__":
    main()