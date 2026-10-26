from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "baseline_100" / "weights" / "best.pt"


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"找不到模型权重：{WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    output = model.export(format="onnx", imgsz=640, device="cpu")
    print(f"ONNX 已导出：{output}")


if __name__ == "__main__":
    main()