from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=30,
        project=str(ROOT / "runs"),
        name="baseline_script",
    )


if __name__ == "__main__":
    main()