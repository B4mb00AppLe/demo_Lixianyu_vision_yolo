from pathlib import Path
from collections import Counter
import random
import shutil

root = Path(r"D:\RoboMaster2027\vision_yolo")
source = Path(r"D:\RoboMaster2027\source_data\RM27视觉组第一阶段考核\部分RM装甲板数据集")
label_source = root / "labels_all"
dataset = root / "dataset"
classes = [x.strip() for x in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]

images = sorted(source.glob("*.jpg"))
if len(classes) != 6:
    raise SystemExit(f"classes.txt 应有 6 类，实际为 {len(classes)}")
if not images:
    raise SystemExit("没有找到 JPG 图片")

missing = [img.name for img in images if not (label_source / f"{img.stem}.txt").exists()]
if missing:
    raise SystemExit(f"缺少标签文件，例如：{missing[:5]}")

parts = ("train", "val", "test")
destinations = [dataset / kind / part for kind in ("images", "labels") for part in parts]
if (dataset / "data.yaml").exists() or any(d.exists() and any(d.iterdir()) for d in destinations):
    raise SystemExit("目标目录已有文件；请先检查内容，不要直接覆盖")

for folder in destinations:
    folder.mkdir(parents=True, exist_ok=True)

random.Random(42).shuffle(images)
n_train = int(len(images) * 0.8)
n_val = int(len(images) * 0.1)
splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:],
}

for part, files in splits.items():
    counts = Counter()
    for img in files:
        label = label_source / f"{img.stem}.txt"
        shutil.copy2(img, dataset / "images" / part / img.name)
        shutil.copy2(label, dataset / "labels" / part / label.name)
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                if class_id < 0 or class_id >= len(classes):
                    raise ValueError(f"{label.name} 中有无效类别编号：{class_id}")
                counts[class_id] += 1

    print(f"{part}: {len(files)} 张图片，{sum(counts.values())} 个框")
    print({name: counts[i] for i, name in enumerate(classes)})

yaml_names = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
yaml_text = (
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    f"{yaml_names}\n"
)
(dataset / "data.yaml").write_text(yaml_text, encoding="utf-8")
print("已生成 dataset/data.yaml")