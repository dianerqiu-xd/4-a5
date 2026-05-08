from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
FULL_ROOT = ROOT / "数据集"
DEMO_ROOT = ROOT / "demo_datasets"


SPECS = [
    {
        "src": FULL_ROOT / "hog_bow_dataset",
        "dst": DEMO_ROOT / "hog_bow_dataset",
        "classes": ["airplane", "automobile", "cat", "dog"],
        "train": 15,
        "test": 5,
    },
    {
        "src": FULL_ROOT / "cnn_lenet_mnist",
        "dst": DEMO_ROOT / "cnn_lenet_mnist",
        "classes": [str(i) for i in range(10)],
        "train": 20,
        "test": 5,
    },
    {
        "src": FULL_ROOT / "resnet_cifar10_dataset",
        "dst": DEMO_ROOT / "resnet_cifar10_dataset",
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "train": 20,
        "test": 5,
    },
]


def image_files(folder: Path) -> list[Path]:
    exts = {".bmp", ".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def copy_split(src_root: Path, dst_root: Path, split: str, classes: list[str], count: int) -> int:
    copied = 0
    for class_name in classes:
        src_dir = src_root / split / class_name
        dst_dir = dst_root / split / class_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source folder: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        for old in image_files(dst_dir):
            old.unlink()
        files = image_files(src_dir)[:count]
        if len(files) < count:
            raise RuntimeError(f"Not enough images in {src_dir}: need {count}, got {len(files)}")
        for idx, src in enumerate(files, start=1):
            shutil.copy2(src, dst_dir / f"{class_name}_{idx:03d}{src.suffix.lower()}")
            copied += 1
    return copied


def write_readme():
    lines = [
        "# A5 Demo Datasets",
        "",
        "Small image subsets committed with the Streamlit app for cloud demos.",
        "The full local datasets live in `数据集/` and are ignored by Git.",
        "",
        "Contents:",
        "",
    ]
    for spec in SPECS:
        total = len(spec["classes"]) * (spec["train"] + spec["test"])
        lines.append(f"- `{spec['dst'].name}`: {total} images")
        lines.append(f"  - train: {spec['train']} images per class")
        lines.append(f"  - test: {spec['test']} images per class")
    lines.append("")
    (DEMO_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    for spec in SPECS:
        total += copy_split(spec["src"], spec["dst"], "train", spec["classes"], spec["train"])
        total += copy_split(spec["src"], spec["dst"], "test", spec["classes"], spec["test"])
    write_readme()
    print(f"Done. Wrote {total} demo images to {DEMO_ROOT}")


if __name__ == "__main__":
    main()
