from pathlib import Path
import pickle
import struct
import tarfile
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parent
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = ROOT / "raw" / "cifar-10-python.tar.gz"
RAW_DIR = ROOT / "raw" / "cifar-10-batches-py"
OUT_DIR = ROOT / "数据集" / "hog_bow_dataset"

SELECTED_CLASSES = ["airplane", "automobile", "cat", "dog"]
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 30
UPSCALE = 3


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="bytes")


def ensure_cifar_raw():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {CIFAR_URL}")
        urllib.request.urlretrieve(CIFAR_URL, ARCHIVE)
    if not RAW_DIR.exists():
        print("Extracting CIFAR-10 archive")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(ARCHIVE.parent)


def write_bmp(path, image_rgb):
    height, width, _ = image_rgb.shape
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data_size = (width * 3 + row_padding) * height
    file_size = 14 + 40 + pixel_data_size

    with path.open("wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        f.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        for y in range(height - 1, -1, -1):
            row_bgr = image_rgb[y, :, ::-1].astype(np.uint8).tobytes()
            f.write(row_bgr)
            f.write(b"\x00" * row_padding)


def cifar_rows_to_images(rows):
    images = rows.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if UPSCALE > 1:
        images = np.repeat(np.repeat(images, UPSCALE, axis=1), UPSCALE, axis=2)
    return images


def prepare_split(split, batch_names, per_class, label_to_name, selected_labels):
    counts = {label: 0 for label in selected_labels}
    for label in selected_labels:
        (OUT_DIR / split / label_to_name[label]).mkdir(parents=True, exist_ok=True)

    for batch_name in batch_names:
        batch = load_pickle(RAW_DIR / batch_name)
        labels = batch[b"labels"]
        images = cifar_rows_to_images(batch[b"data"])

        for image, label in zip(images, labels):
            if label not in selected_labels or counts[label] >= per_class:
                continue
            class_name = label_to_name[label]
            idx = counts[label] + 1
            write_bmp(OUT_DIR / split / class_name / f"{class_name}_{idx:04d}.bmp", image)
            counts[label] += 1

        if all(count >= per_class for count in counts.values()):
            break

    missing = {label_to_name[label]: per_class - count for label, count in counts.items() if count < per_class}
    if missing:
        raise RuntimeError(f"Not enough images for split {split}: {missing}")


def write_manifest(label_to_name, selected_labels):
    lines = [
        "# HOG + Bag of Words + SVM sample image dataset",
        "",
        "Source: CIFAR-10 official Python version",
        "Official page: https://www.cs.toronto.edu/~kriz/cifar.html",
        "",
        "This folder contains a small extracted subset for A5 task 1.",
        f"Classes: {', '.join(label_to_name[label] for label in selected_labels)}",
        f"Training images per class: {TRAIN_PER_CLASS}",
        f"Test images per class: {TEST_PER_CLASS}",
        f"Image format: BMP, {32 * UPSCALE}x{32 * UPSCALE}, RGB",
        "",
        "Directory layout:",
        "  hog_bow_dataset/train/<class_name>/*.bmp",
        "  hog_bow_dataset/test/<class_name>/*.bmp",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_cifar_raw()

    meta = load_pickle(RAW_DIR / "batches.meta")
    label_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    name_to_label = {name: idx for idx, name in enumerate(label_names)}
    selected_labels = [name_to_label[name] for name in SELECTED_CLASSES]
    label_to_name = {idx: name for idx, name in enumerate(label_names)}

    prepare_split(
        "train",
        [f"data_batch_{i}" for i in range(1, 6)],
        TRAIN_PER_CLASS,
        label_to_name,
        selected_labels,
    )
    prepare_split("test", ["test_batch"], TEST_PER_CLASS, label_to_name, selected_labels)
    write_manifest(label_to_name, selected_labels)

    total = len(SELECTED_CLASSES) * (TRAIN_PER_CLASS + TEST_PER_CLASS)
    print(f"Done. Wrote {total} BMP images to {OUT_DIR}")


if __name__ == "__main__":
    main()
