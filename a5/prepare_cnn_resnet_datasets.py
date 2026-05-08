from pathlib import Path
import gzip
import pickle
import struct
import tarfile
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
MNIST_RAW_DIR = RAW_DIR / "mnist"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_ARCHIVE = RAW_DIR / "cifar-10-python.tar.gz"
CIFAR_RAW_DIR = RAW_DIR / "cifar-10-batches-py"
DATASET_ROOT = ROOT / "数据集"
MNIST_OUT_DIR = DATASET_ROOT / "cnn_lenet_mnist"
CIFAR_OUT_DIR = DATASET_ROOT / "resnet_cifar10_dataset"

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def ensure_raw_datasets():
    MNIST_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MNIST_FILES.items():
        path = MNIST_RAW_DIR / filename
        if not path.exists():
            print(f"Downloading MNIST file: {filename}")
            urllib.request.urlretrieve(url, path)

    CIFAR_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not CIFAR_ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {CIFAR_URL}")
        urllib.request.urlretrieve(CIFAR_URL, CIFAR_ARCHIVE)
    if not CIFAR_RAW_DIR.exists():
        print("Extracting CIFAR-10 archive")
        with tarfile.open(CIFAR_ARCHIVE, "r:gz") as tar:
            tar.extractall(CIFAR_ARCHIVE.parent)


def write_bmp(path, image):
    if image.ndim == 2:
        image_rgb = np.repeat(image[:, :, None], 3, axis=2)
    else:
        image_rgb = image

    height, width, _ = image_rgb.shape
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data_size = (width * 3 + row_padding) * height
    file_size = 14 + 40 + pixel_data_size

    with path.open("wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        f.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        for y in range(height - 1, -1, -1):
            f.write(image_rgb[y, :, ::-1].astype(np.uint8).tobytes())
            f.write(b"\x00" * row_padding)


def load_mnist_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic number in {path}: {magic}")
        data = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic number in {path}: {magic}")
        labels = np.frombuffer(f.read(count), dtype=np.uint8)
    return labels


def export_mnist_split(split, image_file, label_file):
    images = load_mnist_images(MNIST_RAW_DIR / image_file)
    labels = load_mnist_labels(MNIST_RAW_DIR / label_file)
    if len(images) != len(labels):
        raise ValueError(f"MNIST {split} image/label count mismatch")

    counts = {str(i): 0 for i in range(10)}
    for digit in counts:
        (MNIST_OUT_DIR / split / digit).mkdir(parents=True, exist_ok=True)

    for idx, (image, label) in enumerate(zip(images, labels), start=1):
        digit = str(int(label))
        counts[digit] += 1
        write_bmp(MNIST_OUT_DIR / split / digit / f"{digit}_{counts[digit]:05d}.bmp", image)

    return counts


def load_cifar_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="bytes")


def cifar_rows_to_images(rows):
    return rows.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def export_cifar_split(split, batch_names, label_names):
    counts = {name: 0 for name in label_names}
    for name in label_names:
        (CIFAR_OUT_DIR / split / name).mkdir(parents=True, exist_ok=True)

    for batch_name in batch_names:
        batch = load_cifar_pickle(CIFAR_RAW_DIR / batch_name)
        labels = batch[b"labels"]
        images = cifar_rows_to_images(batch[b"data"])
        for image, label in zip(images, labels):
            class_name = label_names[label]
            counts[class_name] += 1
            write_bmp(CIFAR_OUT_DIR / split / class_name / f"{class_name}_{counts[class_name]:05d}.bmp", image)

    return counts


def write_readme(path, title, source, layout, counts):
    lines = [
        f"# {title}",
        "",
        f"Source: {source}",
        "",
        "Directory layout:",
        "",
        "```text",
        layout,
        "```",
        "",
        "Image counts:",
        "",
    ]
    for split, split_counts in counts.items():
        total = sum(split_counts.values())
        lines.append(f"- {split}: {total}")
        for label, count in split_counts.items():
            lines.append(f"  - {label}: {count}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_mnist():
    train_counts = export_mnist_split(
        "train",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    )
    test_counts = export_mnist_split(
        "test",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    write_readme(
        MNIST_OUT_DIR / "README.md",
        "LeNet-5 MNIST Dataset",
        "MNIST handwritten digits, Yann LeCun, Corinna Cortes, Christopher J.C. Burges",
        "cnn_lenet_mnist/\n  train/0/*.bmp ... train/9/*.bmp\n  test/0/*.bmp ... test/9/*.bmp",
        {"train": train_counts, "test": test_counts},
    )


def export_cifar():
    meta = load_cifar_pickle(CIFAR_RAW_DIR / "batches.meta")
    label_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    train_counts = export_cifar_split("train", [f"data_batch_{i}" for i in range(1, 6)], label_names)
    test_counts = export_cifar_split("test", ["test_batch"], label_names)
    write_readme(
        CIFAR_OUT_DIR / "README.md",
        "ResNet CIFAR-10 Dataset",
        "CIFAR-10 official Python version, Alex Krizhevsky, Vinod Nair, Geoffrey Hinton",
        "resnet_cifar10_dataset/\n  train/<class_name>/*.bmp\n  test/<class_name>/*.bmp",
        {"train": train_counts, "test": test_counts},
    )


def main():
    ensure_raw_datasets()
    export_mnist()
    export_cifar()
    print(f"Done. MNIST image folder: {MNIST_OUT_DIR}")
    print(f"Done. CIFAR-10 image folder: {CIFAR_OUT_DIR}")


if __name__ == "__main__":
    main()
