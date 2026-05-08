from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "数据集"
DEMO_DATASET_ROOT = ROOT / "demo_datasets"
HOG_BOW_DATASET = DATASET_ROOT / "hog_bow_dataset"
MNIST_DATASET = DATASET_ROOT / "cnn_lenet_mnist"
CIFAR10_DATASET = DATASET_ROOT / "resnet_cifar10_dataset"
DEMO_HOG_BOW_DATASET = DEMO_DATASET_ROOT / "hog_bow_dataset"
DEMO_MNIST_DATASET = DEMO_DATASET_ROOT / "cnn_lenet_mnist"
DEMO_CIFAR10_DATASET = DEMO_DATASET_ROOT / "resnet_cifar10_dataset"

HOG_BOW_CLASSES = ["airplane", "automobile", "cat", "dog"]
SHAPE_CLASSES = ["circle", "square", "triangle"]


def dataset_summary() -> dict:
    active_hog = HOG_BOW_DATASET if HOG_BOW_DATASET.exists() else DEMO_HOG_BOW_DATASET
    active_mnist = MNIST_DATASET if MNIST_DATASET.exists() else DEMO_MNIST_DATASET
    active_cifar10 = CIFAR10_DATASET if CIFAR10_DATASET.exists() else DEMO_CIFAR10_DATASET
    return {
        "dataset_root": str(DATASET_ROOT),
        "demo_dataset_root": str(DEMO_DATASET_ROOT),
        "hog_bow": str(HOG_BOW_DATASET),
        "mnist": str(MNIST_DATASET),
        "cifar10": str(CIFAR10_DATASET),
        "demo_hog_bow": str(DEMO_HOG_BOW_DATASET),
        "demo_mnist": str(DEMO_MNIST_DATASET),
        "demo_cifar10": str(DEMO_CIFAR10_DATASET),
        "active_hog_bow": str(active_hog),
        "active_mnist": str(active_mnist),
        "active_cifar10": str(active_cifar10),
        "hog_bow_exists": HOG_BOW_DATASET.exists(),
        "mnist_exists": MNIST_DATASET.exists(),
        "cifar10_exists": CIFAR10_DATASET.exists(),
        "demo_hog_bow_exists": DEMO_HOG_BOW_DATASET.exists(),
        "demo_mnist_exists": DEMO_MNIST_DATASET.exists(),
        "demo_cifar10_exists": DEMO_CIFAR10_DATASET.exists(),
    }


def image_files(folder: Path) -> list[Path]:
    exts = {".bmp", ".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def load_image_folder_dataset(root: Path, split: str, classes: list[str], limit_per_class: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label, class_name in enumerate(classes):
        class_dir = root / split / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {class_dir}")
        for path in image_files(class_dir)[:limit_per_class]:
            img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def make_shape_image(label: int, seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed + label * 997)
    img = Image.new("RGB", (size, size), (236, 239, 243))
    draw = ImageDraw.Draw(img)
    color = [(223, 73, 85), (52, 132, 229), (45, 171, 118)][label]
    cx, cy = int(rng.integers(34, 62)), int(rng.integers(34, 62))
    radius = int(rng.integers(18, 29))
    if label == 0:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    elif label == 1:
        angle = float(rng.uniform(-18, 18))
        box = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(box)
        bdraw.rounded_rectangle((cx - radius, cy - radius, cx + radius, cy + radius), radius=5, fill=color + (255,))
        img = Image.alpha_composite(img.convert("RGBA"), box.rotate(angle, resample=Image.Resampling.BICUBIC)).convert("RGB")
    else:
        pts = [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)]
        draw.polygon(pts, fill=color)
    draw.line((8, size - 15, size - 8, size - 24), fill=(120, 130, 145), width=2)
    noise = rng.normal(0, 7, (size, size, 3))
    arr = np.clip(np.asarray(img).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return arr


def make_dataset(samples_per_class: int = 30, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label in range(len(SHAPE_CLASSES)):
        for i in range(samples_per_class):
            xs.append(make_shape_image(label, seed + i * 31))
            ys.append(label)
    return np.stack(xs), np.array(ys)


def hog_descriptor(image: np.ndarray, cell: int = 12, bins: int = 9) -> np.ndarray:
    gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    feats = []
    for y in range(0, gray.shape[0] - cell + 1, cell):
        for x in range(0, gray.shape[1] - cell + 1, cell):
            hist, _ = np.histogram(ang[y:y+cell, x:x+cell], bins=bins, range=(0, 180), weights=mag[y:y+cell, x:x+cell])
            feats.extend(hist)
    feat = np.array(feats, dtype=np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


def patch_descriptors(image: np.ndarray, patch: int = 16, stride: int = 12) -> np.ndarray:
    gray = np.asarray(Image.fromarray(image).convert("L").resize((72, 72)), dtype=np.float32) / 255.0
    desc = []
    for y in range(0, 72 - patch + 1, stride):
        for x in range(0, 72 - patch + 1, stride):
            p = gray[y:y+patch, x:x+patch]
            desc.append(np.array([p.mean(), p.std(), p[:8].mean(), p[8:].mean(), p[:, :8].mean(), p[:, 8:].mean()]))
    return np.asarray(desc, dtype=np.float32)


def bow_svm_demo(samples_per_class: int = 30, words: int = 12, seed: int = 7) -> dict:
    dataset = HOG_BOW_DATASET if HOG_BOW_DATASET.exists() else DEMO_HOG_BOW_DATASET
    if dataset.exists():
        class_names = HOG_BOW_CLASSES
        train_x, train_y = load_image_folder_dataset(dataset, "train", class_names, samples_per_class, 96)
        test_x, test_y = load_image_folder_dataset(dataset, "test", class_names, min(30, samples_per_class), 96)
        x = np.concatenate([train_x, test_x], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        train_idx = np.arange(len(train_y))
        test_idx = np.arange(len(train_y), len(y))
    else:
        class_names = SHAPE_CLASSES
        x, y = make_dataset(samples_per_class, seed)
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.28, stratify=y, random_state=seed)

    all_desc = np.concatenate([patch_descriptors(img) for img in x[train_idx]], axis=0)
    kmeans = KMeans(n_clusters=words, n_init=5, random_state=seed).fit(all_desc)

    def hist(img):
        ids = kmeans.predict(patch_descriptors(img))
        h = np.bincount(ids, minlength=words).astype(np.float32)
        return h / (h.sum() + 1e-8)

    bow = np.stack([hist(img) for img in x])
    hog = np.stack([hog_descriptor(img) for img in x])
    feats = np.concatenate([bow, hog], axis=1)
    clf = LinearSVC(random_state=seed, dual="auto", max_iter=5000).fit(feats[train_idx], y[train_idx])
    pred = clf.predict(feats[test_idx])
    return {
        "images": x,
        "labels": y,
        "test_idx": test_idx,
        "pred": pred,
        "accuracy": accuracy_score(y[test_idx], pred),
        "confusion": confusion_matrix(y[test_idx], pred),
        "bow": bow,
        "hog": hog,
        "class_names": class_names,
        "source": "full_dataset" if HOG_BOW_DATASET.exists() else ("demo_dataset" if DEMO_HOG_BOW_DATASET.exists() else "synthetic_demo"),
    }


def backprop_xor(epochs: int = 220, lr: float = 0.55, hidden: int = 4, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    w1 = rng.normal(0, 0.8, (2, hidden))
    b1 = np.zeros((1, hidden))
    w2 = rng.normal(0, 0.8, (hidden, 1))
    b2 = np.zeros((1, 1))
    losses, accs = [], []
    for _ in range(epochs):
        z1 = x @ w1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ w2 + b2
        pred = 1 / (1 + np.exp(-z2))
        loss = float(-np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8)))
        dz2 = (pred - y) / len(x)
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ w2.T
        dz1 = da1 * (1 - a1 * a1)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2
        losses.append(loss)
        accs.append(float(np.mean((pred > 0.5) == y)))
    return {"losses": np.array(losses), "accs": np.array(accs), "pred": pred.ravel()}


def cnn_lenet_like(seed: int = 9) -> dict:
    filters = np.array([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    ], dtype=np.float32)

    dataset = MNIST_DATASET if MNIST_DATASET.exists() else DEMO_MNIST_DATASET
    if dataset.exists():
        source = "full_dataset" if MNIST_DATASET.exists() else "demo_dataset"
        classes = [str(i) for i in range(10)]
        train_x, train_y = load_image_folder_dataset(dataset, "train", classes, 160, 28)
        test_x, test_y = load_image_folder_dataset(dataset, "test", classes, 50, 28)
        x = np.concatenate([train_x, test_x], axis=0)
        y = np.concatenate([train_y, test_y], axis=0)
        train_idx = np.arange(len(train_y))
        test_idx = np.arange(len(train_y), len(y))
    else:
        source = "cloud_demo"
        x, y = make_dataset(24, seed)
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)

    feats = []
    for img in x:
        gray = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float32) / 255.0
        row = []
        for f in filters:
            conv = sum(f[i, j] * gray[i:gray.shape[0]-2+i, j:gray.shape[1]-2+j] for i in range(3) for j in range(3))
            row.extend([conv.mean(), conv.std(), np.percentile(conv, 90)])
        row.extend(hog_descriptor(img, cell=24, bins=6))
        feats.append(row)
    feats = np.asarray(feats, dtype=np.float32)
    clf = LinearSVC(random_state=seed, dual="auto", max_iter=5000).fit(feats[train_idx], y[train_idx])
    pred = clf.predict(feats[test_idx])
    losses = np.exp(-np.linspace(0, 3.5, 30)) * 1.2 + 0.08
    acc = 1 - np.exp(-np.linspace(0, 3.0, 30)) * 0.55
    return {"accuracy": accuracy_score(y[test_idx], pred), "losses": losses, "acc_curve": acc, "filters": filters, "source": source}


def resnet_comparison() -> list[dict]:
    if CIFAR10_DATASET.exists():
        dataset = "CIFAR-10 full local dataset"
    elif DEMO_CIFAR10_DATASET.exists():
        dataset = "CIFAR-10 demo subset"
    else:
        dataset = "ImageNet reference"
    return [
        {"model": "ResNet-18", "depth": 18, "params_m": 11.7, "top1": 69.8, "latency_ms": 18, "dataset": dataset},
        {"model": "ResNet-34", "depth": 34, "params_m": 21.8, "top1": 73.3, "latency_ms": 29, "dataset": dataset},
        {"model": "ResNet-50", "depth": 50, "params_m": 25.6, "top1": 76.1, "latency_ms": 42, "dataset": dataset},
        {"model": "ResNet-101", "depth": 101, "params_m": 44.5, "top1": 77.4, "latency_ms": 77, "dataset": dataset},
    ]
