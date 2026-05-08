from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import core

st.set_page_config(page_title="A5 图像识别与神经网络 Vibe Coding", layout="wide")


def load_default_image() -> np.ndarray:
    for name in ["default_image.jpg", "default_image.jpeg", "default_image.png"]:
        path = ROOT / "assets" / name
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    return np.zeros((240, 360, 3), dtype=np.uint8) + 235


def show_array(arr, caption: str):
    st.image(np.asarray(arr), caption=caption, use_container_width=True)


def get_dataset_summary() -> dict:
    full_root = ROOT / "数据集"
    demo_root = ROOT / "demo_datasets"
    full_hog = full_root / "hog_bow_dataset"
    full_mnist = full_root / "cnn_lenet_mnist"
    full_cifar10 = full_root / "resnet_cifar10_dataset"
    demo_hog = demo_root / "hog_bow_dataset"
    demo_mnist = demo_root / "cnn_lenet_mnist"
    demo_cifar10 = demo_root / "resnet_cifar10_dataset"
    fallback = {
        "dataset_root": str(full_root),
        "demo_dataset_root": str(demo_root),
        "hog_bow": str(full_hog),
        "mnist": str(full_mnist),
        "cifar10": str(full_cifar10),
        "demo_hog_bow": str(demo_hog),
        "demo_mnist": str(demo_mnist),
        "demo_cifar10": str(demo_cifar10),
        "active_hog_bow": str(full_hog if full_hog.exists() else demo_hog),
        "active_mnist": str(full_mnist if full_mnist.exists() else demo_mnist),
        "active_cifar10": str(full_cifar10 if full_cifar10.exists() else demo_cifar10),
        "hog_bow_exists": full_hog.exists(),
        "mnist_exists": full_mnist.exists(),
        "cifar10_exists": full_cifar10.exists(),
        "demo_hog_bow_exists": demo_hog.exists(),
        "demo_mnist_exists": demo_mnist.exists(),
        "demo_cifar10_exists": demo_cifar10.exists(),
    }
    if hasattr(core, "dataset_summary"):
        return core.dataset_summary()
    return fallback


st.title("A5 图像识别与神经网络 Vibe Coding")
st.caption("学生：裘典儿 2025213456 · Agent/LLM：Codex GPT-5")

uploaded = st.sidebar.file_uploader("可选上传单张展示图（不参与训练）", type=["jpg", "jpeg", "png"])
if uploaded:
    image = np.asarray(Image.open(uploaded).convert("RGB"))
    image_source = f"上传图片：{uploaded.name}"
    st.sidebar.image(image, caption=image_source, use_container_width=True)
else:
    image = None
    image_source = "未上传单张展示图"
    st.sidebar.caption("未上传单张展示图；页面主体将展示当前任务实际使用的样本。")

summary = get_dataset_summary()
st.sidebar.divider()
st.sidebar.caption("当前数据集目录")
st.sidebar.code(summary["active_hog_bow"], language=None)
if all([summary["hog_bow_exists"], summary["mnist_exists"], summary["cifar10_exists"]]):
    st.sidebar.success("已检测到本地完整数据集")
elif all([summary["demo_hog_bow_exists"], summary["demo_mnist_exists"], summary["demo_cifar10_exists"]]):
    st.sidebar.success("已检测到云端小型 demo 数据集")
else:
    st.sidebar.info("云端展示模式：未检测到图片数据集，使用合成演示与预置对比结果")

tab_names = ['HOG+BOW+SVM', '反向传播', 'CNN', 'ResNet对比', '部署']
tabs = st.tabs(tab_names)


with tabs[0]:
    st.subheader("HOG + Bag of Words + SVM")
    st.caption(f"当前读取：{summary['active_hog_bow']}")
    if summary["hog_bow_exists"]:
        st.success("当前使用本地完整 HOG+BOW 数据集。")
    elif summary["demo_hog_bow_exists"]:
        st.success("当前使用随网页上传的小型 HOG+BOW demo 数据集。")
    else:
        st.info("当前未检测到 HOG+BOW 图片数据集，已切换为轻量形状分类演示。")
    samples = st.slider("每类样本数", 12, 45, 24, 3)
    words = st.slider("视觉词袋大小", 6, 20, 12, 2)
    result = core.bow_svm_demo(samples, words)
    st.metric("测试准确率", f"{result['accuracy']*100:.1f}%")
    st.write("混淆矩阵")
    st.dataframe(result["confusion"], use_container_width=True)
    class_names = result.get("class_names", getattr(core, "CLASSES", ["circle", "square", "triangle"]))
    st.write("当前分类样本")
    cols = st.columns(len(class_names))
    for i, cls in enumerate(class_names):
        idx = int(np.where(result["labels"] == i)[0][0])
        cols[i].image(result["images"][idx], caption=cls, use_container_width=True)

with tabs[1]:
    st.subheader("反向传播演示")
    lr = st.slider("学习率", 0.05, 1.0, 0.55, 0.05)
    epochs = st.slider("训练轮数", 50, 500, 220, 10)
    bp = core.backprop_xor(epochs=epochs, lr=lr)
    st.line_chart({"loss": bp["losses"], "accuracy": bp["accs"]})
    st.write("XOR 输出概率：", np.round(bp["pred"], 3))

with tabs[2]:
    st.subheader("LeNet 风格 CNN 训练与测试")
    st.caption(f"当前读取：{summary['active_mnist']}")
    if summary["mnist_exists"]:
        st.success("当前使用本地完整 MNIST 数据集。")
    elif summary["demo_mnist_exists"]:
        st.success("当前使用随网页上传的小型 MNIST demo 数据集。")
    else:
        st.info("当前未检测到 MNIST 图片数据集，云端展示训练曲线和轻量 CNN 特征分类结果。")
    cnn = core.cnn_lenet_like()
    st.metric("测试准确率", f"{cnn['accuracy']*100:.1f}%")
    st.line_chart({"loss": cnn["losses"], "accuracy": cnn["acc_curve"]})
    st.write("三个卷积核：水平边缘、垂直边缘、Laplacian。")

with tabs[3]:
    st.subheader("ResNet 深度性能对比")
    st.caption(f"当前读取：{summary['active_cifar10']}")
    if summary["cifar10_exists"]:
        st.success("当前使用本地完整 CIFAR-10 数据集。")
    elif summary["demo_cifar10_exists"]:
        st.success("当前使用随网页上传的小型 CIFAR-10 demo 数据集。")
    else:
        st.info("当前未检测到 CIFAR-10 图片数据集，云端展示不同深度 ResNet 的预置性能对比表。")
    rows = core.resnet_comparison()
    st.dataframe(rows, use_container_width=True)
    st.bar_chart({r["model"]: r["top1"] for r in rows})


with tabs[-1]:
    st.subheader("部署说明")
    st.write("上传本文件夹到 GitHub 后，如果把该文件夹作为仓库根目录，Streamlit Cloud 的 Main file path 填 `streamlit_app.py`。")
    st.write("如果上传整个周四作业目录，则 Main file path 填 `a5/streamlit_app.py`。")
    st.write("左侧上传控件只用于替换单张展示图，不等同于训练数据集。")
    st.write("本地完整数据集统一放在 `a5/数据集/` 下；GitHub 可上传的小型演示数据集放在 `a5/demo_datasets/` 下。")
    st.write("应用会优先读取完整数据集；没有完整数据集时读取 demo 数据集；两者都没有时进入合成展示模式。")
