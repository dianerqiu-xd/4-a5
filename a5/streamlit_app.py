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


st.title("A5 图像识别与神经网络 Vibe Coding")
st.caption("学生：裘典儿 2025213456 · Agent/LLM：Codex GPT-5")

uploaded = st.sidebar.file_uploader("上传图片替换默认素材", type=["jpg", "jpeg", "png"])
if uploaded:
    image = np.asarray(Image.open(uploaded).convert("RGB"))
    image_source = f"上传图片：{uploaded.name}"
else:
    image = load_default_image()
    image_source = "默认图片：assets/default_image.jpg"

st.sidebar.image(image, caption=image_source, use_container_width=True)
summary = core.dataset_summary()
st.sidebar.divider()
st.sidebar.caption("默认数据集目录")
st.sidebar.code(summary["dataset_root"], language=None)
if all([summary["hog_bow_exists"], summary["mnist_exists"], summary["cifar10_exists"]]):
    st.sidebar.success("已检测到本地完整数据集")
else:
    st.sidebar.info("云端展示模式：未上传大数据集，使用轻量演示与预置对比结果")

tab_names = ['HOG+BOW+SVM', '反向传播', 'CNN', 'ResNet对比', '部署']
tabs = st.tabs(tab_names)


with tabs[0]:
    st.subheader("HOG + Bag of Words + SVM")
    st.caption(f"默认读取：{summary['hog_bow']}")
    if not summary["hog_bow_exists"]:
        st.info("当前未检测到本地 HOG+BOW 数据集，已切换为轻量形状分类演示，适合 Streamlit Cloud 展示。")
    st.image(image, caption=image_source, use_container_width=True)
    samples = st.slider("每类样本数", 12, 45, 24, 3)
    words = st.slider("视觉词袋大小", 6, 20, 12, 2)
    result = core.bow_svm_demo(samples, words)
    st.metric("测试准确率", f"{result['accuracy']*100:.1f}%")
    st.write("混淆矩阵")
    st.dataframe(result["confusion"], use_container_width=True)
    cols = st.columns(len(result["class_names"]))
    for i, cls in enumerate(result["class_names"]):
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
    st.caption(f"默认读取：{summary['mnist']}")
    if not summary["mnist_exists"]:
        st.info("当前未检测到本地 MNIST 完整数据集，云端展示训练曲线和轻量 CNN 特征分类结果。")
    cnn = core.cnn_lenet_like()
    st.metric("测试准确率", f"{cnn['accuracy']*100:.1f}%")
    st.line_chart({"loss": cnn["losses"], "accuracy": cnn["acc_curve"]})
    st.write("三个卷积核：水平边缘、垂直边缘、Laplacian。")

with tabs[3]:
    st.subheader("ResNet 深度性能对比")
    st.caption(f"默认读取：{summary['cifar10']}")
    if not summary["cifar10_exists"]:
        st.info("当前未检测到本地 CIFAR-10 完整数据集，云端展示不同深度 ResNet 的预置性能对比表。")
    rows = core.resnet_comparison()
    st.dataframe(rows, use_container_width=True)
    st.bar_chart({r["model"]: r["top1"] for r in rows})


with tabs[-1]:
    st.subheader("部署说明")
    st.write("上传本文件夹到 GitHub 后，如果把该文件夹作为仓库根目录，Streamlit Cloud 的 Main file path 填 `streamlit_app.py`。")
    st.write("如果上传整个周四作业目录，则 Main file path 填 `a5/streamlit_app.py`。")
    st.write("本应用保留上传控件，可用自己的图片替换默认素材。")
    st.write("默认数据集统一放在 `a5/数据集/` 下；如需替换数据集，保持 `train/类别名` 与 `test/类别名` 结构即可。")
    st.write("生成公网 URL 时无需上传 `数据集/` 和 `raw/`，云端会自动进入展示模式；本地运行时检测到完整数据集后会优先使用真实数据。")
