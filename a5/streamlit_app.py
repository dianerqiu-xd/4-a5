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

tab_names = ['HOG+BOW+SVM', '反向传播', 'CNN', 'ResNet对比', '部署']
tabs = st.tabs(tab_names)


with tabs[0]:
    st.subheader("HOG + Bag of Words + SVM")
    st.image(image, caption=image_source, use_container_width=True)
    samples = st.slider("每类样本数", 12, 45, 24, 3)
    words = st.slider("视觉词袋大小", 6, 20, 12, 2)
    result = core.bow_svm_demo(samples, words)
    st.metric("测试准确率", f"{result['accuracy']*100:.1f}%")
    st.write("混淆矩阵")
    st.dataframe(result["confusion"], use_container_width=True)
    cols = st.columns(3)
    for i, cls in enumerate(core.CLASSES):
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
    cnn = core.cnn_lenet_like()
    st.metric("测试准确率", f"{cnn['accuracy']*100:.1f}%")
    st.line_chart({"loss": cnn["losses"], "accuracy": cnn["acc_curve"]})
    st.write("三个卷积核：水平边缘、垂直边缘、Laplacian。")

with tabs[3]:
    st.subheader("ResNet 深度性能对比")
    rows = core.resnet_comparison()
    st.dataframe(rows, use_container_width=True)
    st.bar_chart({r["model"]: r["top1"] for r in rows})


with tabs[-1]:
    st.subheader("部署说明")
    st.write("上传本文件夹到 GitHub 后，如果把该文件夹作为仓库根目录，Streamlit Cloud 的 Main file path 填 `streamlit_app.py`。")
    st.write("如果上传整个周四作业目录，则 Main file path 填 `a5/streamlit_app.py`。")
    st.write("本应用保留上传控件，可用自己的图片替换默认素材。")
