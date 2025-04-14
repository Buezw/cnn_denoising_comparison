import streamlit as st
import os
import time
from PIL import Image
import random

from train import train, get_model
from data.mnist_noisy_dataset import NoisyMNIST

# ------------------- 页面设置 -------------------
st.set_page_config(page_title="CNN 图像去噪平台", layout="wide")
st.title("🧠 CNN 图像去噪对比平台")

# ------------------- 参数设置 -------------------
model = st.selectbox("选择模型结构", ["vgg", "alexnet", "nin"])
epochs = st.slider("训练轮数", 1, 50, 10)
lr = st.number_input("学习率", value=0.005, step=0.001, format="%.5f")

# ------------------- 自动刷新逻辑 -------------------
AUTO_REFRESH_INTERVAL = 3  # 秒
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "epoch_selected" not in st.session_state:
    st.session_state.epoch_selected = 0

if time.time() - st.session_state.last_refresh > AUTO_REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# ------------------- 左栏：训练控制 -------------------
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🚀 开始训练"):
        with st.status("训练中...", expanded=True) as status:
            logs = []

            def log_fn(msg):
                logs.append(msg)
                st.write("✅ " + msg)

            train(model_name=model, epochs=epochs, lr=lr, log_fn=log_fn)

            st.session_state.epoch_selected = epochs - 1
            status.update(label="🎉 训练完成", state="complete")

# ------------------- 右栏：图像预览 -------------------
with col2:
    st.subheader("📷 输出图可视化")

    epoch = st.slider("查看第几轮输出", 0, epochs - 1,
                      value=st.session_state.get("epoch_selected", 0),
                      key="epoch_slider")

    img_path = f"results/output_images/{model}/train/epoch_{epoch:02d}.png"

    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        # ✅ 不使用 key 参数，适配旧版 streamlit
        st.image(img_bytes, caption=f"{model} @ Epoch {epoch}", use_column_width=True)
    else:
        st.warning("尚未生成该轮图像。请等待训练或点击刷新。")

    if st.button("🔁 手动刷新图像"):
        st.experimental_rerun()
