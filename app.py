import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import numpy as np
from PIL import Image
import base64

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="🦙 Alpaca Vision", page_icon="🧠", layout="wide")

# --- Fungsi untuk background ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    bg_image = f"""
        <style>
        .main {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            font-family: 'Poppins', sans-serif;
        }}

        .main::before {{
            content: "";
            position: absolute;
            top: 0; left: 0;
            right: 0; bottom: 0;
            background: rgba(255, 255, 255, 0.55);
            z-index: -1;
        }}

        h1 {{
            text-align: center;
            color: #4A0072;
            font-weight: 800;
            text-shadow: 1px 1px 5px rgba(255,255,255,0.8);
        }}

        section[data-testid="stSidebar"] {{
            background-color: white !important;
            color: #222 !important;
            border-right: 1.5px solid rgba(0,0,0,0.1);
        }}

        div.stButton > button {{
            background: linear-gradient(90deg, #8E2DE2 0%, #4A00E0 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            transition: 0.3s;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
        }}

        div.stButton > button:hover {{
            background: linear-gradient(90deg, #9B4DFF 0%, #512DA8 100%);
            transform: translateY(-2px);
        }}

        .result-box {{
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1rem;
            box-shadow: 0 4px 20px rgba(100, 0, 150, 0.15);
            text-align: center;
            color: #4B0082;
            font-size: 18px;
            font-weight: 600;
        }}

        footer {{visibility: hidden;}}
        </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Ganti sesuai nama file background kamu
add_bg_from_local("f6391d84-1c14-43f3-8d53-44c8685754d5.png")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI HEADER
# ==========================
st.title("🦙 Alpaca Vision Dashboard")

with st.sidebar:
    st.header("✨
