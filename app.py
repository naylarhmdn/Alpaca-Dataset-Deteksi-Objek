import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# CONFIG & STYLE
# ==========================
st.set_page_config(page_title="🦙 Alpaca Vision", page_icon="🧠", layout="wide")

# CSS Kustom dengan gaya pastel lembut & tampilan modern
st.markdown("""
    <style>
    /* Background utama */
    .main {
        background: linear-gradient(145deg, #F8E8FF 0%, #E3FDFD 100%);
        padding: 2rem;
        font-family: 'Poppins', sans-serif;
    }

    /* Judul utama */
    h1 {
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem !important;
        color: #6A0DAD;
        text-shadow: 2px 2px 10px rgba(106, 13, 173, 0.15);
        margin-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F5E6FF;
        color: #000;
        border-right: 2px solid rgba(122, 28, 172, 0.1);
        box-shadow: 2px 0 15px rgba(122, 28, 172, 0.1);
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h3 {
        color: #5B0080;
        font-weight: 700;
    }

    /* Tombol */
    div.stButton > button {
        background: linear-gradient(90deg, #8E2DE2 0%, #4A00E0 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 4px 10px rgba(138, 43, 226, 0.3);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(138, 43, 226, 0.5);
    }

    /* Gambar hasil */
    .stImage {
        border-radius: 15px !important;
        box-shadow: 0px 6px 18px rgba(106, 13, 173, 0.15);
    }

    /* Kotak hasil prediksi */
    .result-box {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(100, 0, 150, 0.1);
        text-align: center;
        color: #4B0082;
        font-size: 18px;
        font-weight: 600;
        animation: fadeIn 1s ease-in-out;
    }

    /* Efek animasi fade */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI HEADER
# ==========================
st.title("🦙 Alpaca Vision Dashboard")

with st.sidebar:
    st.header("✨ Pengaturan Mode")
    menu = st.selectbox("🎯 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.markdown("---")
    st.markdown("""
        <div style="color:black; font-size:15px;">
            💡 <i>Unggah gambar Alpaca atau Non-Alpaca untuk dideteksi atau diklasifikasikan!</i>
        </div>
    """, unsafe_allow_html=True)

# ==========================
# MAIN CONTENT
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Gambar yang Diupload")
        st.image(img, use_container_width=True)

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            with st.spinner("Sedang mendeteksi objek... ⏳"):
                results = yolo_model(img)
                result_img = results[0].plot()
            st.image(result_img, caption="Output Deteksi YOLO", use_container_width=True)

        elif menu == "Klasifikasi Gambar":
            st.subheader("🧠 Hasil Klasifikasi")
            with st.spinner("Sedang menganalisis gambar..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                probability = np.max(prediction)

            labels = ["Non-Alpaca 🐑", "Alpaca 🦙"]
            st.markdown(f"""
            <div class="result-box">
                <p>📊 <b>Prediksi:</b> {labels[class_index]}</p>
                <p>🔥 <b>Probabilitas:</b> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("""
<hr style="margin-top:3rem;">
<div style="text-align:center; font-size:14px; color:gray;">
    by <b>@naylarhmdn</b> | <i>Alpaca Vision Project 🦙</i>
</div>
""", unsafe_allow_html=True)
