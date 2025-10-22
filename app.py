import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
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

        /* Lapisan transparan lembut */
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

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: white !important;
            color: #222 !important;
            border-right: 1.5px solid rgba(0,0,0,0.1);
        }}

        /* Tombol */
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

# Ganti path gambar sesuai nama file kamu
add_bg_from_local("f6391d84-1c14-43f3-8d53-44c8685754d5.png")

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

    st.markdown("---")
    show_info = st.button("📘 Tentang Alpaca")

# ==========================
# INFO ALPACA BUTTON
# ==========================
if show_info:
    st.markdown("""
    <div class="result-box" style="max-width:700px; margin:auto;">
        <h3>🦙 Fakta Singkat tentang Alpaca</h3>
        <p>
        Alpaca adalah hewan mamalia dari keluarga unta yang berasal dari Pegunungan Andes, Amerika Selatan 🌄.  
        Mereka terkenal karena bulunya yang lembut, hangat, dan hipoalergenik — bahkan lebih halus dari wol domba!
        </p>
        <p>
        Alpaca memiliki sifat lembut, cerdas, dan suka berkelompok.  
        Mereka berkomunikasi menggunakan suara lembut yang disebut “humming”.  
        Selain itu, alpaca ramah terhadap manusia dan sering digunakan untuk terapi hewan karena karakternya yang tenang 🩵.
        </p>
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
<div style="text-align:center; font-size:14px; color:#333;">
    by <b>@naylarhmdn</b> | <i>Alpaca Vision Project 🦙</i>
</div>
""", unsafe_allow_html=True)
