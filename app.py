import streamlit as st
import joblib
import numpy as np
import json
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Prediktor Kardiovaskular",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed", # Sidebar disembunyikan secara default
)

def load_lottiefile(filepath: str):
    """Memuat file animasi Lottie dari path yang diberikan."""
    with open(filepath, "r") as f:
        return json.load(f)

def load_css(file_name: str):
    """Memuat dan menyuntikkan file CSS kustom ke dalam aplikasi."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path: str):
    """Memuat model machine learning dari file .joblib dengan cache."""
    return joblib.load(model_path)

# Memuat aset eksternal dan model
load_css('assets/style.css')
model = load_model('best_gradient_boosting_model.pkl')

#--- Bagian Header Aplikasi ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Prediktor Risiko Penyakit Jantung 🩺")
    st.subheader("Sayangi jantungmu seperti kamu menyayangi dirimu sendiri ❤️")
    st.write(
        """
        Aplikasi ini menggunakan *Machine Learning* untuk memprediksi risiko penyakit
        kardiovaskular. Masukkan data medis Anda di bawah untuk melihat
        hasil prediksi secara instan.
        """
    )

with header_col2:
    lottie_url = "https://assets6.lottiefiles.com/packages/lf20_tijmpky4.json"
    st_lottie(lottie_url, height=200, key="heartbeat")

st.markdown("---")

#--- Form Input di Halaman Utama ---
with st.expander("📝 Klik di sini untuk memasukkan data kamu", expanded=True):
    with st.form("prediction_form"):
        # Membagi form menjadi 3 kolom untuk tata letak yang lebih rapi
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("##### Data Pribadi")
            age_in_years = st.slider('Umur (dalam tahun)', 25, 70, 50)
            gender = st.selectbox('Gender', (1, 2), format_func=lambda x: 'Pria' if x == 1 else 'Wanita')
            height = st.slider('Tinggi (cm)', 50, 250, 165)
            weight = st.slider('Berat (kg)', 30, 200, 70)
            bmi = st.slider('Indeks Massa Tubuh (BMI)', 10.0, 50.0, 25.0, step=0.1)

        with col2:
            st.write("##### Data Medis")
            ap_hi = st.slider('Tekanan Darah Sistolik (ap_hi)', 60, 240, 120)
            ap_lo = st.slider('Tekanan Darah Diastolik (ap_lo)', 40, 180, 80)
            cholesterol = st.selectbox('Kolesterol', (1, 2, 3), format_func=lambda x: {1: 'Normal', 2: 'Di Atas Normal', 3: 'Jauh Di Atas Normal'}[x])
            gluc = st.selectbox('Glukosa', (1, 2, 3), format_func=lambda x: {1: 'Normal', 2: 'Di Atas Normal', 3: 'Jauh Di Atas Normal'}[x])
        
        with col3:
            st.write("##### Gaya Hidup")
            smoke = st.radio('Merokok?', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            alco = st.radio('Konsumsi alkohol?', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            active = st.radio('Aktif berolahraga?', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

        # Tombol submit di tengah
        st.markdown("---")
        submit_button = st.form_submit_button(label='✨ Lakukan Prediksi!')


#--- Logika dan Tampilan Hasil Prediksi ---
if submit_button:
    with st.spinner('Menganalisis data Anda...'):
        # Mengonversi umur dari tahun kembali ke hari untuk model
        age_in_days = age_in_years * 365

        # PENTING: Pastikan urutan fitur di bawah ini SAMA PERSIS dengan saat training model
        features = np.array([[
            age_in_days, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active, bmi
        ]])
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        st.markdown("---")
        st.header("🔬 Hasil Analisis Model")
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            st.subheader("Hasil Prediksi:")
            if prediction[0] == 1:
                st.error("Risiko Tinggi")
                st.markdown('<div class="result-icon">💔</div>', unsafe_allow_html=True)
            else:
                st.success("Risiko Rendah")
                st.markdown('<div class="result-icon">❤️‍🩹</div>', unsafe_allow_html=True)
        
        with result_col2:
            st.subheader("Tingkat Keyakinan Model:")
            prob_risk = probability[0][1] * 100
            st.metric(label="Probabilitas Risiko Tinggi", value=f"{prob_risk:.2f} %")
            st.progress(int(prob_risk))

    st.info(
        "**Disclaimer:** Hasil prediksi ini berdasarkan model Machine Learning dan tidak menggantikan "
        "diagnosis medis profesional. Silakan konsultasikan dengan dokter untuk evaluasi kesehatan yang akurat."
    )
