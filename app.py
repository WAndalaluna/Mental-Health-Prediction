import streamlit as st
import pandas as pd
import joblib

# Load models
models = {
    'Did you seek any specialist for a treatment?': joblib.load("Treatment_Seeked.pkl"),
}

# Define label encoder for consistent processing
label_encoder = {
    'Gender': {'Male': 0, 'Female': 1},
    'Marital status': {'Single': 0, 'Married': 1},
    'Course': {'Engineering': 0, 'BCS': 1, 'Islamic education': 2, 'Other': 3},
    'Depression': {'Yes': 1, 'No': 0},
    'Anxiety': {'Yes': 1, 'No': 0},
    'Panic_Attack': {'Yes': 1, 'No': 0},
}

# Streamlit interface
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        color: #2E8B57;
        font-weight: bold;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #555555;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #4682B4;
        margin-top: 20px;
    }
    .result {
        font-size: 18px;
        color: #4B0082;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="title">🌟 Aplikasi Prediksi Kesehatan Mental 🌟</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Aplikasi ini memprediksi kemungkinan kondisi kesehatan mental berdasarkan input pengguna.</p>', unsafe_allow_html=True)

# Sidebar with mental health information
st.sidebar.title("Kesehatan Mental")
st.sidebar.info(
    """
    Kesehatan mental mencakup kesejahteraan emosional, psikologis, dan sosial kita. Ini memengaruhi cara kita berpikir, merasa, dan bertindak dalam kehidupan sehari-hari. Kesehatan mental yang baik membantu kita menghadapi stres, menjalin hubungan dengan orang lain, dan membuat keputusan.

    **Sumber Informasi:**
    - [WHO Kesehatan Mental](https://www.who.int/health-topics/mental-health)
    - [MentalHealth.gov](https://www.mentalhealth.gov)
    - [Layanan Bantuan Nasional](https://www.samhsa.gov/find-help/national-helpline)
    """
)

# Sidebar with mental health community information
st.sidebar.title("Komunitas Kesehatan Mental")
st.sidebar.info(
    """
    Bergabung dengan komunitas kesehatan mental dapat memberikan dukungan emosional dan informasi yang bermanfaat. Berikut beberapa komunitas yang dapat Anda ikuti:

    - [Komunitas Peduli Kesehatan Mental](https://www.komunitaspedulikesehatanmental.org)
    - [Mental Health Community](https://www.mentalhealthcommunity.org)
    - [Support Group Indonesia](https://www.supportgroupindonesia.org)
    - [Mental Health Foundation](https://www.mentalhealth.org.uk)
    """
)

# Input form
st.markdown('<p class="section-header">🗒 Masukkan Detail Anda</p>', unsafe_allow_html=True)

name = st.text_input("Nama", placeholder="Masukkan nama Anda")
age = st.number_input("Usia", min_value=10, max_value=100, step=1, help="Usia Anda dalam tahun")
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], help="Pilih jenis kelamin Anda")
course = st.selectbox("Jurusan", ["Engineering", "BCS", "Islamic education", "Other"], help="Pilih jurusan Anda")
year_of_study = st.selectbox("Tahun Studi", ["Year 1", "Year 2", "Year 3", "Year 4"], help="Pilih tahun studi Anda")
cgpa = st.slider("IPK", 0.0, 4.0, step=0.01, help="Masukkan Indeks Prestasi Kumulatif Anda")
marital_status = st.selectbox("Status Pernikahan", ["Single", "Married"], help="Pilih status pernikahan Anda")
depression = st.selectbox("Depresi", ["Yes", "No"], help="Apakah Anda pernah mengalami depresi?")
anxiety = st.selectbox("Kecemasan", ["Yes", "No"], help="Apakah Anda pernah mengalami kecemasan?")
panic_attack = st.selectbox("Serangan Panik", ["Yes", "No"], help="Apakah Anda pernah mengalami serangan panik?")

# Predict button
if st.button("Prediksi"):
    # Prepare input data with the correct column names
    input_data = pd.DataFrame({
        'Gender': [label_encoder['Gender'][gender]],
        'Age': [age],
        'Course': [label_encoder['Course'][course]],
        'Year_of_Study': [int(year_of_study.split()[1])],
        'CGPA': [cgpa],
        'Marital_status': [label_encoder['Marital status'][marital_status]],
        'Depression': [label_encoder['Depression'][depression]],
        'Anxiety': [label_encoder['Anxiety'][anxiety]],
        'Panic_Attack': [label_encoder['Panic_Attack'][panic_attack]],
    })

    # Make predictions
    results = {}
    for condition, model in models.items():
        proba = model.predict_proba(input_data)
        prob_yes = proba[0][1]  # Probability for 'Yes'
        prob_no = proba[0][0]  # Probability for 'No'
        results[condition] = {'Yes': prob_yes * 100, 'No': prob_no * 100}

    # Display results in columns
    st.markdown('<p class="section-header">📊 Persentase untuk Mengunjungi Spesialis atau Mendapatkan Perawatan</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<p class='result'>**Ya:** {results['Did you seek any specialist for a treatment?']['Yes']:.2f}%</p>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<p class='result'>**Tidak:** {results['Did you seek any specialist for a treatment?']['No']:.2f}%</p>", unsafe_allow_html=True)

    st.success("✅ Prediksi selesai! Lihat hasil Anda di atas.")

# Footer with additional information
st.markdown(
    """
    <footer style="text-align: center; margin-top: 50px; font-size: 14px; color: #555555;">
    <hr>
    <p>Dikembangkan dengan ❤️ untuk mendukung kesadaran kesehatan mental.</p>
    <p><a href="https://www.who.int/health-topics/mental-health" target="_blank">Pelajari Lebih Lanjut Tentang Kesehatan Mental</a></p>
    </footer>
    """,
    unsafe_allow_html=True,
)