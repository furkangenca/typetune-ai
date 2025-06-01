import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors
import nltk


# Gerekli NLTK verisi
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    return [lemmatizer.lemmatize(word) for word in text.split() if len(word) > 2]


# --- Sayfa ayarları ---
st.set_page_config(
    page_title="MBTI Müzik Öneri Sistemi", 
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Daha uyumlu gri tonları ile güncellenmiş
st.markdown("""
<style>
    /* Ana sayfa arka planı */
    .stApp {
        background-color: #222223 !important;
        color: #e2e1db !important;
    }
    
    /* Ana container */
    .main .block-container {
        background-color: #222223 !important;
        color: #e2e1db !important;
    }
    
    /* Sidebar arka planı - Daha uyumlu koyu gri */
    [data-testid="stSidebar"] {
        background-color: #2d2d30 !important;
        color: #e2e1db !important;
    }
    
    /* Sidebar içeriği */
    [data-testid="stSidebar"] .css-1d391kg, 
    [data-testid="stSidebar"] .css-163ttbj,
    [data-testid="stSidebar"] .css-1wrcr25,
    [data-testid="stSidebar"] .css-6qob1r,
    [data-testid="stSidebar"] .css-1adrfps {
        background-color: #2d2d30 !important;
        color: #e2e1db !important;
    }
    
    /* Header bölümü */
    [data-testid="stHeader"] {
        background-color: #222223 !important;
    }
    
    /* Tüm toolbar elementleri */
    .stToolbar, .stToolbar button, .stToolbar [role="button"] {
        background-color: #222223 !important;
        color: #e2e1db !important;
    }
    
    /* Tüm view options */
    [data-testid="stDecoration"], [data-testid="stDecoration"] div {
        background-color: #222223 !important;
    }
    
    /* Tüm butonlar ve interaktif elementler */
    button, [role="button"], [role="tab"] {
        background-color: #3c3c3c !important;
        color: #e2e1db !important;
    }
    
    /* Tüm divider'lar */
    hr {
        border-color: #3c3c3c !important;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #e2e1db 0%, #babbb8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #babbb8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .mbti-result {
        background: linear-gradient(135deg, #3c3c3c 0%, #4a4a4a 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #e2e1db;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .song-card {
        background: #3c3c3c;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
        margin: 1rem 0;
        border-left: 4px solid #babbb8;
        color: #e2e1db;
    }

    .song-card h4 {
        color: #e2e1db;
        margin-bottom: 0.5rem;
    }

    .song-card p {
        color: #babbb8;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3c3c3c 0%, #4a4a4a 100%) !important;
        color: #e2e1db !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 1.1rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5) !important;
        background: linear-gradient(135deg, #4a4a4a 0%, #5a5a5a 100%) !important;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: #e2e1db !important;
    }
    
    /* Markdown içindeki tüm metin elementleri */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown p, .stMarkdown li, .stMarkdown a {
        color: #e2e1db !important;
    }
    
    /* Sidebar içindeki markdown */
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3, 
    [data-testid="stSidebar"] .stMarkdown h4, 
    [data-testid="stSidebar"] .stMarkdown h5, 
    [data-testid="stSidebar"] .stMarkdown h6, 
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown li, 
    [data-testid="stSidebar"] .stMarkdown a,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #e2e1db !important;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background-color: #3c3c3c !important;
        color: #e2e1db !important;
        border: 2px solid #4a4a4a !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #babbb8 !important;
    }
    
    /* Columns */
    .element-container {
        color: #e2e1db !important;
    }
    
    /* Warning and error messages */
    .stAlert {
        background-color: #3c3c3c !important;
        color: #e2e1db !important;
        border: 1px solid #4a4a4a !important;
    }
    
    /* Spinner */
    .stSpinner {
        color: #e2e1db !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e2e1db !important;
    }
    
    /* Paragraphs */
    p {
        color: #e2e1db !important;
    }
    
    /* Sidebar elements */
    .sidebar .element-container {
        margin-bottom: 1rem !important;
    }
    
    /* Session state */
    .stSelectbox > div > div {
        background-color: #3c3c3c !important;
        color: #e2e1db !important;
    }
    
    /* Emoji ve ikonlar */
    .emoji {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: MBTI Hakkında ---
with st.sidebar:
    st.markdown("## 🧠 MBTI Nedir?")
    
    st.markdown("""
    **Myers-Briggs Type Indicator (MBTI)**, kişilik tiplerini 16 farklı kategoride sınıflandıran bir sistemdir.
    
    ### 4 Ana Boyut:
    
    **🔍 Enerji Yönelimi:**
    - **E** (Extraversion) - Dışa dönük
    - **I** (Introversion) - İçe dönük
    
    **🧭 Bilgi Toplama:**
    - **S** (Sensing) - Duyusal
    - **N** (Intuition) - Sezgisel
    
    **⚖️ Karar Verme:**
    - **T** (Thinking) - Düşünsel
    - **F** (Feeling) - Duygusal
    
    **📋 Yaşam Tarzı:**
    - **J** (Judging) - Yargılayıcı
    - **P** (Perceiving) - Algılayıcı
    """)
    
    st.markdown("---")
    st.markdown("### 🎵 Nasıl Çalışır?")
    st.markdown("""
    1. Yazdığınız cümle analiz edilir
    2. MBTI kişilik tipiniz tahmin edilir
    3. Kişiliğinize uygun müzikler önerilir
    """)

# --- Ana Sayfa ---
st.markdown('<h1 class="main-header">🎧 MBTI Müzik Keşfi</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Kişiliğinizi keşfedin, müziğinizi bulun</p>', unsafe_allow_html=True)

# --- Model Yükleme ---
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("models/vectorizer.pkl")
        model = joblib.load("models/model_log.pkl")
        encoder = joblib.load("models/target_encoder.pkl")
        return vectorizer, model, encoder
    except FileNotFoundError:
        st.error("Model dosyaları bulunamadı!")
        return None, None, None

@st.cache_data
def load_datasets():
    try:
        mbti_df = pd.read_csv("data/mbti_dataset.csv")
        spotify_df = pd.read_csv("data/spotify_dataset.csv")
        return mbti_df, spotify_df
    except FileNotFoundError:
        st.error("Veri dosyaları bulunamadı!")
        return None, None

vectorizer, model_log, target_encoder = load_models()
mbti_df, spotify_df = load_datasets()

if vectorizer is None or mbti_df is None:
    st.stop()

# --- Kullanıcı Girişi ---
st.markdown("### 💭 Kendinizi bir cümle ile tanıtın:")

user_input = st.text_area(
    "",
    value="I enjoy deep thinking and being alone in nature.",
    height=100,
    placeholder="Örnek: I love spending time with close friends and exploring new places...",
    label_visibility="collapsed"
)

# --- Fonksiyonlar ---

def predict_mbti(text):
    # Metin temizleme
    text = re.sub(r"https?:\/\/[\^\s]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    
    # Lemmatization 
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if len(word) > 2])
    
    # Tahmin
    try:
        X = vectorizer.transform([text]).toarray()
        pred = model_log.predict(X)
        return target_encoder.inverse_transform(pred)[0]
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
        return "INFP"  # Varsayılan değer


def get_random_representatives(mbti_type, mbti_df, n=5):
    subset = mbti_df[mbti_df['mbti'] == mbti_type.upper()]
    if len(subset) == 0:
        return np.array([])
    selected = subset.sample(n=min(n, len(subset)), random_state=np.random.randint(0, 99999))
    return selected[["danceability_mean", "energy_mean", "valence_mean", "tempo_mean"]].values

def recommend_songs(rep_vec, spotify_df, k=3, max_results=5):
    if len(rep_vec) == 0:
        return pd.DataFrame()

    features = ["danceability", "energy", "valence", "tempo"]
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(spotify_df[features])
    indices = []
    for vec in rep_vec:
        _, idx = knn.kneighbors([vec])
        indices.extend(idx[0])
    
    unique = list(set(indices))
    recommended = spotify_df.iloc[unique][["track_name", "artists"] + features]
    
    return recommended.head(max_results)


# MBTI açıklamaları
mbti_descriptions = {
    "INTJ": "Mimar - Stratejik düşünür, bağımsız ve kararlı",
    "INTP": "Düşünür - Yaratıcı mucit, güçlü teorik bilgi",
    "ENTJ": "Komutan - Cesur lider, her zaman bir yol bulur",
    "ENTP": "Tartışmacı - Akıllı ve meraklı düşünür",
    "INFJ": "Savunucu - Yaratıcı ve içgörülü, ilkeli",
    "INFP": "Arabulucu - Şiirsel, nazik ve özgecil",
    "ENFJ": "Kahraman - Karizmatik ve ilham verici lider",
    "ENFP": "Kampanyacı - Coşkulu, yaratıcı ve sosyal",
    "ISTJ": "Lojistikçi - Pratik ve gerçekçi, güvenilir",
    "ISFJ": "Savunucu - Sıcakkanlı ve özverili koruyucu",
    "ESTJ": "Yönetici - Mükemmel yönetici, eşsiz lider",
    "ESFJ": "Konsolos - Olağanüstü sosyal ve popüler",
    "ISTP": "Virtüöz - Cesur ve pratik deneyimci",
    "ISFP": "Maceracı - Esnek ve çekici sanatçı",
    "ESTP": "Girişimci - Akıllı, enerjik ve algılayıcı",
    "ESFP": "Eğlendirici - Kendiliğinden, enerjik ve coşkulu"
}

# --- Ana Butonlar ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Kişiliğimi Analiz Et"):
        if user_input.strip():
            with st.spinner("Analiz ediliyor..."):
                predicted_type = predict_mbti(user_input)
                
                st.markdown(f"""
                <div class="mbti-result">
                    <h2>🎭 Kişilik Tipiniz: {predicted_type}</h2>
                    <p>{mbti_descriptions.get(predicted_type, "")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.predicted_mbti = predicted_type
        else:
            st.warning("Lütfen bir cümle yazın!")

with col2:
    if st.button("🎵 Müzik Önerilerini Getir"):
        if user_input.strip():
            with st.spinner("Müzikler bulunuyor..."):
                predicted_type = predict_mbti(user_input)
                rep_vec = get_random_representatives(predicted_type, mbti_df)
                
                if len(rep_vec) > 0:
                    recommended = recommend_songs(rep_vec, spotify_df, max_results=5)

                    
                    if not recommended.empty:
                        st.markdown(f"""
                        <div class="mbti-result">
                            <h3>🎧 {predicted_type} İçin Özel Müzikler</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for idx, row in recommended.iterrows():
                            st.markdown(f"""
                            <div class="song-card">
                                <h4 style="color: #e2e1db;">🎵 {row['track_name']}</h4>
                                <p style="color: #babbb8;"><strong>Sanatçı:</strong> {row['artists']}</p>
                                <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.9rem; color: #babbb8;">
                                    <span>🕺 Dans: {row['danceability']:.2f}</span>
                                    <span>⚡ Enerji: {row['energy']:.2f}</span>
                                    <span>😊 Pozitiflik: {row['valence']:.2f}</span>
                                    <span>🎼 Tempo: {row['tempo']:.0f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Öneri bulunamadı.")
        else:
            st.warning("Lütfen bir cümle yazın!")