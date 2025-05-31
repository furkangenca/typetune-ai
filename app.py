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

# Lemmatizer sınıfını model ile uyumlu hale getir
class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

# --- Sayfa ayarları ---
st.set_page_config(
    page_title="MBTI Müzik Öneri Sistemi", 
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .mbti-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .song-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        color: #333;
    }

    .song-card h4 {
        color: #333;
        margin-bottom: 0.5rem;
    }

    .song-card p {
        color: #555;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .element-container {
        margin-bottom: 1rem;
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
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if len(word) > 2]
    processed = " ".join(tokens)
    
    # Tahmin
    try:
        X = vectorizer.transform([processed]).toarray()
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
                                <h4 style="color: #333;">🎵 {row['track_name']}</h4>
                                <p style="color: #555;"><strong>Sanatçı:</strong> {row['artists']}</p>
                                <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.9rem; color: #666;">
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
