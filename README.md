# 🎧 TypeTune AI – MBTI-Based Music Recommender

**TypeTune AI** is a Streamlit app that predicts a user’s **MBTI personality type** from a written sentence and recommends songs tailored to that type.

It uses simple but effective machine learning techniques like TF-IDF and KNN to analyze language and music features.

---

## ✨ Key Features

- 🧠 Predicts MBTI type using Logistic Regression  
- 🎵 Recommends music based on average MBTI song profiles  
- 📈 Uses real Spotify track data to compare musical attributes  
- 💡 Streamlit-powered web interface for real-time interaction

---

## 🧪 How to Run

### Requirements

- Python 3.8+  
- Streamlit  
- Packages in `requirements.txt`

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your default browser.

---

## 📁 Project Structure

```
typetune-ai/
├── app.py               # Streamlit interface
├── models/              # Trained model .pkl files
├── data/                # Input datasets (MBTI & Spotify)
├── notebooks/           # Jupyter Notebooks (analysis)
├── requirements.txt     # Dependencies
└── README.md
```

---

## 📊 Datasets Used

- `mbti_dataset.csv` — User posts with MBTI labels  
- `spotify_dataset.csv` — Danceability, valence, energy, etc.  
- `mbti_personality.csv` — Aggregated profiles per MBTI type

---

## 🤖 Modeling Pipeline

| Step                 | Method                        |
|----------------------|-------------------------------|
| Text Vectorization   | TF-IDF                        |
| Personality Prediction | Logistic Regression (16 classes) |
| Music Recommendation | KNN on music feature space    |

---

## 🖼️ App Preview

![resim](https://github.com/user-attachments/assets/864dfcc2-d988-4178-bb16-1e152e534c06)

---

## 📬 Contact

**Furkan Gença**  
[@furkangenca](https://github.com/furkangenca)

---

## ⚠️ License

This project is **not open source**.  
Code is provided for demonstration only. Reuse, redistribution, or commercial use is **not permitted**.
