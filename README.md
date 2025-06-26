# 🎧 MBTI Music Recommender AI

This project is a personality-based music recommendation system that predicts a user’s MBTI (Myers-Briggs Type Indicator) personality type based on a sentence they write, and then suggests songs that match that personality. It integrates text mining, machine learning, and recommendation system techniques into a streamlined Streamlit web app.

![resim](https://github.com/user-attachments/assets/864dfcc2-d988-4178-bb16-1e152e534c06)

---

## 🚀 Key Features

- 🧠 Predicts MBTI type from a single sentence using TF-IDF and Logistic Regression

- 🔍 Analyzes average music features (danceability, energy, valence, tempo) for each MBTI type

- 🤖 Uses KNN to recommend songs similar to the predicted profile

- 🌐 Clean and minimal Streamlit-based user interface

- 📊 Includes Jupyter Notebooks for development, analysis, and reproducibility

---

## 📁 Project Structure

```
mbti-music-recommender-ai/
├── app.py               # Streamlit interface
├── models/              # Pretrained models (.pkl)
├── data/                # Datasets (.csv)
├── notebooks/           # Development notebooks and experiments
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## 🧪 How to Run

Install the required packages:
```
pip install -r requirements.txt
```
Run the application:
```
streamlit run app.py
```
The app will automatically open in your default browser.

---

## 📊 Datasets Used

- mbti_dataset.csv — Contains user posts and MBTI labels

- spotify_dataset.csv — Song features (danceability, energy, valence, tempo)

- mbti_personality.csv — Aggregated music features per MBTI type

---

## 🤖 Modeling Pipeline
| Task                         | Method                                     |
| ---------------------------- | ------------------------------------------ |
| **MBTI Prediction**          | TF-IDF vectorization + Logistic Regression |
| **Music Profile Generation** | Averaged feature vectors for MBTI types    |
| **Song Recommendation**      | K-Nearest Neighbors (Euclidean distance)   |


The Logistic Regression model achieved 68% accuracy on 16-class MBTI classification. Deep learning models like CNN were tested but discarded due to overfitting and class imbalance.

---

📈 Highlights from the Report

-   Integrated three datasets from Kaggle (MBTI posts, personality-based playlists, and song features)

-   Performed full NLP preprocessing, including tokenization, lemmatization, stop-word filtering, TF-IDF, and SMOTE

-   Applied N-gram and frequency analysis to identify language patterns per personality type

-   Used dynamic prototype vectors instead of static averages to boost personalization

-   Streamlit app delivers 10 personalized song suggestions based on predicted MBTI profile

