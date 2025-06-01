# 🎧 MBTI Music Recommender

Bu proje, kullanıcının yazdığı bir cümle üzerinden MBTI (Myers-Briggs Type Indicator) kişilik tipini tahmin eder ve bu tipe uygun müzik önerileri sunar. Uygulama Streamlit arayüzüyle çalışır ve eğitilmiş makine öğrenmesi modelleriyle entegredir.

![resim](https://github.com/user-attachments/assets/864dfcc2-d988-4178-bb16-1e152e534c06)

---

## 🚀 Özellikler

- 📜 Serbest metinden MBTI kişilik tipi tahmini
- 🧠 TF-IDF ve Logistic Regression tabanlı sınıflandırma
- 🎼 MBTI tipine göre ortalama müzik özellikleri çıkarımı
- 🤖 KNN ile en uygun müzik önerilerini bulma
- 🌐 Streamlit tabanlı web arayüzü

---

## 📁 Proje Yapısı

```
mbti-music-recommender/
├── app.py # Ana uygulama dosyası (Streamlit)
├── models/ # Eğitilmiş modeller (.pkl)
├── data/ # Veri setleri (.csv)
├── notebooks/ # Geliştirme aşamasındaki notebook'lar
├── requirements.txt # Gereken Python kütüphaneleri
└── README.md # Bu dosya
```

---

## 🧪 Nasıl Çalıştırılır?

### Gereksinimleri yükleyin:

```
pip install -r requirements.txt
```

Uygulamayı başlatın:

```
streamlit run app.py
```

Uygulama tarayıcınızda otomatik olarak açılacaktır.

## 📊 Kullanılan Veri Setleri

    mbti_dataset.csv – MBTI gönderileri

    spotify_dataset.csv – Müzik özellikleri (danceability, energy, valence, tempo)

    mbti_personality.csv – Tip başına özet bilgiler

## 💡 Model Bilgisi

    MBTI Tahmini: TF-IDF + Logistic Regression

    Müzik Profili Tahmini: Ortalama vektör çıkarımı

    Öneri Motoru: KNN (euclidean mesafe)


## ✍️ Geliştirici

    Furkan Gença – GitHub/furkangenca

## 📄 Lisans

MIT License

---


