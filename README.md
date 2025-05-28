# 🎧 MBTI Music Recommender

Bu proje, kullanıcının yazdığı bir cümle üzerinden MBTI (Myers-Briggs Type Indicator) kişilik tipini tahmin eder ve bu tipe uygun müzik önerileri sunar. Uygulama Streamlit arayüzüyle çalışır ve eğitilmiş makine öğrenmesi modelleriyle entegredir.

![image](https://github.com/user-attachments/assets/5ba89d5b-23d0-4a91-82b3-f9fe6e539f4c)

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

### 📌 Dikkat Etmen Gerekenler:

- `ekran_goruntusu.png` dosyasını proje dizinine koy ve yukarıdaki linke uygun hale getir.
- Görsel linkini GitHub’a yüklendikten sonra düzenlemen gerekebilir.
- Streamlit arayüzünün tam sayfa bir görüntüsü en idealidir.

İstersen bu içeriği direkt sana `.md` dosyası olarak da hazırlayıp verebilirim.  
Ek olarak, görseli şimdi istersen ben senin için optimize de edebilirim. Hazır mısın?
