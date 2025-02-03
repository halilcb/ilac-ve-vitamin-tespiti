# İlaç ve Vitamin Sınıflandırma Projesi

Bu proje, **ilaç ve vitamin görüntülerini** kullanarak **derin öğrenme ve transfer öğrenme** teknikleri ile sınıflandırma yapmayı amaçlamaktadır. Projede veri ön işleme, veri artırma, transfer öğrenme ile model eğitimi ve performans değerlendirme aşamaları yer almaktadır.



## Veri Seti

Bu projede kullanılan veri seti, **"Pharmaceutical Drugs and Vitamins Synthetic Images"** adıyla popüler olan ilaç ve vitaminlerin görüntülerinden oluşmaktadır.

### **Veri Seti Özellikleri:**
- **Görüntü Sayısı:** 10 farklı sınıfta toplamda 10,000 görüntü
- **Görüntü Formatı:** JPG ve PNG
- **Sınıflar:** 
  - Alaxan
  - Bactidol
  - Bioflu
  - Biogesic
  - DayZinc
  - Decolgen
  - Fish Oil
  - Kremil S
  - Medicol
  - Neozep



## Proje Aşamaları

### 1. Veri Yükleme ve Görselleştirme
- Görüntüler, dosya yolları ve etiketler birleştirilerek bir veri çerçevesi oluşturulmuştur.
- Rastgele seçilen 25 görüntü görselleştirilmiştir.

### 2. Veri Ön İşleme
- **Train-Test Split:** Veriler %80 eğitim, %20 test olarak bölünmüştür.
- **Veri Artırma (Augmentation):** Dönme, yakınlaştırma, yatay/dikey çevirme gibi tekniklerle veri artırımı yapılmıştır.
- **Yeniden Boyutlandırma:** Görüntüler 224x224 piksele dönüştürülmüştür.
- **Normalizasyon:** Piksel değerleri 0-1 aralığına ölçeklendirilmiştir.

### 3. Model Eğitimi (Transfer Learning)
- **Önceden Eğitilmiş Model:** MobileNetV2 kullanılmıştır.
- Modelin ağırlıkları ImageNet ile önceden eğitilmiş olup son katmanlar yeniden eğitilmiştir.
- Model mimarisi:
  - Dense katmanlar (256 nöron, ReLU aktivasyonu)
  - Dropout (aşırı öğrenmeyi engellemek için %20)
  - Son katmanda softmax aktivasyonu (10 sınıf)

### 4. Modelin Eğitilmesi
- **Optimizasyon:** Adam optimizer (0.0001 öğrenme oranı)
- **Kayıp Fonksiyonu:** Categorical Crossentropy
- **Epoch Sayısı:** 50
- **Early Stopping:** Aşırı öğrenmeyi önlemek için erken durdurma uygulanmıştır.
- **Checkpoint:** En iyi model ağırlıkları kaydedilmiştir.


## Model Performansı ve Değerlendirme

### **1. Test Sonuçları:**
- **Test Loss:** 0.43
- **Test Accuracy:** %85.6

### **2. Eğitim ve Doğrulama Sonuçları:**
| Epoch | Eğitim Doğruluğu (%) | Doğrulama Doğruluğu (%) | Eğitim Kaybı | Doğrulama Kaybı |
|-------|----------------------|-------------------------|--------------|-----------------|
| 1     | 33.16                | 76.38                   | 1.93         | 0.80            |
| 5     | 83.74                | 83.94                   | 0.47         | 0.46            |
| 10    | 92.24                | 85.44                   | 0.24         | 0.43            |
| 15    | 96.23                | 86.12                   | 0.12         | 0.41            |

### **3. Sınıflandırma Raporu:**

| Sınıf      | Doğruluk (Precision) | Duyarlılık (Recall) | F1-Score |
|------------|----------------------|---------------------|----------|
| Alaxan     | 0.85                 | 0.84                | 0.84     |
| Bactidol   | 0.85                 | 0.81                | 0.83     |
| Bioflu     | 0.88                 | 0.88                | 0.88     |
| Biogesic   | 0.81                 | 0.78                | 0.80     |
| DayZinc    | 0.95                 | 0.86                | 0.90     |
| Decolgen   | 0.92                 | 0.84                | 0.88     |
| Fish Oil   | 0.87                 | 0.94                | 0.90     |
| Kremil S   | 0.75                 | 0.83                | 0.79     |
| Medicol    | 0.93                 | 0.94                | 0.94     |
| Neozep     | 0.76                 | 0.82                | 0.79     |

- **Genel Doğruluk:** %86
- **Makro Ortalama F1-Score:** 0.86



## Sonuç

Bu proje kapsamında ilaç ve vitamin görüntülerini sınıflandırmak için bir derin öğrenme modeli oluşturulmuştur. Model, genel olarak %85 doğruluk oranı ile başarılı bir performans sergilemiş olup, bazı sınıflarda daha yüksek doğruluk değerleri elde edilmiştir.
