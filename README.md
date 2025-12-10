# 🔥 ASUFM - Wildfire Spread Prediction

Attention Swin U-Net with Focal Modulation kullanarak bir sonraki gün yangın yayılım tahmini.

## 📋 Proje Hakkında

Bu proje, Next Day Wildfire Spread (NDWS) veri seti üzerinde çalışan bir derin öğrenme modelidir. ASUFM mimarisi, Swin Transformer ve Focal Modulation tekniklerini birleştirerek yangın yayılımı tahmininde state-of-the-art performans sağlar.

**Orijinal Paper:** [IEEE CAI 2024](https://doi.org/10.1109/CAI59869.2024.00278)

**Orijinal Repo:** [bronteee/fire-asufm](https://github.com/bronteee/fire-asufm)

## 🚀 Kurulum

### Gereksinimler

- Python 3.9+
- CUDA 11.8+ (GPU eğitimi için)

### Adımlar

```bash
git clone https://github.com/KULLANICI_ADI/REPO_ADI.git
cd REPO_ADI

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 📊 Veri Seti

[Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) veri setini kullanır.

### Input Features (12 adet)

| Feature      | Açıklama                  |
| ------------ | ------------------------- |
| elevation    | Yükseklik (m)             |
| th           | Rüzgar yönü (derece)      |
| vs           | Rüzgar hızı (m/s)         |
| tmmn         | Min sıcaklık (K)          |
| tmmx         | Max sıcaklık (K)          |
| sph          | Spesifik nem              |
| pr           | Yağış (mm)                |
| pdsi         | Palmer Kuraklık İndeksi   |
| NDVI         | Bitki örtüsü indeksi      |
| population   | Nüfus yoğunluğu           |
| erc          | Enerji salınım bileşeni   |
| PrevFireMask | Önceki gün yangın maskesi |

### Output

- **FireMask**: Bir sonraki gün yangın maskesi (64x64 binary mask)

## 🏋️ Eğitim

### Lokal

```bash
python main.py --epochs 50 --batch_size 16
```

### Kaggle

```python
!git clone https://github.com/KULLANICI_ADI/REPO_ADI.git
%cd REPO_ADI
!pip install -q -r requirements.txt
!python main.py --epochs 50 --batch_size 16
```

### Argümanlar

| Argüman            | Default       | Açıklama                |
| ------------------ | ------------- | ----------------------- |
| `--epochs`         | 10            | Epoch sayısı            |
| `--batch_size`     | 16            | Batch boyutu            |
| `--seed`           | 42            | Random seed             |
| `--dir_checkpoint` | ./checkpoints | Checkpoint dizini       |
| `--skip_eval`      | False         | Validation atla         |
| `--load_model`     | None          | Önceki checkpoint yükle |

## 📈 Değerlendirme

```bash
python evaluate.py --load_model checkpoints/best_model.pth
```

## 📁 Proje Yapısı

```
├── model/
│   ├── asufm/          # ASUFM model modülleri
│   └── focalnet/       # FocalNet modülü
├── configs/            # Model konfigürasyonları
├── main.py             # Training entry point
├── train.py            # Training loop
├── evaluate.py         # Evaluation script
├── dataset.py          # PyTorch Dataset
├── data_utils.py       # Data utilities
├── config.yaml         # Hyperparameters
└── requirements.txt    # Dependencies
```

## 🎯 Beklenen Sonuçlar

| Metrik    | Hedef |
| --------- | ----- |
| F1 Score  | >0.65 |
| AUC-PR    | >0.70 |
| Precision | >0.60 |
| Recall    | >0.70 |

## 🔧 Konfigürasyon

Tüm hyperparameter'lar config.yaml dosyasından yönetilebilir.

## 📚 Referanslar

- B. Li and R. Rad, "Wildfire Spread Prediction in North America Using Satellite Imagery and Vision Transformer," IEEE CAI 2024
- [Next Day Wildfire Spread Dataset](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
- [Extended Dataset (2012-2023)](https://www.kaggle.com/datasets/bronteli/next-day-wildfire-spread-north-america-2012-2023)

## 📄 Lisans

Apache 2.0 License
