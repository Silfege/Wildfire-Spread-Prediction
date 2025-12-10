# Fire-ASUFM Proje Analizi

## 1. DOSYA YAPISI

```
fire-asufm/
├── configs/
│   └── asufm.py                    # Model konfigürasyon dosyaları
├── data_utils.py                    # TFRecord parsing ve data preprocessing utilities
├── dataset.py                       # PyTorch Dataset sınıfı (NextDayFireDataset)
├── evaluate.py                      # Model değerlendirme scripti
├── main.py                          # Ana training scripti
├── metrics_loss.py                  # Loss functions ve metrikler
├── train.py                         # Training loop ve evaluation fonksiyonları
├── requirements.txt                 # Python dependencies
├── README.md                        # Proje dokümantasyonu
├── LICENSE                          # Lisans dosyası
└── model/
    ├── __init__.py
    ├── README.md
    ├── asufm/                       # ASUFM model implementasyonu
    │   ├── __init__.py
    │   ├── asufm.py                 # Ana model sınıfı (ASUFM)
    │   ├── encoder.py                # Encoder modülü
    │   ├── decoder.py                # Decoder modülü
    │   ├── embedding.py              # Patch embedding
    │   ├── attention.py              # Window attention mekanizması
    │   ├── swin_block.py             # Swin Transformer blokları
    │   ├── partitioning.py          # Window partitioning
    │   └── skipconnection/
    │       └── crossvit.py           # Cross attention skip connections
    └── focalnet/
        ├── __init__.py
        └── focalnet.py               # FocalNet implementasyonu (kullanılmıyor gibi görünüyor)
```

---

## 2. DEPENDENCIES

### requirements.txt İçeriği:

```
torchmetrics          # Versiyon belirtilmemiş
tensorboard           # Versiyon belirtilmemiş
scipy                 # Versiyon belirtilmemiş
timm==0.4.12          # ✅ Versiyon belirtilmiş
yacs==0.1.8           # ✅ Versiyon belirtilmiş
wandb                 # Versiyon belirtilmemiş
einops                # Versiyon belirtilmemiş
pytorch_warmup        # Versiyon belirtilmemiş
ml_collections        # Versiyon belirtilmemiş
```

### Eksik Dependencies (Kodda kullanılıyor ama requirements.txt'de yok):
- **torch** (PyTorch) - Ana framework
- **tensorflow** - TFRecord okuma için
- **numpy** - Array işlemleri
- **matplotlib** - Görselleştirme (evaluate.py'de)
- **tqdm** - Progress bar

### Öneriler:
1. Eksik paketlerin eklenmesi gerekiyor
2. Versiyonların belirtilmesi önerilir (reproducibility için)
3. TensorFlow ve PyTorch versiyonları uyumluluğu kontrol edilmeli

---

## 3. MODEL MİMARİSİ

### Ana Model Sınıfı
- **Dosya**: `model/asufm/asufm.py`
- **Sınıf Adı**: `ASUFM` (Attention Swin U-Net with Focal Modulation)

### Model Yapısı:
```
ASUFM
├── PatchEmbed (Embedding Layer)
├── Encoder
│   ├── 4 BasicLayer (her biri 2 SwinTransformerBlock içerir)
│   └── PatchMerging (downsampling)
└── Decoder
    ├── 4 BasicLayer_up (upsampling)
    ├── PatchExpand (upsampling)
    └── FinalPatchExpand_X4 (final upsampling)
    └── Conv2d (output layer)
```

### Input/Output Shape:

**Input Shape:**
- `(batch_size, n_channels, 64, 64)`
- `n_channels`: Config'de `in_chans` parametresi ile belirlenir
  - `get_asfum_6_configs()`: `in_chans = 6`
  - `get_asufm_12_configs()`: `in_chans = 12`
- Default kullanım: **6 kanal** (main.py'de `limit_features` listesi 6 öğe içeriyor)

**Output Shape:**
- `(batch_size, 1, 64, 64)` - Binary segmentation mask
- `num_classes = 1` (binary classification)

### Model Konfigürasyonları (`configs/asufm.py`):

#### `get_asfum_6_configs()` (Default kullanılan):
```python
image_size = 64
patch_size = 4
num_classes = 1
in_chans = 6
embed_dim = 96
depths = [2, 2, 2, 2]          # Her stage'de 2 blok
num_heads = [3, 6, 12, 24]     # Her stage'de attention head sayısı
window_size = 8
mlp_ratio = 4
drop_rate = 0.0
drop_path_rate = 0.1
focal = True                   # ✅ Focal Modulation aktif
use_checkpoint = True          # Memory optimization
```

#### Diğer Konfigürasyonlar:
- `get_swin_unet_attention_configs()`: Focal olmadan, 6 kanal
- `get_ca_swin_unet_attention_configs()`: Cross attention ile, 12 kanal
- `get_asufm_12_configs()`: 12 kanal, focal=True

---

## 4. DATA PIPELINE

### Dataset Sınıfı: `NextDayFireDataset`

**Dosya**: `dataset.py`

**Özellikler:**
- TensorFlow TFRecord formatından veri okur
- PyTorch Dataset interface'i implement eder
- Feature selection, normalization ve sampling yapar

### TFRecord Parsing Mantığı:

1. **Feature Dictionary Oluşturma** (`data_utils.py`):
   - `_get_features_dict()`: Her feature için `FixedLenFeature` oluşturur
   - Shape: `[64, 64]` (square tiles)
   - Dtype: `tf.float32`

2. **Feature Listesi** (`data_utils.py`):
   ```python
   INPUT_FEATURES = [
       'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 
       'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask'
   ]
   
   OUTPUT_FEATURES = ['FireMask']
   ```

3. **Parsing İşlemi** (`dataset.py`):
   - `tf.io.parse_single_example()` ile TFRecord parse edilir
   - `limit_features_list` ile sadece seçilen feature'lar alınır
   - `_clip_and_normalize()` ile normalization yapılır

### Beklenen Feature Sayısı ve İsimleri:

**Default kullanılan features** (`main.py`):
```python
limit_features = [
    'elevation',    # Yükseklik
    'th',           # Wind direction (degrees)
    'sph',          # Specific humidity
    'pr',           # Precipitation (mm)
    'NDVI',         # Normalized Difference Vegetation Index
    'PrevFireMask', # Önceki günün yangın maskesi
]
# Toplam: 6 feature
```

**Tüm mevcut features** (12 adet):
1. `elevation` - Yükseklik (m)
2. `th` - Wind direction (degrees)
3. `vs` - Wind speed (m/s)
4. `tmmn` - Min temperature (Kelvin)
5. `tmmx` - Max temperature (Kelvin)
6. `sph` - Specific humidity
7. `pr` - Precipitation (mm)
8. `pdsi` - Palmer Drought Severity Index
9. `NDVI` - Normalized Difference Vegetation Index
10. `population` - Population density
11. `erc` - Energy Release Component (fire danger index)
12. `PrevFireMask` - Previous day fire mask

### Data Preprocessing:

1. **Clipping & Normalization** (`data_utils.py`):
   - Her feature için `DATA_STATS` dictionary'sinde min/max/mean/std değerleri var
   - `_clip_and_normalize()`: Clip + z-score normalization
   - `_clip_and_rescale()`: Clip + min-max rescaling (0-1)

2. **Sampling Methods** (`dataset.py`):
   - `'random_crop'`: Random 32x32 crop (default)
   - `'center_crop'`: Center crop (16 pixel margin)
   - `'downsample'`: 2x downsampling
   - `'original'`: Orijinal 64x64 boyut

3. **Target Processing**:
   - Binary mask: `> 0` değerleri `1`, diğerleri `0`
   - Shape: `(1, H, W)`

---

## 5. TRAINING

### main.py ve train.py Rolleri:

**main.py** (Entry Point):
- Argument parsing (seed, checkpoint dir, epochs, batch_size)
- Dataset path'lerini tanımlar
- Model initialization
- Checkpoint loading
- `train_next_day_fire()` fonksiyonunu çağırır

**train.py** (Training Logic):
- `train_next_day_fire()`: Ana training loop
- `evaluate()`: Validation evaluation
- `load_checkpoint()`: Checkpoint yükleme
- `seed_all()`: Reproducibility için seed ayarlama

### Loss Function:

**Kullanılan Loss Functions:**
1. **BCEWithLogitsLoss** (default, `loss_function='bce'`):
   - `pos_weight=3.0` (imbalanced data için)
   - Binary cross-entropy with logits

2. **FocalTverskyLoss** (`loss_function='ft'`):
   - `alpha=0.75, beta=0.25, gamma=1`
   - Imbalanced segmentation için

### Default Hyperparameters:

```python
# main.py
seed = 42
epochs = 10
batch_size = 16
learning_rate = 0.0001
img_scale = 0.5                    # Kullanılmıyor gibi görünüyor
val_percent = 20                  # Kullanılmıyor (dataset ayrı)
amp = True                        # Mixed precision training
optimizer = 'adamw'
pos_weight = 3.0
loss_function = 'bce'
activation = 'relu'               # Kullanılmıyor gibi görünüyor
sampling = 'original'
skip_eval = True                  # ⚠️ Validation skip ediliyor!
use_checkpointing = False

# train.py (içeride)
weight_decay = 0.05
momentum = 0.9                    # SGD için
gradient_clipping = 1.0
accum_iter = 16                   # Gradient accumulation
val_batch_size = 64               # Validation için
use_warmup = True
warmup_lr_init = 5e-7
```

### Optimizer:

- **AdamW** (default): `lr=0.0001, weight_decay=0.05, amsgrad=True`
- Alternatifler: `adam`, `rmsprop`, `sgd`

### Learning Rate Schedule:

- **CosineAnnealingLR**: `T_max = num_steps` (total training steps)
- **Linear Warmup**: `pytorch_warmup.UntunedLinearWarmup`

### Training Features:

- ✅ Mixed Precision Training (AMP)
- ✅ Gradient Accumulation (16 iter)
- ✅ Gradient Clipping (1.0)
- ✅ Checkpointing (memory optimization, optional)
- ✅ Wandb logging
- ⚠️ `skip_eval=True` default - Validation çalışmıyor!

---

## 6. CONFIG

### configs/asufm.py

**Değiştirilebilir Parametreler:**

#### Model Architecture:
- `image_size`: Input image size (default: 64)
- `patch_size`: Patch size for embedding (default: 4)
- `in_chans`: Input channels (6 veya 12)
- `embed_dim`: Embedding dimension (default: 96)
- `depths`: Number of blocks per stage (default: [2,2,2,2])
- `num_heads`: Attention heads per stage (default: [3,6,12,24])
- `window_size`: Window size for attention (default: 8)

#### Regularization:
- `drop_rate`: Dropout rate (default: 0.0)
- `drop_path_rate`: Stochastic depth rate (default: 0.1)
- `attn_drop_rate`: Attention dropout (default: 0)

#### Advanced:
- `mlp_ratio`: MLP expansion ratio (default: 4)
- `qkv_bias`: Query/Key/Value bias (default: True)
- `ape`: Absolute position embedding (default: False)
- `patch_norm`: Patch normalization (default: True)
- `use_checkpoint`: Gradient checkpointing (default: True/False)
- `focal`: Focal modulation (default: True/False)
- `mode`: Attention mode ("swin" veya "cross_contextual_attention")
- `skip_num`: Skip connection sayısı (1, 2, veya 3)
- `spatial_attention`: Spatial attention kullanımı ('1' veya '0')

---

## 7. POTANSIYEL SORUNLAR

### ⚠️ Kritik Sorunlar:

1. **Hardcoded Dataset Paths**:
   ```python
   # main.py (satır 69, 73)
   '../dataset/next-day-fire/next_day_wildfire_spread_train_*.tfrecord'
   '../dataset/next-day-fire/next_day_wildfire_spread_eval_*.tfrecord'
   
   # evaluate.py (satır 41-58)
   '../dataset/next-day-fire-2012-2023/...'
   ```
   **Sorun**: Kaggle'da bu path'ler çalışmayabilir
   **Çözüm**: Environment variable veya config file kullanılmalı

2. **Deprecated TensorFlow API**:
   ```python
   # data_utils.py (satır 280, 282, 294, 298)
   tf.data.experimental.AUTOTUNE
   ```
   **Sorun**: TensorFlow 2.x'de `tf.data.AUTOTUNE` kullanılmalı
   **Çözüm**: `tf.data.experimental.AUTOTUNE` → `tf.data.AUTOTUNE`

3. **Validation Skip Ediliyor**:
   ```python
   # main.py (satır 64)
   skip_eval = True  # ⚠️ Default olarak validation çalışmıyor!
   ```
   **Sorun**: Model performansı takip edilemiyor
   **Çözüm**: `skip_eval = False` yapılmalı veya argüman olarak eklenmeli

4. **Eksik Dependencies**:
   - `torch`, `tensorflow`, `numpy`, `matplotlib`, `tqdm` requirements.txt'de yok
   - Versiyonlar belirtilmemiş (reproducibility sorunu)

### ⚠️ Orta Öncelikli Sorunlar:

5. **Wandb Entity Hardcoded**:
   ```python
   # train.py (satır 121)
   entity='fire-dream'  # ⚠️ Hardcoded entity
   ```
   **Sorun**: Farklı kullanıcılar için çalışmayabilir
   **Çözüm**: Environment variable veya argüman olarak eklenmeli

6. **Checkpoint Path Logic**:
   ```python
   # main.py (satır 40-48)
   # Son checkpoint'i otomatik buluyor ama hata yönetimi zayıf
   ```
   **Sorun**: IndexError durumunda program crash olabilir
   **Çözüm**: Try-except zaten var ama daha iyi hata mesajı verilebilir

7. **Device Compatibility**:
   ```python
   # train.py (satır 254)
   device.type if device.type != 'mps' else 'cpu'
   ```
   **Sorun**: MPS (Apple Silicon) için CPU'ya fallback yapıyor
   **Not**: Bu aslında bir çözüm, sorun değil

8. **Memory Format**:
   ```python
   # main.py (satır 88)
   model = model.to(memory_format=torch.channels_last)
   ```
   **Not**: Bu optimizasyon, sorun değil

### ℹ️ İyileştirme Önerileri:

9. **Config Management**:
   - Hyperparameter'lar main.py'de hardcoded
   - YAML veya JSON config file kullanılabilir

10. **Logging**:
    - Bazı print statement'lar var, logging kullanılabilir
    - TensorBoard logging eksik (requirements.txt'de var ama kullanılmıyor)

11. **Error Handling**:
    - Dataset loading'de hata yönetimi eksik
    - TFRecord file bulunamazsa açık hata mesajı yok

12. **Code Duplication**:
    - `main.py` ve `evaluate.py`'de benzer dataset path tanımlamaları var
    - Ortak bir config/utility fonksiyonu kullanılabilir

### ✅ Kaggle Uyumluluğu:

**Çalışacak Kısımlar:**
- Model mimarisi
- Training loop
- PyTorch kodları
- TFRecord parsing

**Düzeltilmesi Gerekenler:**
1. Dataset path'lerini Kaggle input path'ine göre ayarlamak
2. `tf.data.experimental.AUTOTUNE` → `tf.data.AUTOTUNE`
3. Wandb entity'yi kaldırmak veya config'den almak
4. Eksik dependencies'i requirements.txt'ye eklemek

---

## ÖZET

### Güçlü Yönler:
- ✅ Modern model mimarisi (Swin Transformer + Focal Modulation)
- ✅ İyi organize edilmiş kod yapısı
- ✅ Mixed precision training desteği
- ✅ Gradient accumulation ve clipping
- ✅ Checkpointing ve resume training
- ✅ Comprehensive metrics (Dice, AUC, Precision, Recall, F1)

### Zayıf Yönler:
- ⚠️ Hardcoded path'ler
- ⚠️ Deprecated TensorFlow API
- ⚠️ Eksik dependencies
- ⚠️ Default olarak validation skip ediliyor
- ⚠️ Config management eksik

### Kaggle için Gerekli Değişiklikler:
1. Dataset path'lerini `/kaggle/input/...` formatına çevir
2. `tf.data.experimental.AUTOTUNE` → `tf.data.AUTOTUNE`
3. Requirements.txt'ye eksik paketleri ekle
4. Wandb entity'yi config'den al veya kaldır
5. `skip_eval=False` yap veya argüman ekle

