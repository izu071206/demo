# 📖 Hướng Dẫn Chi Tiết - Từng Bước

## Mục Lục

1. [Chuẩn Bị Môi Trường](#1-chuẩn-bị-môi-trường)
2. [Cài Đặt Dependencies](#2-cài-đặt-dependencies)
3. [Chuẩn Bị Dữ Liệu](#3-chuẩn-bị-dữ-liệu)
4. [Generate Dataset](#4-generate-dataset)
5. [Train Models](#5-train-models)
6. [Evaluate Models](#6-evaluate-models)
7. [Chạy Dashboard](#7-chạy-dashboard)
8. [Test với Files Mới](#8-test-với-files-mới)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Chuẩn Bị Môi Trường

### 1.1 Kiểm Tra Python

```bash
# Kiểm tra version
python --version
# Cần Python 3.8 trở lên

# Kiểm tra pip
pip --version
```

### 1.2 Tạo Virtual Environment (Khuyến Nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Cài Đặt Dependencies

### 2.1 Cài Đặt Core Dependencies

```bash
# Di chuyển vào thư mục dự án
cd demo2

# Cài đặt dependencies
pip install -r requirements.txt
```

**Nếu gặp lỗi**, thử:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 2.2 Cài Đặt Neural Network Support (Tùy Chọn)

```bash
# Chỉ cần nếu muốn train Neural Network
pip install -r requirements-dl.txt
```

**Lưu ý**: PyTorch có thể cần cài đặt riêng tùy theo hệ điều hành.

### 2.3 Kiểm Tra Cài Đặt

```bash
python scripts/check_environment.py
```

---

## 3. Chuẩn Bị Dữ Liệu

### 3.1 Tạo Cấu Trúc Thư Mục

```bash
python scripts/create_sample_structure.py
```

### 3.2 Thêm Benign Samples

**Cách 1: Copy Files Thủ Công**

```bash
# Windows PowerShell
copy C:\Windows\System32\notepad.exe data\benign\
copy C:\Windows\System32\calc.exe data\benign\

# Linux/macOS
cp /usr/bin/ls data/benign/
cp /usr/bin/cat data/benign/
```

**Cách 2: Download Samples**

```bash
# Sử dụng script (nếu có)
python scripts/download_samples.py --tag packed --limit 25
```

**Lưu ý**: 
- Cần ít nhất 10-20 files benign
- Files phải là binary (.exe, .dll, .bin)
- Kích thước tối thiểu: 100 bytes

### 3.3 Thêm Obfuscated Samples

**Cách 1: Copy Files Thủ Công**

```bash
# Copy các file obfuscated/malware vào
copy C:\path\to\obfuscated.exe data\obfuscated\
```

**Cách 2: Sử Dụng Script**

```bash
# Chỉ copy, không obfuscate thật
python scripts/obfuscate_samples.py \
    --source data/benign/ \
    --output data/obfuscated/ \
    --method copy
```

**Lưu ý**: 
- Cần ít nhất 10-20 files obfuscated
- ⚠️ **QUAN TRỌNG**: Chỉ test malware trong VM!

### 3.4 Kiểm Tra Dữ Liệu

```bash
# Kiểm tra số lượng files
# Windows
dir data\benign\ /b | find /c ".exe"
dir data\obfuscated\ /b | find /c ".exe"

# Linux/macOS
ls data/benign/*.exe | wc -l
ls data/obfuscated/*.exe | wc -l
```

---

## 4. Generate Dataset

### 4.1 Chạy Dataset Generation

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

### 4.2 Quá Trình Xử Lý

Script sẽ:

1. **Scan Files**: Tìm tất cả binary files trong `data/benign/` và `data/obfuscated/`
2. **Validate PE**: Kiểm tra file có phải PE hợp lệ không
3. **Extract Features**: 
   - Opcode n-grams (2, 3, 4-gram)
   - API calls
   - CFG metrics (nếu enabled)
   - Metadata
4. **Family Detection**: Xác định family từ tên thư mục
5. **Strict Split**: Chia train/val/test theo family (không có leakage)
6. **Save Results**: Lưu features và metadata

### 4.3 Kết Quả Mong Đợi

```
INFO - Starting dataset generation...
INFO - Processing 50 valid binary files from data/benign/
INFO - Processing 50 valid binary files from data/obfuscated/
INFO - Found 20 unique families for group splitting
INFO - Dataset generated:
INFO -   Train: 70 samples
INFO -   Val: 15 samples
INFO -   Test: 15 samples
INFO -   Feature dimension: 3500
INFO -   Train families: 14
INFO -   Val families: 3
INFO -   Test families: 3
INFO - ✓ No family leakage detected in final splits
```

### 4.4 Files Được Tạo

- `data/processed/train_features.pkl` - Training features và labels
- `data/processed/val_features.pkl` - Validation features và labels
- `data/processed/test_features.pkl` - Test features và labels
- `data/processed/feature_metadata.json` - **CRITICAL**: Feature schema và config
- `data/processed/sample_metadata.csv` - Metadata cho từng sample

### 4.5 Thời Gian Xử Lý

**Ước tính**:
- 10-20 files: 2-5 phút
- 50-100 files: 10-30 phút
- 200+ files: 30-60 phút

**Tối ưu hóa**:
- Disable CFG nếu không cần: `enable_cfg: false` trong `config/dataset_config.yaml`
- Giảm `max_features` cho opcode n-grams

### 4.6 Kiểm Tra Kết Quả

```bash
# Kiểm tra files đã được tạo
# Windows
dir data\processed\

# Linux/macOS
ls -lh data/processed/
```

**Phải có**:
- ✅ `train_features.pkl`
- ✅ `val_features.pkl`
- ✅ `test_features.pkl`
- ✅ `feature_metadata.json` (QUAN TRỌNG!)
- ✅ `sample_metadata.csv`

---

## 5. Train Models

### 5.1 Cấu Hình Training

Chỉnh sửa `config/train_config.yaml` nếu cần:

```yaml
training:
  # Data paths
  train_data: "data/processed/train_features.pkl"
  val_data: "data/processed/val_features.pkl"
  test_data: "data/processed/test_features.pkl"
  
  # Models to train
  models:
    - random_forest
    - xgboost
    # - neural_network  # Uncomment nếu muốn train NN
  
  # Preprocessing
  preprocessing: standard  # Options: 'standard', 'minmax', 'none'
  
  # Random Forest
  random_forest:
    n_estimators: 100
    max_depth: 20
    random_state: 42
    class_weight: balanced
  
  # XGBoost
  xgboost:
    n_estimators: 100
    max_depth: 6
    random_state: 42
    scale_pos_weight: auto  # Sẽ tự tính từ data
```

### 5.2 Chạy Training

```bash
python src/models/train.py
```

### 5.3 Quá Trình Training

Script sẽ:

1. **Load Data**: Load train/val/test từ pickle files
2. **Check Imbalance**: Phân tích class distribution
3. **Apply Preprocessing**: StandardScaler (nếu enabled)
4. **Train Models**: 
   - Random Forest
   - XGBoost
   - Neural Network (nếu enabled)
5. **Log Feature Importance**: Top features cho tree-based models
6. **Check Overfitting**: So sánh train vs validation accuracy
7. **Evaluate**: Test trên test set
8. **Save Models**: Lưu models và scaler

### 5.4 Kết Quả Mong Đợi

```
============================================================
Starting model training
============================================================
Loading data splits...
Loaded 70 samples from data/processed/train_features.pkl
Loaded 15 samples from data/processed/val_features.pkl
Loaded 15 samples from data/processed/test_features.pkl
Feature dimension: 3500

============================================================
Class Distribution Analysis
============================================================
Train - Class 0: 35 samples (50.0%)
Train - Class 1: 35 samples (50.0%)
Train - Imbalance ratio (neg/pos): 1.00

============================================================
Applying Preprocessing
============================================================
Applied StandardScaler to training data
Applied StandardScaler to validation data
Applied StandardScaler to test data
Saved scaler to models/scaler.pkl

============================================================
Training Model: RANDOM_FOREST
============================================================
Training RandomForest...
Training accuracy: 0.9857
Validation accuracy: 0.9333
✓ No overfitting detected. Train score: 0.9857, Val score: 0.9333, Gap: 0.0524
RandomForest - Top 20 most important features:
  1. Feature 1234: 0.023456
  2. Feature 567: 0.019876
  ...
✓ Model saved to models/random_forest_model.pkl
Test Results:
  Accuracy: 0.9333
  Precision: 0.9286
  Recall: 0.9286
  F1 Score: 0.9286

============================================================
Training Model: XGBOOST
============================================================
...
```

### 5.5 Files Được Tạo

- `models/random_forest_model.pkl` - Random Forest model
- `models/xgboost_model.json` - XGBoost model
- `models/scaler.pkl` - **CRITICAL**: Scaler cho preprocessing (nếu dùng)
- `data/evaluation_results/random_forest_metrics.json` - Metrics
- `data/evaluation_results/xgboost_metrics.json` - Metrics

### 5.6 Thời Gian Training

**Ước tính** (100 samples, feature_dim=3500):

| Model | Thời Gian | Ghi Chú |
|-------|-----------|---------|
| Random Forest | 1-3 phút | Fast, CPU-only |
| XGBoost | 2-5 phút | Fast với early stopping |
| Neural Network | 10-30 phút | Cần GPU để nhanh |

**Tối ưu hóa**:
- Giảm `n_estimators` cho tree-based models
- Giảm `epochs` cho Neural Network
- Sử dụng GPU cho Neural Network

### 5.7 Kiểm Tra Kết Quả

```bash
# Kiểm tra models đã được tạo
# Windows
dir models\

# Linux/macOS
ls -lh models/
```

**Phải có**:
- ✅ `random_forest_model.pkl`
- ✅ `xgboost_model.json`
- ✅ `scaler.pkl` (nếu dùng preprocessing)

---

## 6. Evaluate Models

### 6.1 Chạy Evaluation

```bash
python src/evaluation/evaluate.py \
    --model models/random_forest_model.pkl \
    --model_type random_forest \
    --test_data data/processed/test_features.pkl \
    --output_dir data/evaluation_results/
```

### 6.2 Kết Quả

Evaluation sẽ tạo:
- Metrics JSON file
- Confusion matrix (nếu có visualization)
- ROC curve (nếu có visualization)

---

## 7. Chạy Dashboard

### 7.1 Chọn Dashboard

**Option 1: Basic Dashboard** (Đơn giản, nhanh)

```bash
python src/dashboard/app.py
```

**Option 2: Enhanced Dashboard** (Nhiều tính năng hơn)

```bash
python src/dashboard/app_with_dynamic_models.py
```

### 7.2 Cấu Hình Dashboard

Chỉnh sửa `config/inference_config.yaml`:

```yaml
inference:
  model_type: "random_forest"
  model_path: "models/random_forest_model.pkl"
  feature_metadata: "data/processed/feature_metadata.json"
  scaler_path: "models/scaler.pkl"  # CRITICAL: Nếu dùng preprocessing
  enable_explainability: true
  top_features: 5
```

### 7.3 Truy Cập Dashboard

Mở browser: **http://localhost:5000**

### 7.4 Sử Dụng Dashboard

1. **Upload File**: Chọn file binary (.exe, .dll, .bin)
2. **Select Models**: Chọn models để predict (nếu multi-model)
3. **Analyze**: Xem kết quả prediction với confidence
4. **View Details**: Xem probabilities và top features (nếu enabled)

### 7.5 API Endpoints

Xem [README.md](README.md) để biết danh sách đầy đủ API endpoints.

---

## 8. Test với Files Mới

### 8.1 Sử Dụng Dashboard

Upload file qua web interface tại http://localhost:5000

### 8.2 Sử Dụng Command Line

```bash
# Sử dụng debug script
python debug.py --test-predict path/to/file.exe --model-type random_forest
```

### 8.3 Sử Dụng Inference Pipeline Trực Tiếp

```python
from src.pipeline.inference_pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    model_path="models/random_forest_model.pkl",
    model_type="random_forest",
    feature_metadata="data/processed/feature_metadata.json",
    scaler_path="models/scaler.pkl"  # Nếu dùng preprocessing
)

# Predict
result = pipeline.predict_file("path/to/file.exe")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

---

## 9. Troubleshooting

### 9.1 Lỗi: "No valid binary files found"

**Nguyên nhân**: Chưa thêm binary files

**Giải pháp**:
```bash
# Kiểm tra thư mục
dir data\benign\
dir data\obfuscated\

# Thêm files nếu trống
```

### 9.2 Lỗi: "Feature metadata not found"

**Nguyên nhân**: Chưa generate dataset

**Giải pháp**:
```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

### 9.3 Lỗi: "Model expects X features, pipeline expects Y"

**Nguyên nhân**: Model được train với schema khác

**Giải pháp**:
- Retrain models với dataset mới
- Hoặc sử dụng đúng feature_metadata từ training

### 9.4 Lỗi: "Dimension mismatch"

**Nguyên nhân**: Scaler được train với dimension khác

**Giải pháp**:
- Đảm bảo scaler được tạo cùng với models
- Hoặc retrain với preprocessing

### 9.5 Lỗi: "All models failed to initialize"

**Nguyên nhân**: 
- Models không tồn tại
- Feature metadata không đúng
- Scaler path không đúng

**Giải pháp**:
1. Kiểm tra models có trong `models/` không
2. Kiểm tra `feature_metadata.json` có tồn tại không
3. Kiểm tra `scaler.pkl` có tồn tại không (nếu dùng preprocessing)
4. Xem logs để biết lỗi chi tiết

### 9.6 Dataset Generation Quá Chậm

**Giải pháp**:
- Disable CFG extraction: `enable_cfg: false` trong config
- Giảm `max_features` cho opcode n-grams
- Sử dụng SSD thay vì HDD

### 9.7 Training Quá Chậm

**Giải pháp**:
- Giảm `n_estimators` cho tree-based models
- Giảm `epochs` cho Neural Network
- Sử dụng GPU cho Neural Network
- Giảm feature dimension (nếu có thể)

---

## ✅ Checklist Hoàn Chỉnh

### Trước Khi Bắt Đầu

- [ ] Python 3.8+ đã cài
- [ ] Dependencies đã cài (`pip install -r requirements.txt`)
- [ ] Cấu trúc thư mục đã tạo
- [ ] Đã thêm binary samples vào `data/benign/` và `data/obfuscated/`

### Sau Khi Generate Dataset

- [ ] `train_features.pkl` đã được tạo
- [ ] `val_features.pkl` đã được tạo
- [ ] `test_features.pkl` đã được tạo
- [ ] `feature_metadata.json` đã được tạo (QUAN TRỌNG!)
- [ ] `sample_metadata.csv` đã được tạo
- [ ] Không có family leakage (check logs)

### Sau Khi Train Models

- [ ] `random_forest_model.pkl` đã được tạo
- [ ] `xgboost_model.json` đã được tạo
- [ ] `scaler.pkl` đã được tạo (nếu dùng preprocessing)
- [ ] Metrics files đã được tạo trong `data/evaluation_results/`
- [ ] Models có accuracy hợp lý (>70%)

### Sau Khi Chạy Dashboard

- [ ] Dashboard khởi động không lỗi
- [ ] Models được load thành công
- [ ] Có thể upload và predict files
- [ ] Kết quả hiển thị đúng

---

## 📊 Tổng Kết Thời Gian

### Workflow Hoàn Chỉnh (100 samples)

| Bước | Thời Gian | Ghi Chú |
|------|-----------|---------|
| Chuẩn bị dữ liệu | 5-10 phút | Tùy thuộc số lượng files |
| Generate dataset | 10-30 phút | Tùy thuộc CFG extraction |
| Train models | 5-15 phút | Random Forest + XGBoost |
| Evaluate | 1-2 phút | Quick |
| **Tổng cộng** | **20-60 phút** | Tùy thuộc hardware |

### Workflow Hoàn Chỉnh (1000 samples)

| Bước | Thời Gian | Ghi Chú |
|------|-----------|---------|
| Chuẩn bị dữ liệu | 30-60 phút | Tùy thuộc số lượng files |
| Generate dataset | 60-120 phút | Tùy thuộc CFG extraction |
| Train models | 30-60 phút | Random Forest + XGBoost |
| Evaluate | 5-10 phút | Quick |
| **Tổng cộng** | **2-4 giờ** | Tùy thuộc hardware |

---

## 💡 Tips & Best Practices

1. **Bắt đầu nhỏ**: Test với 10-20 files mỗi loại trước
2. **Disable CFG**: Nếu không cần, disable CFG để tăng tốc
3. **Backup**: Backup models và results quan trọng
4. **VM cho malware**: Luôn test malware trong VM
5. **Monitor logs**: Xem logs để biết lỗi chi tiết
6. **Check consistency**: Verify models không mâu thuẫn

---

## 🆘 Cần Giúp Đỡ?

1. Xem logs để biết lỗi chi tiết
2. Kiểm tra các file trong `docs/` để biết thêm
3. Đảm bảo đã làm đúng các bước trên
4. Xem [README.md](README.md) để biết tổng quan

---

**Chúc bạn thành công! 🎉**

