# 🛡️ Hệ Thống Phát Hiện Obfuscation trong Malware bằng Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Hệ thống phân tích malware/obfuscation **đúng – ổn định – nhất quán – tái lập được**, sử dụng machine learning để phát hiện các kỹ thuật obfuscation trong mã độc.

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Tính Năng Chính](#-tính-năng-chính)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Các Cập Nhật Mới](#-các-cập-nhật-mới)
- [Thời Gian Xử Lý](#-thời-gian-xử-lý)
- [API & Dashboard](#-api--dashboard)
- [Tài Liệu](#-tài-liệu)

---

## 🎯 Tổng Quan

Dự án này xây dựng một pipeline hoàn chỉnh để phân tích và phát hiện obfuscation trong malware, bao gồm:

- **Feature Extraction**: Trích xuất features từ binary files (opcode, API calls, CFG metrics)
- **Machine Learning**: Training và evaluation với Random Forest, XGBoost, Neural Network
- **Inference Pipeline**: Dự đoán với tính deterministic và consistency checks
- **Web Dashboard**: Giao diện web để upload và phân tích files

### ✨ Điểm Nổi Bật

- ✅ **Deterministic**: Cùng file → cùng kết quả (100% reproducible)
- ✅ **Consistent**: Models không mâu thuẫn, có consistency checks
- ✅ **Fixed Schema**: Feature dimensions bất biến giữa train/test/inference
- ✅ **Strict Splitting**: Family-based split để tránh data leakage
- ✅ **Production Ready**: Đủ tiêu chuẩn nghiên cứu hoặc triển khai thực tế

---

## 🚀 Tính Năng Chính

### 1. Feature Extraction

- **Opcode N-grams**: 2-gram, 3-gram, 4-gram từ disassembly
- **API Calls**: Static analysis của imports và dynamic loading detection
- **CFG Metrics**: Control Flow Graph properties (cyclomatic complexity, depth, etc.)
- **Metadata**: File size, entropy, import statistics

### 2. Machine Learning Models

- **Random Forest**: Fast, interpretable, good baseline
- **XGBoost**: High performance, handles imbalance
- **Neural Network**: Deep learning approach (optional)

### 3. Evaluation & Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve
- Feature Importance Analysis

### 4. Web Dashboard

- Upload và phân tích files
- Multi-model comparison
- Consistency checks
- SHAP explainability (optional)

---

## 📁 Cấu Trúc Dự Án

```
demo2/
├── src/
│   ├── features/              # Feature extraction
│   │   ├── feature_combiner.py      # Fixed schema system
│   │   ├── feature_pipeline.py      # Deterministic extraction
│   │   └── static/                  # Static analysis extractors
│   ├── dataset/               # Dataset generation
│   │   └── generate_dataset.py      # Strict family-based split
│   ├── models/                # ML models
│   │   ├── train.py                 # Training với shared preprocessing
│   │   ├── random_forest_model.py
│   │   └── xgboost_model.py
│   ├── pipeline/              # Inference
│   │   └── inference_pipeline.py    # Exact schema loading
│   ├── evaluation/            # Evaluation metrics
│   └── dashboard/             # Web dashboard
│       ├── app.py                   # Basic dashboard
│       └── app_with_dynamic_models.py  # Enhanced dashboard
├── data/
│   ├── benign/                # Benign samples
│   ├── obfuscated/            # Obfuscated samples
│   └── processed/             # Processed features & metadata
├── models/                    # Trained models
├── config/                    # Configuration files
│   ├── dataset_config.yaml
│   ├── train_config.yaml
│   └── inference_config.yaml
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
└── tests/                     # Test files
```

---

## 💻 Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Windows/Linux/macOS
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- Disk: ~2GB cho dependencies và models

### Cài Đặt Dependencies

```bash
# Clone repository (nếu có)
# cd demo2

# Cài đặt core dependencies
pip install -r requirements.txt

# Tùy chọn: Neural Network support (PyTorch)
pip install -r requirements-dl.txt
```

### Kiểm Tra Cài Đặt

```bash
python scripts/check_environment.py
```

---

## 📖 Hướng Dẫn Sử Dụng

### Workflow Hoàn Chỉnh

#### Bước 1: Chuẩn Bị Dữ Liệu

```bash
# Tạo cấu trúc thư mục
python scripts/create_sample_structure.py

# Thêm binary samples vào:
# - data/benign/        (file hợp pháp)
# - data/obfuscated/    (file obfuscated/malware)
```

**Lưu ý**: Cần ít nhất 10-20 samples mỗi loại để train model hiệu quả.

#### Bước 2: Generate Dataset

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

**Kết quả**:
- `data/processed/train_features.pkl`
- `data/processed/val_features.pkl`
- `data/processed/test_features.pkl`
- `data/processed/feature_metadata.json` (chứa schema)
- `data/processed/sample_metadata.csv`

**Thời gian**: Xem [Thời Gian Xử Lý](#-thời-gian-xử-lý)

#### Bước 3: Train Models

```bash
python src/models/train.py
```

**Kết quả**:
- `models/random_forest_model.pkl`
- `models/xgboost_model.json`
- `models/scaler.pkl` (nếu dùng preprocessing)

**Thời gian**: Xem [Thời Gian Xử Lý](#-thời-gian-xử-lý)

#### Bước 4: Evaluate Models

```bash
python src/evaluation/evaluate.py \
    --model models/random_forest_model.pkl \
    --model_type random_forest \
    --test_data data/processed/test_features.pkl
```

#### Bước 5: Chạy Dashboard

```bash
# Option 1: Basic dashboard
python src/dashboard/app.py

# Option 2: Enhanced dashboard với multi-model support
python src/dashboard/app_with_dynamic_models.py
```

Sau đó mở browser: **http://localhost:5000**

---

## 🔄 Các Cập Nhật Mới

### Version 2.0 - Major Refactoring (2024)

#### 1. Fixed Feature Schema System

**Vấn đề cũ**: Feature dimensions không đồng nhất, padding thiếu kiểm soát

**Giải pháp mới**:
- `FeatureSchema` class định nghĩa fixed schema với offsets cố định
- Thứ tự features: opcode (2,3,4-gram) → API → CFG → metadata → dynamic loader
- Schema được lưu vào metadata và load lại cho inference

**Files thay đổi**:
- `src/features/feature_combiner.py` - **VIẾT LẠI HOÀN TOÀN**
- `src/features/feature_pipeline.py` - **VIẾT LẠI HOÀN TOÀN**

#### 2. Deterministic Feature Extraction

**Vấn đề cũ**: Cùng file cho kết quả khác nhau giữa các lần chạy

**Giải pháp mới**:
- Fixed schema đảm bảo cùng file → cùng vector
- Không phụ thuộc thứ tự scan file
- Deterministic processing

#### 3. Strict Family-Based Splitting

**Vấn đề cũ**: Data leakage do cùng family xuất hiện ở train và test

**Giải pháp mới**:
- Strict group split: không có family nào xuất hiện ở cả train và test
- Verify không có overlap giữa train/val/test families
- Metadata lưu đầy đủ: file_path, label, family, split, feature_dim

**Files thay đổi**:
- `src/dataset/generate_dataset.py` - **VIẾT LẠI HOÀN TOÀN**

#### 4. Shared Preprocessing

**Vấn đề cũ**: Models học trên feature space khác nhau

**Giải pháp mới**:
- Shared preprocessing: cùng scaler cho train/val/test
- Scaler được lưu và load lại cho inference
- Fixed random states cho reproducibility

**Files thay đổi**:
- `src/models/train.py` - **VIẾT LẠI HOÀN TOÀN**
- `config/train_config.yaml` - Thêm preprocessing option

#### 5. Exact Schema Loading trong Inference

**Vấn đề cũ**: Inference dùng metadata không đồng bộ với training

**Giải pháp mới**:
- Load exact schema từ training metadata
- Deterministic predictions
- Consistency checks giữa models

**Files thay đổi**:
- `src/pipeline/inference_pipeline.py` - **VIẾT LẠI HOÀN TOÀN**
- `src/dashboard/app.py` - **CẬP NHẬT HOÀN TOÀN**
- `src/dashboard/app_with_dynamic_models.py` - **CẬP NHẬT HOÀN TOÀN**

#### 6. Ensemble với Consistency Checks

**Tính năng mới**:
- `EnsembleInferencePipeline` cho multi-model comparison
- Conflict detection: cảnh báo nếu models cho kết quả mâu thuẫn
- Consensus voting với confidence scores

**Files thay đổi**:
- `src/pipeline/inference_pipeline.py` - Thêm `EnsembleInferencePipeline`

#### 7. Dashboard Updates

**Tính năng mới**:
- Scaler/preprocessing support
- Multi-model comparison với conflict detection
- Improved error handling và logging

**Files thay đổi**:
- `src/dashboard/app.py` - **CẬP NHẬT HOÀN TOÀN**
- `src/dashboard/app_with_dynamic_models.py` - **CẬP NHẬT HOÀN TOÀN**
- `config/inference_config.yaml` - Thêm scaler_path field

### Tổng Kết Files Đã Thay Đổi

| File | Trạng Thái | Mô Tả |
|------|-----------|-------|
| `src/features/feature_combiner.py` | 🔄 Viết lại | Fixed schema system |
| `src/features/feature_pipeline.py` | 🔄 Viết lại | Deterministic extraction |
| `src/dataset/generate_dataset.py` | 🔄 Viết lại | Strict family-based split |
| `src/models/train.py` | 🔄 Viết lại | Shared preprocessing |
| `src/pipeline/inference_pipeline.py` | 🔄 Viết lại | Exact schema loading |
| `src/dashboard/app.py` | ✏️ Cập nhật | Scaler support |
| `src/dashboard/app_with_dynamic_models.py` | ✏️ Cập nhật | Multi-model comparison |
| `config/dataset_config.yaml` | ✏️ Cập nhật | Thêm random_state |
| `config/train_config.yaml` | ✏️ Cập nhật | Thêm preprocessing option |
| `config/inference_config.yaml` | ✏️ Cập nhật | Thêm scaler_path |
| `debug.py` | ✏️ Cập nhật | Scaler support |
| `src/evaluation/evaluate.py` | ✏️ Cập nhật | Preprocessing support |

---

## ⏱️ Thời Gian Xử Lý

### Dataset Generation

Thời gian generate dataset phụ thuộc vào:
- Số lượng files
- Kích thước files
- Feature extraction complexity (CFG extraction là bottleneck)

**Ước tính**:

| Số Files | Thời Gian (ước tính) | Ghi Chú |
|----------|---------------------|---------|
| 10-20 files | 2-5 phút | CFG extraction có thể chậm |
| 50-100 files | 10-30 phút | Tùy thuộc vào kích thước files |
| 200+ files | 30-60 phút | Nên disable CFG nếu không cần |

**Tối ưu hóa**:
- Disable CFG extraction nếu không cần: `enable_cfg: false` trong config
- Giảm `max_features` cho opcode n-grams
- Sử dụng SSD thay vì HDD

**Ví dụ thực tế**:
```
Processing 50 benign files: ~8 phút
Processing 50 obfuscated files: ~12 phút
Total: ~20 phút
```

### Model Training

Thời gian training phụ thuộc vào:
- Số lượng samples
- Feature dimension
- Model type
- Hardware (CPU/GPU)

**Ước tính**:

| Model | Samples | Thời Gian | Ghi Chú |
|-------|---------|-----------|---------|
| Random Forest | 100 samples | 1-3 phút | Fast, CPU-only |
| Random Forest | 1000 samples | 5-15 phút | Tùy thuộc n_estimators |
| XGBoost | 100 samples | 2-5 phút | Fast với early stopping |
| XGBoost | 1000 samples | 10-30 phút | Có thể tối ưu hơn RF |
| Neural Network | 100 samples | 10-30 phút | Cần GPU để nhanh |
| Neural Network | 1000 samples | 30-60 phút | Tùy thuộc epochs |

**Ví dụ thực tế** (100 samples, feature_dim=3500):
```
Random Forest (100 trees): ~2 phút
XGBoost (100 rounds): ~3 phút
Neural Network (50 epochs): ~15 phút
Total: ~20 phút
```

**Tối ưu hóa**:
- Giảm `n_estimators` cho tree-based models
- Giảm `epochs` cho Neural Network
- Sử dụng GPU cho Neural Network
- Disable validation nếu không cần

### Inference (Single File)

**Ước tính**: 1-5 giây per file

- Feature extraction: 0.5-3 giây (tùy file size)
- Model prediction: <0.1 giây
- Total: 1-5 giây

---

## 🌐 API & Dashboard

### Dashboard Features

- **File Upload**: Upload binary files (.exe, .dll, .bin)
- **Multi-Model Prediction**: So sánh kết quả từ nhiều models
- **Consistency Checks**: Cảnh báo nếu models mâu thuẫn
- **SHAP Explainability**: Top features ảnh hưởng (optional)
- **History**: Lưu lịch sử predictions

### API Endpoints

#### Basic Dashboard (`app.py`)

- `GET /` - Main dashboard page
- `POST /predict` - Upload file và predict với selected models
- `DELETE /api/history` - Clear history

#### Enhanced Dashboard (`app_with_dynamic_models.py`)

- `GET /` - Main dashboard page
- `POST /predict` - Upload file và predict với current model
- `GET /api/models` - List available models
- `POST /api/models/load` - Load specific model
- `POST /api/models/compare` - Compare multiple models với conflict detection
- `GET /api/history` - Get prediction history
- `GET /api/stats` - Get dashboard statistics
- `GET /api/metrics` - Get model metrics

### Cấu Hình Dashboard

Chỉnh sửa `config/inference_config.yaml`:

```yaml
inference:
  model_type: "random_forest"
  model_path: "models/random_forest_model.pkl"
  feature_metadata: "data/processed/feature_metadata.json"
  scaler_path: "models/scaler.pkl"  # Optional
  enable_explainability: true
  top_features: 5
```

---

## 📚 Tài Liệu

### Hướng Dẫn Chi Tiết

- **[HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md)** - Hướng dẫn chạy file chi tiết (tiếng Việt)
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Tổng kết refactoring
- **[DASHBOARD_UPDATE_SUMMARY.md](DASHBOARD_UPDATE_SUMMARY.md)** - Cập nhật dashboard

### Technical Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Kiến trúc hệ thống
- **[docs/FEATURES.md](docs/FEATURES.md)** - Chi tiết về features
- **[docs/USAGE.md](docs/USAGE.md)** - Hướng dẫn sử dụng nâng cao
- **[docs/MALWARE_TESTING.md](docs/MALWARE_TESTING.md)** - Hướng dẫn test malware
- **[docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md)** - Setup VM cho malware testing

### Quick References

- **[QUICK_START.md](QUICK_START.md)** - Hướng dẫn nhanh
- **[RUN_GUIDE.md](RUN_GUIDE.md)** - Hướng dẫn chạy
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Tổng quan dự án

---

## ⚠️ Lưu Ý Bảo Mật

**QUAN TRỌNG**: Dự án này làm việc với mã độc. Luôn:

- ✅ Sử dụng trong môi trường cách ly (sandbox/VM)
- ✅ Không test malware trên máy thật
- ✅ Tuân thủ các quy định pháp lý
- ✅ Backup dữ liệu quan trọng

Xem [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md) để biết cách setup VM an toàn.

---

## 🐛 Troubleshooting

### Lỗi Thường Gặp

1. **"Feature metadata not found"**
   - Giải pháp: Generate dataset trước

2. **"No valid binary files found"**
   - Giải pháp: Thêm binary files vào `data/benign/` và `data/obfuscated/`

3. **"Model expects X features, pipeline expects Y"**
   - Giải pháp: Retrain models với dataset mới

4. **"Dimension mismatch"**
   - Giải pháp: Đảm bảo scaler được tạo cùng với models

Xem [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md) để biết thêm chi tiết.

---

## 📊 Kết Quả Mong Muốn

Sau khi refactor, pipeline đạt được:

- ✅ **Deterministic**: Cùng file → cùng kết quả (100%)
- ✅ **Consistent**: Models không mâu thuẫn
- ✅ **Stable**: Kết quả lặp lại được
- ✅ **Fixed Schema**: Feature dimensions bất biến
- ✅ **Production Ready**: Đủ tiêu chuẩn nghiên cứu/triển khai

---

## 🤝 Đóng Góp

Dự án nghiên cứu về phát hiện obfuscation trong mã độc.

---

## 📄 License

MIT License

---

## 📞 Liên Hệ

Xem các file trong `docs/` để biết thêm thông tin chi tiết.

---

**Version**: 2.0 (Major Refactoring)  
**Last Updated**: 2024  
**Status**: ✅ Production Ready
