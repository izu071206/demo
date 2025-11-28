# Hệ Phát Hiện Rối Mã (Obfuscation Detection) trong Mã Độc bằng ML

## Mô tả dự án

Dự án này xây dựng một hệ thống phát hiện obfuscation trong mã độc sử dụng machine learning, kết hợp các kỹ thuật phân tích tĩnh và động.

## Tính năng chính

- **Feature Extraction**: 
  - Opcode n-grams (static analysis)
  - CFG (Control Flow Graph) properties
  - API calls (static/dynamic)
  
- **Machine Learning Models**:
  - Random Forest
  - XGBoost
  - Neural Network (PyTorch)

- **Evaluation**:
  - Báo cáo false positives/negatives
  - Metrics chi tiết (precision, recall, F1-score)
- **Explainability**:
  - SHAP-based feature attribution cho dashboard để giải thích vì sao mẫu bị flag

- **Dashboard**: Giao diện web gọn nhẹ để visualize kết quả

## Cấu trúc dự án

```
demo2/
├── src/
│   ├── features/          # Feature extraction modules
│   │   ├── static/        # Static analysis features
│   │   └── dynamic/       # Dynamic analysis features
│   ├── models/            # ML models
│   ├── dataset/           # Dataset generation và management
│   ├── evaluation/        # Evaluation metrics và reports
│   └── dashboard/         # Web dashboard
├── data/
│   ├── raw/               # Raw samples
│   ├── benign/            # Benign samples
│   ├── obfuscated/        # Obfuscated samples
│   └── processed/         # Processed features
├── models/                # Trained models
├── results/               # Evaluation results
├── config/                # Configuration files
└── tests/                 # Test files
```

## Cài đặt

```bash
# Core dependencies (tree models, dashboard, feature extraction)
pip install -r requirements.txt

# Tuỳ chọn: hỗ trợ Neural Network (PyTorch)
pip install -r requirements-dl.txt
```

## Sử dụng

### 1. Chuẩn bị dữ liệu

- Đọc hướng dẫn tại `data/README.md` để biết cấu trúc thư mục.
- Tải mẫu thực tế bằng script:

```bash
python scripts/download_samples.py --tag packed --limit 25
```

- Thêm các binary sạch (bao gồm cả file đã pack để chống crack) vào `data/benign/<vendor_or_app>/`.

### 2. Tạo dataset

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

Script sẽ:

- Kiểm tra chất lượng PE (header hợp lệ).
- Gắn nhãn family dựa trên thư mục (`data/obfuscated/<family>/sample.exe`).
- Chia train/val/test theo **family split** để hạn chế data leakage.
- Lưu thêm `data/processed/sample_metadata.csv` và `feature_metadata.json`.

### 3. Train models

```bash
python src/models/train.py --config config/train_config.yaml
```

### 4. Evaluate models

```bash
python src/evaluation/evaluate.py --model models/best_model.pkl --test data/test/
```

### 5. Chạy dashboard

```bash
python -m src.dashboard.app
```

Hoặc:

```bash
python main.py dashboard
```

Sau đó mở trình duyệt tại: **http://localhost:5000**

**Tính năng Dashboard:**
- Upload và phân tích file binary (.exe, .dll, .bin)
- Hiển thị kết quả prediction (Obfuscated/Benign) với độ tin cậy
- Xem xác suất cho từng class
- Giải thích SHAP (top features ảnh hưởng) - nếu được bật
- Xem metrics và biểu đồ đánh giá models

**Cấu hình:**
Dashboard kết nối trực tiếp tới model thông qua `config/inference_config.yaml`.
Sửa file này để:
- Trỏ tới model tốt nhất (RandomForest/XGBoost/NN)
- Chỉ định file metadata (`feature_metadata.json`)
- Bật/tắt explainability (SHAP)

**Lưu ý:** Dashboard tự động align features với model's expected dimension để tránh lỗi feature mismatch.

**API Endpoints:**

Dashboard cung cấp RESTful API đầy đủ:
- `/api/predict` - Prediction cho single file
- `/api/predict/batch` - Batch prediction cho nhiều files
- `/api/status` - Kiểm tra trạng thái pipeline
- `/api/health` - Health check endpoint
- `/api/stats` - Thống kê dashboard
- `/api/history` - Lịch sử predictions
- `/api/models` - Danh sách models có sẵn
- `/api/config` - Cấu hình inference
- `/api/metrics` - Metrics chi tiết của models
- `/api/features/info` - Thông tin về feature extraction

Xem chi tiết: [src/dashboard/api_docs.md](src/dashboard/api_docs.md)

## Tài Liệu

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Tổng quan dự án chi tiết
- [docs/USAGE.md](docs/USAGE.md) - Hướng dẫn sử dụng
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Kiến trúc hệ thống
- [docs/FEATURES.md](docs/FEATURES.md) - Chi tiết về features
- [docs/MALWARE_TESTING.md](docs/MALWARE_TESTING.md) - Hướng dẫn test malware
- [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md) - Hướng dẫn setup VM và test
- [docs/BUGFIXES.md](docs/BUGFIXES.md) - Các lỗi đã sửa

## Test Malware trên VM

⚠️ **QUAN TRỌNG**: Chỉ test malware trong môi trường cách ly!

Xem hướng dẫn chi tiết: [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md)

### Quick Start

```bash
# 1. Setup VM (xem VM_SETUP_GUIDE.md)

# 2. Test một file
python scripts/test_malware.py <path_to_malware> \
    --model models/random_forest_model.pkl \
    --model-type random_forest

# 3. Batch test nhiều files
python scripts/batch_test.py <malware_directory> \
    --model models/random_forest_model.pkl \
    --model-type random_forest
```

## Lưu ý bảo mật

⚠️ **CẢNH BÁO**: Dự án này làm việc với mã độc. Luôn sử dụng trong môi trường cách ly (sandbox/VM) và tuân thủ các quy định pháp lý.

## Tác giả

Dự án nghiên cứu về phát hiện obfuscation trong mã độc.

