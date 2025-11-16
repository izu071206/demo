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
pip install -r requirements.txt
```

## Sử dụng

### 1. Tạo dataset

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

### 2. Train models

```bash
python src/models/train.py --config config/train_config.yaml
```

### 3. Evaluate models

```bash
python src/evaluation/evaluate.py --model models/best_model.pkl --test data/test/
```

### 4. Chạy dashboard

```bash
python src/dashboard/app.py
```

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

