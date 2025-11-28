# Tổng Quan Dự Án: Hệ Phát Hiện Rối Mã (Obfuscation Detection)

## Mục Tiêu Dự Án

Xây dựng hệ thống phát hiện obfuscation trong mã độc sử dụng Machine Learning, kết hợp:
- **Static Analysis**: Opcode n-grams, CFG properties, API calls
- **Dynamic Analysis**: (Future work)
- **ML Models**: Random Forest, XGBoost, Neural Network
- **Dashboard**: Giao diện web để visualize và test

## Cấu Trúc Dự Án

```
demo2/
├── src/                      # Source code chính
│   ├── features/            # Feature extraction
│   │   ├── static/         # Static analysis features
│   │   └── dynamic/       # Dynamic analysis (future)
│   ├── models/             # ML models
│   ├── dataset/            # Dataset generation
│   ├── evaluation/        # Evaluation metrics
│   └── dashboard/         # Web dashboard
│
├── data/                    # Data storage
│   ├── benign/            # Benign samples
│   ├── obfuscated/        # Obfuscated samples
│   ├── processed/         # Processed features
│   └── upload/            # Uploaded files for testing
│
├── models/                  # Trained models
├── results/                 # Evaluation results
├── config/                  # Configuration files
├── tests/                   # Unit tests
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## Workflow Hoàn Chỉnh

### 1. Setup Môi Trường

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Tạo cấu trúc thư mục
python scripts/create_sample_structure.py
```

### 2. Chuẩn Bị Dataset

- Đặt **benign samples** vào `data/benign/`
- Đặt **obfuscated samples** vào `data/obfuscated/`
- Hoặc sử dụng script để tạo obfuscated samples:
  ```bash
  python scripts/obfuscate_samples.py --source data/benign/ --output data/obfuscated/
  ```

### 3. Tạo Dataset

```bash
python main.py generate-dataset --config config/dataset_config.yaml
```

Quá trình này sẽ:
- Trích xuất features từ tất cả samples
- Chia dataset thành train/val/test (70/15/15)
- Lưu vào `data/processed/`

### 4. Train Models

```bash
python main.py train --config config/train_config.yaml
```

Train 3 models:
- Random Forest
- XGBoost
- Neural Network

Models được lưu vào `models/`

### 5. Đánh Giá Models

```bash
python main.py evaluate \
    --model models/random_forest_model.pkl \
    --model-type random_forest
```

Kết quả bao gồm:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- False Positives/Negatives
- Visualization charts

### 6. Test với Malware (Trong VM/Sandbox)

```bash
python scripts/test_malware.py <path_to_malware> \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/malware_test.json
```

⚠️ **CẢNH BÁO**: Chỉ chạy trong môi trường cách ly!

### 7. Dashboard

```bash
python main.py dashboard
```

Truy cập: http://localhost:5000

Features:
- Upload file để phân tích
- Xem kết quả đánh giá models
- Visualization charts

## Features Được Trích Xuất

### 1. Opcode N-grams
- 2-grams, 3-grams, 4-grams
- Frequency-based features
- Max 1000 features per n-gram type

### 2. CFG Properties
- num_nodes, num_edges
- cyclomatic_complexity
- num_loops
- max_depth, avg_path_length
- clustering_coefficient

### 3. API Calls
- Windows API imports
- API calls trong strings
- Max 500 features

## Models

### Random Forest
- n_estimators: 100
- max_depth: 20
- Feature importance analysis

### XGBoost
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- Early stopping support

### Neural Network (PyTorch)
- Architecture: [128, 64, 32]
- Dropout: 0.3
- Learning rate: 0.001
- Batch size: 32
- Early stopping

## Evaluation Metrics

- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Tỷ lệ dự đoán obfuscated đúng
- **Recall**: Tỷ lệ phát hiện obfuscated
- **F1-Score**: Harmonic mean của Precision và Recall
- **ROC-AUC**: Area under ROC curve
- **False Positives**: Benign bị nhận diện là obfuscated
- **False Negatives**: Obfuscated bị nhận diện là benign

## Configuration

### Dataset Config (`config/dataset_config.yaml`)
- Paths cho data
- Feature extraction settings
- Dataset split ratios

### Training Config (`config/train_config.yaml`)
- Model parameters
- Training settings
- Output paths

## Tài Liệu

- `README.md`: Tổng quan dự án
- `docs/USAGE.md`: Hướng dẫn sử dụng chi tiết
- `docs/ARCHITECTURE.md`: Kiến trúc hệ thống
- `docs/FEATURES.md`: Chi tiết về features
- `docs/MALWARE_TESTING.md`: Hướng dẫn test với malware

## Lưu Ý Quan Trọng

### Bảo Mật
- ⚠️ Chỉ test malware trong VM/sandbox
- ⚠️ Không chạy trên hệ thống production
- ⚠️ Tuân thủ pháp luật về malware research

### Dependencies
- Một số dependencies có thể cần cài đặt thủ công:
  - **angr**: Có thể cần setup phức tạp
  - **Ghidra API**: Cần Java và Ghidra
  - **UPX**: Để pack samples (optional)

### Performance
- CFG extraction có thể chậm với angr
- Có thể skip CFG nếu không cần
- Neural Network training cần GPU (optional)

## Extension Points

### Thêm Features Mới
1. Tạo extractor trong `src/features/static/` hoặc `src/features/dynamic/`
2. Implement extraction logic
3. Thêm vào `generate_dataset.py`

### Thêm Model Mới
1. Kế thừa từ `BaseModel`
2. Implement: train, predict, predict_proba, save, load
3. Thêm vào `src/models/__init__.py`
4. Cập nhật `train.py`

### Thêm Dynamic Analysis
1. Implement trong `src/features/dynamic/`
2. Tích hợp với sandbox (Cuckoo, CAPE)
3. Update `generate_dataset.py`

## Troubleshooting

### Lỗi khi extract features
- Kiểm tra file có phải binary hợp lệ
- Kiểm tra architecture support
- Xem logs để biết chi tiết

### Lỗi khi train
- Kiểm tra dataset đã được tạo
- Kiểm tra số lượng samples
- Kiểm tra memory (giảm batch size nếu cần)

### Lỗi với angr
- Đảm bảo đã cài đặt đúng
- Có thể skip CFG extraction nếu không cần

## Kế Hoạch Phát Triển

### Phase 1 (Current)
- ✅ Static analysis features
- ✅ Basic ML models
- ✅ Evaluation system
- ✅ Dashboard

### Phase 2 (Future)
- [ ] Dynamic analysis features
- [ ] Feature selection và dimensionality reduction
- [ ] Ensemble methods
- [ ] Real-time detection API

### Phase 3 (Advanced)
- [ ] Deep learning với graph neural networks
- [ ] Transfer learning
- [ ] Explainable AI
- [ ] Integration với security tools

## Liên Hệ & Đóng Góp

Dự án nghiên cứu về phát hiện obfuscation trong mã độc.

## License

Dự án nghiên cứu - Sử dụng cho mục đích giáo dục và nghiên cứu.

