# Hướng Dẫn Chạy Dự Án - Từng Bước

## Bước 1: Cài Đặt Dependencies

### 1.1. Kiểm tra Python

```bash
python --version
# Cần Python 3.8 trở lên
```

Nếu chưa có Python, download từ: https://www.python.org/downloads/

### 1.2. Cài đặt các thư viện cần thiết

```bash
# Di chuyển vào thư mục dự án
cd d:\Code\demo2

# Cài đặt dependencies
pip install -r requirements.txt
```

**Lưu ý**: Một số thư viện có thể cần cài đặt thêm:
- **angr**: Có thể cần cài đặt phức tạp hơn, xem: https://docs.angr.io/
- **capstone**: Thường cài được qua pip
- **pefile**: Cài được qua pip

Nếu gặp lỗi, thử:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## Bước 2: Tạo Cấu Trúc Thư Mục

```bash
python scripts/create_sample_structure.py
```

Lệnh này sẽ tạo các thư mục cần thiết:
- `data/benign/` - Đặt file binary hợp pháp ở đây
- `data/obfuscated/` - Đặt file đã obfuscate ở đây
- `data/processed/` - Dữ liệu đã xử lý
- `models/` - Models đã train
- `results/` - Kết quả đánh giá

## Bước 3: Chuẩn Bị Dữ Liệu

### 3.1. Thêm Benign Samples

Đặt các file binary hợp pháp (`.exe`, `.dll`) vào thư mục `data/benign/`:

```bash
# Ví dụ: Copy file vào thư mục
copy C:\path\to\benign_file.exe data\benign\
```

**Lưu ý**: 
- Cần ít nhất vài file để train model
- File phải là binary hợp lệ (Windows PE files)
- Không đặt file `.gitkeep` (sẽ bị bỏ qua)

### 3.2. Thêm Obfuscated Samples (Tùy chọn)

Nếu có file đã obfuscate, đặt vào `data/obfuscated/`:

```bash
copy C:\path\to\obfuscated_file.exe data\obfuscated\
```

**Hoặc** sử dụng script để tạo obfuscated samples (nếu có UPX):

```bash
python scripts/obfuscate_samples.py --source data/benign/ --output data/obfuscated/ --method upx
```

**Lưu ý**: Script `obfuscate_samples.py` chỉ copy files nếu không có UPX. Để obfuscate thật, cần:
- OLLVM (Obfuscator-LLVM)
- UPX packer
- Hoặc các công cụ obfuscation khác

## Bước 4: Tạo Dataset

```bash
python main.py generate-dataset --config config/dataset_config.yaml
```

Hoặc:

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

**Kết quả mong đợi**:
```
INFO - Starting dataset generation...
INFO - Processing X valid binary files from data/benign/
INFO - Processing Y valid binary files from data/obfuscated/
INFO - Dataset generated:
INFO -   Train: Z samples
INFO -   Val: W samples
INFO -   Test: V samples
INFO -   Feature dimension: N
```

**Nếu gặp lỗi**:
- "No valid binary files found": Kiểm tra xem đã thêm binary files chưa
- Lỗi về angr: Có thể bỏ qua CFG extraction nếu không cần
- Lỗi về capstone: Đảm bảo đã cài đặt đúng

## Bước 5: Train Models

```bash
python main.py train --config config/train_config.yaml
```

Hoặc:

```bash
python src/models/train.py --config config/train_config.yaml
```

**Quá trình này sẽ**:
1. Load dataset từ `data/processed/`
2. Train 3 models: Random Forest, XGBoost, Neural Network
3. Lưu models vào `models/`
4. Đánh giá trên test set
5. Lưu kết quả vào `results/`

**Thời gian**: Tùy thuộc vào số lượng samples và cấu hình:
- Random Forest: Nhanh (~vài phút)
- XGBoost: Trung bình (~vài phút)
- Neural Network: Chậm hơn (~10-30 phút tùy epochs)

**Kết quả mong đợi**:
```
INFO - Training RandomForest...
INFO - Training accuracy: 0.XXXX
INFO - Validation accuracy: 0.XXXX
...
INFO - Best model: random_forest with F1-score: 0.XXXX
```

## Bước 6: Đánh Giá Models

### 6.1. Đánh giá một model

```bash
python main.py evaluate \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --test-data data/processed/test_features.pkl \
    --output-dir results/
```

**Kết quả sẽ được lưu vào `results/`**:
- `random_forest_metrics.csv` - Metrics
- `random_forest_classification_report.csv` - Classification report
- `random_forest_confusion_matrix.png` - Confusion matrix
- `random_forest_roc_curve.png` - ROC curve
- `random_forest_report.txt` - Báo cáo chi tiết

### 6.2. Xem kết quả

```bash
# Xem metrics
cat results/random_forest_metrics.csv

# Hoặc mở file trong Excel/Notepad
```

## Bước 7: Test với Malware (Trong VM)

⚠️ **CẢNH BÁO**: Chỉ test trong môi trường cách ly!

### 7.1. Setup VM (Xem `docs/VM_SETUP_GUIDE.md`)

### 7.2. Test một file

```bash
python scripts/test_malware.py C:\malware_test\sample.exe \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/malware_test.json
```

**Kết quả**:
```
==================================================
KẾT QUẢ PHÂN TÍCH
==================================================
File: C:\malware_test\sample.exe
Kết quả: Obfuscated
Độ tin cậy: 85.23%
Xác suất Benign: 14.77%
Xác suất Obfuscated: 85.23%
Model: random_forest
==================================================
```

### 7.3. Batch test nhiều files

```bash
python scripts/batch_test.py C:\malware_test\ \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/batch_results.json
```

## Bước 8: Chạy Dashboard

```bash
python main.py dashboard
```

Hoặc:

```bash
python src/dashboard/app.py
```

**Truy cập**: Mở browser và vào: http://localhost:5000

**Tính năng**:
- Upload file để phân tích
- Xem kết quả đánh giá models
- Visualization charts

## Workflow Hoàn Chỉnh (Tóm Tắt)

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Tạo cấu trúc
python scripts/create_sample_structure.py

# 3. Thêm binary samples vào data/benign/ và data/obfuscated/

# 4. Tạo dataset
python main.py generate-dataset

# 5. Train models
python main.py train

# 6. Đánh giá
python main.py evaluate --model models/random_forest_model.pkl --model-type random_forest

# 7. Test malware (trong VM)
python scripts/test_malware.py <path> --model models/random_forest_model.pkl --model-type random_forest

# 8. Dashboard (tùy chọn)
python main.py dashboard
```

## Troubleshooting

### Lỗi: "No module named 'xxx'"

**Giải pháp**:
```bash
pip install xxx
# Hoặc
pip install -r requirements.txt
```

### Lỗi: "No valid binary files found"

**Nguyên nhân**: Chưa thêm binary files vào thư mục

**Giải pháp**:
- Kiểm tra `data/benign/` và `data/obfuscated/` có files không
- Đảm bảo files là binary hợp lệ (.exe, .dll, .bin)
- Xem logs để biết chi tiết

### Lỗi: "Model not found"

**Nguyên nhân**: Chưa train models

**Giải pháp**:
```bash
python main.py train
```

### Lỗi với angr

**Nguyên nhân**: angr có thể khó cài đặt

**Giải pháp**:
- Có thể bỏ qua CFG extraction nếu không cần
- Hoặc cài angr theo hướng dẫn: https://docs.angr.io/

### Dataset quá nhỏ

**Nguyên nhân**: Không đủ samples để train

**Giải pháp**:
- Cần ít nhất vài chục samples mỗi loại
- Thêm nhiều binary files hơn
- Hoặc sử dụng data augmentation

## Lưu Ý

1. **Dữ liệu**: Cần có đủ binary samples để train model hiệu quả
2. **Thời gian**: Training có thể mất thời gian, đặc biệt Neural Network
3. **Memory**: Đảm bảo có đủ RAM (khuyến nghị 8GB+)
4. **Malware Testing**: Chỉ test trong VM, không test trên hệ thống production

## Tài Liệu Tham Khảo

- [README.md](README.md) - Tổng quan
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Chi tiết dự án
- [docs/USAGE.md](docs/USAGE.md) - Hướng dẫn sử dụng
- [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md) - Setup VM
- [QUICK_START.md](QUICK_START.md) - Hướng dẫn nhanh

