# Hướng Dẫn Sử Dụng

## 1. Cài Đặt

### Yêu cầu hệ thống
- Python 3.8+
- Java (cho Ghidra API - tùy chọn)
- 8GB+ RAM (khuyến nghị)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cài đặt Ghidra (Tùy chọn)

Ghidra API cần Java và Ghidra được cài đặt:
1. Tải Ghidra từ https://ghidra-sre.org/
2. Cài đặt Java JDK
3. Cấu hình đường dẫn trong code nếu cần

## 2. Tạo Cấu Trúc Thư Mục

```bash
python scripts/create_sample_structure.py
```

## 3. Chuẩn Bị Dataset

### 3.1. Thu thập samples

- **Benign samples**: Đặt các file binary hợp pháp vào `data/benign/`
- **Obfuscated samples**: Đặt các file đã obfuscate vào `data/obfuscated/`

### 3.2. Tạo obfuscated samples (nếu cần)

Bạn có thể sử dụng các công cụ obfuscation như:
- OLLVM (Obfuscator-LLVM)
- UPX (packer)
- Các công cụ obfuscation khác

## 4. Tạo Dataset

```bash
python main.py generate-dataset --config config/dataset_config.yaml
```

Hoặc:

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

Quá trình này sẽ:
- Trích xuất features từ tất cả samples
- Chia dataset thành train/val/test
- Lưu vào `data/processed/`

## 5. Train Models

```bash
python main.py train --config config/train_config.yaml
```

Hoặc:

```bash
python src/models/train.py --config config/train_config.yaml
```

Models sẽ được lưu vào `models/` và kết quả vào `results/`

## 6. Đánh Giá Models

```bash
python main.py evaluate --model models/random_forest_model.pkl --model-type random_forest
```

## 7. Test với Malware Samples

⚠️ **CẢNH BÁO**: Chỉ chạy trong môi trường cách ly!

```bash
python scripts/test_malware.py <path_to_sample> --model models/best_model.pkl --model-type random_forest
```

## 8. Chạy Dashboard

```bash
python main.py dashboard
```

Hoặc:

```bash
python src/dashboard/app.py
```

Truy cập: http://localhost:5000

## 9. Workflow Hoàn Chỉnh

```bash
# 1. Tạo cấu trúc
python scripts/create_sample_structure.py

# 2. Đặt samples vào data/benign/ và data/obfuscated/

# 3. Tạo dataset
python main.py generate-dataset

# 4. Train models
python main.py train

# 5. Đánh giá
python main.py evaluate --model models/random_forest_model.pkl --model-type random_forest

# 6. Chạy dashboard
python main.py dashboard
```

## 10. Tùy Chỉnh

### Thay đổi features

Chỉnh sửa `config/dataset_config.yaml`:
- Thay đổi n-gram sizes
- Thay đổi số lượng features
- Bật/tắt các loại features

### Thay đổi model parameters

Chỉnh sửa `config/train_config.yaml`:
- Thay đổi hyperparameters
- Thêm/bớt models để train

## 11. Troubleshooting

### Lỗi khi extract features
- Kiểm tra file có phải là binary hợp lệ không
- Kiểm tra quyền truy cập file
- Xem logs để biết chi tiết lỗi

### Lỗi khi train
- Kiểm tra dataset đã được tạo chưa
- Kiểm tra số lượng samples (cần đủ để train)
- Kiểm tra memory (có thể cần giảm batch size)

### Lỗi với angr/Ghidra
- Đảm bảo đã cài đặt đúng
- Kiểm tra đường dẫn
- Có thể bỏ qua CFG extraction nếu không cần

