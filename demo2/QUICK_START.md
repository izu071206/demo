# Quick Start Guide

## Tóm Tắt Các Lỗi Đã Sửa

### ✅ Lỗi Dataset Generation

**Vấn đề**: Script đang xử lý `.gitkeep` files như binary files

**Đã sửa**:
- Thêm function `is_valid_binary_file()` để filter files
- Bỏ qua các file không phải binary (.gitkeep, .txt, .md, etc.)
- Kiểm tra file size và binary detection
- Thêm logging chi tiết hơn

**File đã sửa**: `src/dataset/generate_dataset.py`

## Hướng Dẫn Test Malware trên VM

### Bước 1: Setup VM

Xem hướng dẫn chi tiết: [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md)

**Tóm tắt**:
1. Tạo VM (VMware/VirtualBox)
2. Cài OS (Windows 10/11 hoặc Linux)
3. Cài Python và dependencies
4. Copy project vào VM
5. **Tạo snapshot** (quan trọng!)
6. **Tắt network** trong VM

### Bước 2: Train Models (Nếu chưa có)

```bash
# 1. Thêm binary samples vào:
#    - data/benign/  (file .exe hợp pháp)
#    - data/obfuscated/  (file đã obfuscate)

# 2. Tạo dataset
python main.py generate-dataset

# 3. Train models
python main.py train
```

### Bước 3: Test Malware

#### Option 1: Test một file

```bash
python scripts/test_malware.py C:\malware_test\sample.exe \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/malware_test.json
```

#### Option 2: Batch test nhiều files

```bash
python scripts/batch_test.py C:\malware_test\ \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/batch_results.json
```

#### Option 3: Sử dụng Dashboard

```bash
# 1. Chạy dashboard
python main.py dashboard

# 2. Mở browser: http://localhost:5000

# 3. Upload file và xem kết quả
```

### Bước 4: Sau khi test

1. **Restore snapshot** về clean state
2. Hoặc xóa VM và tạo lại
3. **KHÔNG copy files từ VM ra ngoài**

## Checklist Trước Khi Test

- [ ] VM đã được setup và có snapshot
- [ ] Network đã tắt trong VM
- [ ] Models đã được train
- [ ] Malware samples đã được copy vào VM
- [ ] Đã đọc [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md)

## Troubleshooting

### Lỗi: "No valid binary files found"

**Nguyên nhân**: Không có binary files trong thư mục

**Giải pháp**:
- Kiểm tra đường dẫn trong config
- Đảm bảo có file .exe, .dll, .bin trong thư mục
- Xem logs để biết chi tiết

### Lỗi: "No features extracted"

**Nguyên nhân**: 
- File không phải binary hợp lệ
- Architecture không support
- File bị corrupt

**Giải pháp**:
- Kiểm tra file type
- Thử với file khác
- Xem logs chi tiết

### Lỗi: "Model not found"

**Nguyên nhân**: Model chưa được train

**Giải pháp**:
```bash
python main.py train
```

## Tài Liệu Tham Khảo

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Tổng quan dự án
- [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md) - Hướng dẫn VM chi tiết
- [docs/MALWARE_TESTING.md](docs/MALWARE_TESTING.md) - Hướng dẫn test malware
- [docs/BUGFIXES.md](docs/BUGFIXES.md) - Các lỗi đã sửa

## Lưu Ý Bảo Mật

⚠️ **QUAN TRỌNG**:
- Chỉ test trong VM/sandbox
- Tắt network khi test
- Tạo snapshot trước khi test
- Restore snapshot sau khi test
- Tuân thủ pháp luật về malware research

