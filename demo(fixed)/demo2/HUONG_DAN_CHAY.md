# Hướng Dẫn Chạy Dự Án - Thực Tế

## 🚀 Bắt Đầu Nhanh

### Bước 1: Kiểm Tra Môi Trường

```powershell
# Kiểm tra Python
python --version
# Cần Python 3.8 trở lên

# Kiểm tra pip
pip --version
```

### Bước 2: Cài Đặt Dependencies

```powershell
# Di chuyển vào thư mục dự án
cd d:\Code\demo2

# Cài đặt tất cả thư viện
pip install -r requirements.txt
```

**Nếu gặp lỗi**, thử:
```powershell
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Bước 3: Tạo Cấu Trúc Thư Mục

```powershell
python scripts/create_sample_structure.py
```

## 📁 Chuẩn Bị Dữ Liệu

### Bước 4: Thêm Binary Samples

**Quan trọng**: Bạn cần có ít nhất vài file binary để train model!

#### 4.1. Thêm Benign Samples (File hợp pháp)

```powershell
# Copy các file .exe hợp pháp vào thư mục
# Ví dụ:
copy C:\Windows\System32\notepad.exe data\benign\
copy C:\Windows\System32\calc.exe data\benign\
```

**Hoặc** tạo một số file test đơn giản (nếu không có sẵn):
- Download các tool hợp pháp từ internet
- Sử dụng các file từ Windows System32 (chỉ đọc, không chạy)

#### 4.2. Thêm Obfuscated Samples (Tùy chọn)

```powershell
# Nếu có file đã obfuscate, copy vào:
copy C:\path\to\obfuscated.exe data\obfuscated\
```

**Hoặc** sử dụng script (chỉ copy, không obfuscate thật):
```powershell
python scripts/obfuscate_samples.py --source data/benign/ --output data/obfuscated/ --method copy
```

## 🔧 Chạy Dự Án

### Bước 5: Tạo Dataset

```powershell
python main.py generate-dataset
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

**Nếu thấy lỗi "No valid binary files found"**:
- Kiểm tra xem đã thêm binary files vào `data/benign/` chưa
- Đảm bảo files là `.exe`, `.dll`, hoặc `.bin`
- Files phải có kích thước > 100 bytes

### Bước 6: Train Models

```powershell
python main.py train
```

**Quá trình này sẽ**:
1. Load dataset từ `data/processed/`
2. Train 3 models: Random Forest, XGBoost, Neural Network
3. Lưu vào `models/`
4. Đánh giá và lưu kết quả vào `results/`

**Thời gian**: 
- Random Forest: ~1-5 phút
- XGBoost: ~2-10 phút  
- Neural Network: ~10-30 phút (tùy epochs)

**Kết quả sẽ được lưu trong**:
- `models/random_forest_model.pkl`
- `models/xgboost_model.json`
- `models/neural_network_model.pt`

### Bước 7: Đánh Giá Models

```powershell
python main.py evaluate --model models/random_forest_model.pkl --model-type random_forest
```

**Kết quả sẽ được lưu trong `results/`**:
- Metrics CSV
- Confusion matrix (PNG)
- ROC curve (PNG)
- Báo cáo chi tiết (TXT)

### Bước 8: Chạy Dashboard (Tùy chọn)

```powershell
python main.py dashboard
```

Sau đó mở browser: **http://localhost:5000**

## 🧪 Test với Malware (Trong VM)

⚠️ **CẢNH BÁO**: Chỉ test trong VM, không test trên máy thật!

### Setup VM (Xem `docs/VM_SETUP_GUIDE.md`)

### Test một file:

```powershell
python scripts/test_malware.py C:\malware_test\sample.exe --model models/random_forest_model.pkl --model-type random_forest
```

### Batch test nhiều files:

```powershell
python scripts/batch_test.py C:\malware_test\ --model models/random_forest_model.pkl --model-type random_forest --output results/batch_results.json
```

## 📋 Workflow Hoàn Chỉnh (Copy & Paste)

```powershell
# 1. Cài đặt
cd d:\Code\demo2
pip install -r requirements.txt

# 2. Tạo cấu trúc
python scripts/create_sample_structure.py

# 3. Thêm binary samples (QUAN TRỌNG!)
# Copy các file .exe vào data/benign/ và data/obfuscated/

# 4. Tạo dataset
python main.py generate-dataset

# 5. Train models
python main.py train

# 6. Đánh giá
python main.py evaluate --model models/random_forest_model.pkl --model-type random_forest

# 7. Dashboard (tùy chọn)
python main.py dashboard
```

## ❌ Xử Lý Lỗi Thường Gặp

### Lỗi 1: "No module named 'xxx'"

```powershell
pip install xxx
# Hoặc
pip install -r requirements.txt
```

### Lỗi 2: "No valid binary files found"

**Nguyên nhân**: Chưa thêm binary files

**Giải pháp**:
```powershell
# Kiểm tra thư mục
dir data\benign\
dir data\obfuscated\

# Nếu trống, thêm files:
# Copy các file .exe vào các thư mục trên
```

### Lỗi 3: "Model not found"

**Nguyên nhân**: Chưa train models

**Giải pháp**:
```powershell
python main.py train
```

### Lỗi 4: Dataset quá nhỏ

**Nguyên nhân**: Không đủ samples

**Giải pháp**:
- Cần ít nhất 10-20 samples mỗi loại
- Thêm nhiều binary files hơn
- Hoặc giảm tỷ lệ train/val/test trong config

### Lỗi 5: Lỗi với angr

**Giải pháp**: 
- Có thể bỏ qua CFG extraction nếu không cần
- Hoặc cài angr theo: https://docs.angr.io/

## ✅ Checklist

Trước khi chạy:
- [ ] Python 3.8+ đã cài
- [ ] Dependencies đã cài (`pip install -r requirements.txt`)
- [ ] Cấu trúc thư mục đã tạo
- [ ] Đã thêm binary samples vào `data/benign/` và `data/obfuscated/`

Sau khi train:
- [ ] Models đã được lưu trong `models/`
- [ ] Kết quả đánh giá trong `results/`
- [ ] Có thể test với samples mới

## 📚 Tài Liệu Tham Khảo

- [RUN_GUIDE.md](RUN_GUIDE.md) - Hướng dẫn chi tiết
- [QUICK_START.md](QUICK_START.md) - Hướng dẫn nhanh
- [docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md) - Setup VM
- [docs/BUGFIXES.md](docs/BUGFIXES.md) - Các lỗi đã sửa

## 💡 Tips

1. **Bắt đầu với ít samples**: Test với 5-10 files mỗi loại trước
2. **Kiểm tra logs**: Xem logs để biết lỗi chi tiết
3. **Backup**: Backup models và results quan trọng
4. **VM cho malware**: Luôn test malware trong VM, không test trên máy thật

## 🆘 Cần Giúp Đỡ?

1. Xem logs để biết lỗi chi tiết
2. Kiểm tra các file trong `docs/` để biết thêm
3. Đảm bảo đã làm đúng các bước trên

