# 🚀 BẮT ĐẦU TẠI ĐÂY

## Bước 1: Kiểm Tra Môi Trường

Chạy script kiểm tra:

```powershell
python scripts/check_environment.py
```

Nếu thiếu dependencies, cài đặt:

```powershell
pip install -r requirements.txt
```

## Bước 2: Tạo Cấu Trúc Thư Mục

```powershell
python scripts/create_sample_structure.py
```

## Bước 3: Thêm Binary Samples

**QUAN TRỌNG**: Bạn cần có binary files để train model!

### Thêm vào `data/benign/`:
- Copy các file `.exe` hợp pháp vào thư mục này
- Ví dụ: notepad.exe, calc.exe, hoặc các tool hợp pháp khác

### Thêm vào `data/obfuscated/`:
- Copy các file đã obfuscate (nếu có)
- Hoặc để trống nếu chưa có

## Bước 4: Chạy Dự Án

### 4.1. Tạo Dataset

```powershell
python main.py generate-dataset
```

### 4.2. Train Models

```powershell
python main.py train
```

### 4.3. Đánh Giá

```powershell
python main.py evaluate --model models/random_forest_model.pkl --model-type random_forest
```

### 4.4. Dashboard (Tùy chọn)

```powershell
python main.py dashboard
```

Truy cập: http://localhost:5000

## 📚 Tài Liệu Chi Tiết

- **[HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md)** - Hướng dẫn chạy chi tiết
- **[RUN_GUIDE.md](RUN_GUIDE.md)** - Hướng dẫn từng bước
- **[QUICK_START.md](QUICK_START.md)** - Hướng dẫn nhanh
- **[docs/VM_SETUP_GUIDE.md](docs/VM_SETUP_GUIDE.md)** - Setup VM để test malware

## ⚠️ Lưu Ý

1. **Cần binary samples**: Phải có ít nhất vài file `.exe` để train
2. **Test malware**: Chỉ test trong VM, không test trên máy thật
3. **Dependencies**: Một số thư viện có thể cần cài đặt thêm (xem logs)

## 🆘 Gặp Lỗi?

1. Chạy `python scripts/check_environment.py` để kiểm tra
2. Xem [docs/BUGFIXES.md](docs/BUGFIXES.md) để biết các lỗi đã sửa
3. Kiểm tra logs để biết chi tiết lỗi

