# 🔧 Troubleshooting Guide

## Lỗi Thường Gặp và Cách Fix

### 1. Dimension Mismatch Errors

#### Lỗi: `ValueError: Model and pipeline dimension mismatch: 531 vs 3522`

**Nguyên nhân**: Model được train với schema cũ, pipeline dùng schema mới

**Giải pháp**:
- ✅ Pipeline tự động xử lý (đã fix)
- ⚠️ Nên retrain models để đảm bảo accuracy

#### Lỗi: `ValueError: Feature shape mismatch, expected: 10, got 531`

**Nguyên nhân**: Vector chưa được align đúng trước khi vào model

**Giải pháp đã áp dụng**:
- ✅ Multiple alignment layers
- ✅ Verification trước khi predict
- ✅ Force alignment nếu cần

**Nếu vẫn lỗi**:
```bash
# Retrain models với dataset mới
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
python src/models/train.py
```

---

### 2. Dashboard Không Load Models

#### Lỗi: "All models failed to initialize"

**Nguyên nhân**:
- Models không tồn tại
- Feature metadata không đúng
- Scaler path không đúng

**Giải pháp**:
1. Kiểm tra models có trong `models/` không:
```bash
dir models\  # Windows
ls models/   # Linux/macOS
```

2. Kiểm tra feature metadata:
```bash
# Phải có file này
data/processed/feature_metadata.json
```

3. Kiểm tra scaler (nếu dùng preprocessing):
```bash
# Nếu dùng preprocessing, phải có file này
models/scaler.pkl
```

4. Xem logs để biết lỗi chi tiết

---

### 3. Dataset Generation Quá Chậm

**Nguyên nhân**: CFG extraction rất chậm

**Giải pháp**:
1. Disable CFG extraction:
```yaml
# config/dataset_config.yaml
features:
  cfg:
    extract_metrics: false  # Disable CFG
```

2. Giảm max_features:
```yaml
features:
  opcode_ngrams:
    max_features: 500  # Giảm từ 1000
  api_calls:
    max_features: 250  # Giảm từ 500
```

**Thời gian ước tính sau khi tối ưu**:
- 50 files: 5-10 phút (thay vì 10-30 phút)
- 100 files: 10-20 phút (thay vì 30-60 phút)

---

### 4. Training Quá Chậm

**Giải pháp**:
1. Giảm n_estimators:
```yaml
# config/train_config.yaml
random_forest:
  n_estimators: 50  # Giảm từ 100
xgboost:
  n_estimators: 50  # Giảm từ 100
```

2. Disable Neural Network (nếu không cần):
```yaml
models:
  - random_forest
  - xgboost
  # - neural_network  # Comment out
```

---

### 5. Models Cho Kết Quả Mâu Thuẫn

**Nguyên nhân**: Models được train với feature space khác nhau

**Giải pháp**:
1. Retrain tất cả models với cùng dataset:
```bash
python src/dataset/generate_dataset.py
python src/models/train.py
```

2. Đảm bảo dùng cùng preprocessing:
```yaml
# config/train_config.yaml
preprocessing: standard  # Phải giống nhau cho tất cả models
```

3. Sử dụng EnsembleInferencePipeline để detect conflicts:
```python
from src.pipeline.inference_pipeline import EnsembleInferencePipeline

ensemble = EnsembleInferencePipeline(
    model_configs=[...],
    feature_metadata="data/processed/feature_metadata.json"
)
result = ensemble.predict_file("file.exe")
if result['has_conflict']:
    print("⚠️ Models disagree!")
```

---

### 6. Feature Metadata Not Found

**Lỗi**: `FileNotFoundError: Feature metadata not found`

**Giải pháp**:
```bash
# Generate dataset trước
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

---

### 7. No Valid Binary Files Found

**Lỗi**: "No valid binary files found in data/benign/"

**Giải pháp**:
1. Kiểm tra files có trong thư mục không:
```bash
dir data\benign\  # Windows
ls data/benign/   # Linux/macOS
```

2. Đảm bảo files là binary (.exe, .dll, .bin)
3. Files phải có kích thước > 100 bytes
4. Files phải là PE hợp lệ (cho Windows)

---

### 8. Scaler Dimension Mismatch

**Lỗi**: "Scaler expects X features, but have Y"

**Nguyên nhân**: Scaler được train với dimension khác

**Giải pháp**:
1. Retrain với preprocessing:
```bash
python src/models/train.py
```

2. Hoặc không dùng preprocessing:
```yaml
# config/train_config.yaml
preprocessing: none
```

---

## 📊 Debug Commands

### Kiểm Tra Dataset

```bash
python debug.py --check-dataset
```

### Kiểm Tra Models

```bash
python debug.py --check-models
```

### Test Feature Extraction

```bash
python debug.py --test-features path/to/file.exe
```

### Test Prediction

```bash
python debug.py --test-predict path/to/file.exe --model-type random_forest
```

### Compare Multiple Files

```bash
python debug.py --compare-files file1.exe file2.exe file3.exe
```

---

## 🔍 Log Analysis

### Xem Logs Chi Tiết

```bash
# Windows
type debug_pipeline.log

# Linux/macOS
cat debug_pipeline.log
```

### Tìm Lỗi Trong Logs

```bash
# Windows PowerShell
Select-String -Path "*.log" -Pattern "ERROR|CRITICAL"

# Linux/macOS
grep -r "ERROR\|CRITICAL" *.log
```

---

## ✅ Verification Checklist

Sau khi fix lỗi, verify:

- [ ] Dashboard khởi động không lỗi
- [ ] Models được load thành công
- [ ] Có thể upload và predict files
- [ ] Kết quả hiển thị đúng
- [ ] Không có dimension mismatch warnings (hoặc chỉ có warnings, không có errors)
- [ ] Timing logs hiển thị đúng

---

## 🆘 Vẫn Không Fix Được?

1. **Xem logs chi tiết**: Check `debug_pipeline.log` và console output
2. **Retrain từ đầu**: 
   ```bash
   # Xóa models cũ
   del models\*.pkl models\*.json
   
   # Generate dataset mới
   python src/dataset/generate_dataset.py
   
   # Train models mới
   python src/models/train.py
   ```
3. **Check versions**: Đảm bảo dependencies đúng version
4. **Check configs**: Verify tất cả config files đúng format

---

**Last Updated**: 2024

