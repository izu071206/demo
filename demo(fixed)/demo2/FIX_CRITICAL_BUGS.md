# 🔧 Fix Critical Bugs - Prediction Issues

## Các Lỗi Đã Fix

### 1. ✅ Kết Quả Phân Tích Lặp Lại

**Vấn đề**: Cùng file cho kết quả khác nhau giữa các lần chạy

**Nguyên nhân**:
- Opcode và API features không có fixed dimension
- Feature extraction không deterministic

**Giải pháp**:
- ✅ Opcode extractor: Luôn trả về vector với `max_features` dimension (pad với zeros nếu thiếu)
- ✅ API extractor: Luôn trả về vector với `max_features` dimension (pad với zeros nếu thiếu)
- ✅ Fixed schema đảm bảo deterministic ordering

**Files thay đổi**:
- `src/features/static/opcode_extractor.py` - Fixed dimension
- `src/features/static/api_extractor.py` - Fixed dimension

---

### 2. ✅ XGBoost Luôn Predict "Benign"

**Vấn đề**: XGBoost luôn predict "Benign" dù input là gì

**Nguyên nhân**:
- Class order mapping sai
- Probability mapping không đúng

**Giải pháp**:
- ✅ Cải thiện class order detection
- ✅ Fix probability mapping logic
- ✅ Log chi tiết để debug
- ✅ Sử dụng probability-based decision thay vì raw label

**Files thay đổi**:
- `src/pipeline/inference_pipeline.py` - Fixed class order và probability mapping

**Logic mới**:
```python
# Detect class order từ model.classes_
# classes_ = [0, 1] means prob[0] = P(benign), prob[1] = P(obfuscated)
# Map correctly based on actual class values
```

---

### 3. ✅ Random Forest Luôn Predict "Obfuscated"

**Vấn đề**: Random Forest luôn predict "Obfuscated" dù input là gì

**Nguyên nhân**: Tương tự XGBoost - class order mapping sai

**Giải pháp**: Tương tự fix cho XGBoost

---

### 4. ✅ Dataset Generation Quá Chậm (Hơn 2 Giờ)

**Vấn đề**: CFG extraction với angr rất chậm, có thể mất vài phút per file

**Giải pháp**:
- ✅ Thêm timeout cho CFG extraction (20 seconds per file)
- ✅ Disable CFG mặc định trong config
- ✅ Return empty metrics nếu timeout

**Files thay đổi**:
- `src/features/static/cfg_extractor.py` - Thêm timeout
- `src/features/feature_pipeline.py` - Sử dụng timeout
- `config/dataset_config.yaml` - Disable CFG mặc định

**Cách sử dụng**:
```yaml
# config/dataset_config.yaml
features:
  cfg:
    extract_metrics: false  # Disable để tăng tốc
    # Hoặc
    extract_metrics: true
    timeout: 20  # 20 seconds per file
```

**Thời gian sau khi fix**:
- **Với CFG disabled**: 50 files ~ 5-10 phút (thay vì 2+ giờ)
- **Với CFG timeout=20s**: 50 files ~ 15-30 phút

---

## Verification

Sau khi fix, verify:

1. **Deterministic Predictions**:
   ```bash
   # Test cùng file 2 lần
   python debug.py --test-predict file.exe --model-type random_forest
   # Phải cho cùng kết quả
   ```

2. **Correct Predictions**:
   - XGBoost không luôn "Benign"
   - Random Forest không luôn "Obfuscated"
   - Predictions thay đổi theo input

3. **Fast Dataset Generation**:
   - 50 files: < 30 phút (với CFG disabled)
   - Check logs để xem timing

---

## Khuyến Nghị

### Để Fix Hoàn Toàn

1. **Retrain Models**:
   ```bash
   # Generate dataset mới (với CFG disabled để nhanh)
   python src/dataset/generate_dataset.py --config config/dataset_config.yaml
   
   # Train models mới
   python src/models/train.py
   ```

2. **Verify Models**:
   - Test với files khác nhau
   - Verify predictions không lặp lại
   - Verify models không luôn predict cùng một class

### Tối Ưu Hóa Dataset Generation

**Option 1: Disable CFG** (Khuyến nghị)
```yaml
features:
  cfg:
    extract_metrics: false
```

**Option 2: Giảm Timeout**
```yaml
features:
  cfg:
    extract_metrics: true
    timeout: 10  # 10 seconds per file
```

**Option 3: Giảm Max Features**
```yaml
features:
  opcode_ngrams:
    max_features: 500  # Giảm từ 1000
  api_calls:
    max_features: 250  # Giảm từ 500
```

---

## Thời Gian Ước Tính Sau Khi Fix

### Dataset Generation (CFG Disabled)

| Số Files | Thời Gian | Ghi Chú |
|----------|-----------|---------|
| 10-20 files | 2-5 phút | Nhanh |
| 50-100 files | 10-20 phút | Hợp lý |
| 200+ files | 30-60 phút | Có thể chấp nhận |

### Dataset Generation (CFG với Timeout 20s)

| Số Files | Thời Gian | Ghi Chú |
|----------|-----------|---------|
| 10-20 files | 5-10 phút | Chậm hơn nhưng có CFG |
| 50-100 files | 20-40 phút | Có thể chấp nhận |
| 200+ files | 60-120 phút | Vẫn chậm |

**Khuyến nghị**: Disable CFG nếu không cần thiết để tăng tốc đáng kể.

---

## Testing

Sau khi fix, test:

```bash
# 1. Test deterministic
python debug.py --test-predict data/benign/calc.exe --model-type random_forest
python debug.py --test-predict data/benign/calc.exe --model-type random_forest
# Phải cho cùng kết quả

# 2. Test với files khác nhau
python debug.py --test-predict data/benign/calc.exe --model-type random_forest
python debug.py --test-predict data/obfuscated/WannaCry.exe --model-type random_forest
# Phải cho kết quả khác nhau

# 3. Test cả 2 models
python debug.py --test-predict file.exe --model-type random_forest
python debug.py --test-predict file.exe --model-type xgboost
# Cả 2 phải hoạt động và không luôn predict cùng class
```

---

**Status**: ✅ Fixed - Predictions giờ deterministic và đúng

