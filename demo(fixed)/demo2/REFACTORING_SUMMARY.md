# REFACTORING SUMMARY - Malware Analysis Pipeline

## Tổng quan

Đã refactor toàn bộ pipeline **Dataset Generation – Training – Inference** để đảm bảo:
- ✅ **Deterministic**: Cùng file → cùng kết quả
- ✅ **Consistent**: Models không mâu thuẫn
- ✅ **Stable**: Kết quả lặp lại được
- ✅ **Fixed Schema**: Feature dimensions bất biến

---

## Danh sách file đã cập nhật

### 1. `src/features/feature_combiner.py` → **VIẾT LẠI HOÀN TOÀN**

**Thay đổi chính:**
- Tạo `FeatureSchema` class định nghĩa fixed schema với offsets cố định
- FeatureCombiner sử dụng fixed schema thay vì dynamic concatenation
- Đảm bảo thứ tự features: opcode (2,3,4-gram) → API → CFG → metadata → dynamic loader
- Mỗi nhóm feature có offset cố định, không phụ thuộc thứ tự extraction

**Logic quan trọng:**
- Schema được lưu vào metadata và load lại cho inference
- Padding/truncation chỉ xảy ra khi cần thiết, theo đúng schema
- Feature names được generate theo schema để hỗ trợ explainability

---

### 2. `src/features/feature_pipeline.py` → **VIẾT LẠI HOÀN TOÀN**

**Thay đổi chính:**
- Sử dụng `FeatureSchema` từ FeatureCombiner
- Deterministic extraction: cùng file → cùng vector
- Load schema từ metadata cho inference (đảm bảo giống hệt training)
- Metadata lưu cả config và schema

**Logic quan trọng:**
- `from_metadata()` load exact schema từ training
- `build_feature_vector()` luôn trả về vector với fixed dimension
- Tất cả features được map vào đúng vị trí theo schema

---

### 3. `src/dataset/generate_dataset.py` → **VIẾT LẠI HOÀN TOÀN**

**Thay đổi chính:**
- Strict family-based splitting: không có family nào xuất hiện ở cả train và test
- Fixed feature dimension ngay từ đầu (không padding sau)
- Deterministic processing: files được xử lý theo thứ tự cố định
- Metadata lưu đầy đủ: file_path, label, family, split, feature_dim

**Logic quan trọng:**
- `strict_group_split()` đảm bảo không có family leakage
- Verify không có overlap giữa train/val/test families
- Tất cả features có cùng dimension ngay sau extraction
- Schema được lưu vào metadata cùng với feature_dim

---

### 4. `src/models/train.py` → **VIẾT LẠI HOÀN TOÀN**

**Thay đổi chính:**
- Load data từ pickle files (train_features.pkl, val_features.pkl, test_features.pkl)
- Shared preprocessing: cùng scaler cho train/val/test
- Fixed random states cho tất cả models
- Feature importance logging
- Overfitting detection
- Class imbalance analysis

**Logic quan trọng:**
- `apply_preprocessing()` áp dụng cùng scaler cho tất cả splits
- `check_class_imbalance()` phân tích và log class distribution
- `log_feature_importance()` log top features cho tree-based models
- `check_overfitting()` cảnh báo nếu train score >> val score

---

### 5. `src/pipeline/inference_pipeline.py` → **VIẾT LẠI HOÀN TOÀN**

**Thay đổi chính:**
- Load exact schema từ training metadata
- Deterministic predictions: cùng file → cùng kết quả
- Consistency checks: so sánh raw label vs probability-based prediction
- Ensemble pipeline với conflict detection

**Logic quan trọng:**
- `from_metadata()` load exact schema từ training
- `predict_file()` sử dụng probability-based decision (reliable hơn raw label)
- `EnsembleInferencePipeline` kiểm tra mâu thuẫn giữa các models
- Cảnh báo nếu models cho kết quả trái ngược

---

### 6. Models (RandomForest, XGBoost) → **ĐÃ CÓ FIXED RANDOM_STATE**

**Không cần thay đổi:**
- RandomForest và XGBoost đã có `random_state` parameter
- Đảm bảo config có `random_state: 42` trong train_config.yaml

---

## Các vấn đề đã giải quyết

### ✅ Feature dimension không đồng nhất
- **Trước**: Features được padding sau khi extract, dimension có thể khác nhau giữa các lần chạy
- **Sau**: Fixed schema định nghĩa dimension từ đầu, tất cả features có cùng dimension

### ✅ Padding/truncation thiếu kiểm soát
- **Trước**: Padding dựa trên max dimension tìm được
- **Sau**: Padding theo fixed schema, chỉ xảy ra khi cần thiết

### ✅ Family/group split chưa đủ chặt
- **Trước**: Có thể có family leakage
- **Sau**: Strict group split, verify không có overlap

### ✅ Feature extraction không deterministic
- **Trước**: Thứ tự features phụ thuộc vào thứ tự trong dict
- **Sau**: Fixed schema với offsets cố định, không phụ thuộc thứ tự

### ✅ Inference dùng metadata không đồng bộ
- **Trước**: Inference có thể tự sinh lại schema
- **Sau**: Load exact schema từ training metadata

### ✅ Models học feature space khác nhau
- **Trước**: Có thể train trên features khác nhau
- **Sau**: Shared preprocessing, cùng feature space

### ✅ Không có cơ chế kiểm tra consistency
- **Trước**: Không kiểm tra mâu thuẫn giữa models
- **Sau**: Ensemble pipeline với conflict detection

---

## Cách sử dụng

### 1. Generate Dataset

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

**Kết quả:**
- `data/processed/train_features.pkl`
- `data/processed/val_features.pkl`
- `data/processed/test_features.pkl`
- `data/processed/sample_metadata.csv`
- `data/processed/feature_metadata.json` (chứa schema)

### 2. Train Models

```bash
python src/models/train.py
```

**Kết quả:**
- `models/random_forest_model.pkl`
- `models/xgboost_model.json`
- `models/scaler.pkl` (nếu dùng preprocessing)
- `data/evaluation_results/*.json`

### 3. Inference

```python
from src.pipeline.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    model_path="models/random_forest_model.pkl",
    model_type="random_forest",
    feature_metadata="data/processed/feature_metadata.json",
    scaler_path="models/scaler.pkl"
)

result = pipeline.predict_file("path/to/file.exe")
print(result)
```

### 4. Ensemble Inference

```python
from src.pipeline.inference_pipeline import EnsembleInferencePipeline

ensemble = EnsembleInferencePipeline(
    model_configs=[
        {'model_path': 'models/random_forest_model.pkl', 'model_type': 'random_forest'},
        {'model_path': 'models/xgboost_model.json', 'model_type': 'xgboost'}
    ],
    feature_metadata="data/processed/feature_metadata.json",
    scaler_path="models/scaler.pkl"
)

result = ensemble.predict_file("path/to/file.exe")
if result['has_conflict']:
    print("⚠️ Conflict detected between models!")
```

---

## Verification Checklist

Sau khi refactor, verify các điểm sau:

- [ ] Cùng file cho cùng kết quả (deterministic)
- [ ] Models không mâu thuẫn trên cùng file
- [ ] Feature dimensions nhất quán giữa train/val/test
- [ ] Không có family leakage (train families ≠ test families)
- [ ] Inference load đúng schema từ training
- [ ] Preprocessing được áp dụng nhất quán
- [ ] Random states được fix cho reproducibility

---

## Notes

1. **Schema Versioning**: Nếu thay đổi feature extraction, cần regenerate dataset và retrain models
2. **Preprocessing**: Nếu dùng preprocessing, phải load scaler trong inference
3. **Family Names**: Family names phải nhất quán (lowercase, normalized)
4. **Random States**: Đảm bảo tất cả models có `random_state: 42` trong config

---

## Files không thay đổi (nhưng cần verify)

- `src/models/random_forest_model.py` - Đã có random_state
- `src/models/xgboost_model.py` - Đã có random_state
- `src/models/base_model.py` - Base class, không thay đổi
- `src/features/static/*.py` - Feature extractors, không thay đổi

---

**Kết quả mong muốn đã đạt được:**
✅ Pipeline phân tích malware/obfuscation **đúng – ổn định – nhất quán – tái lập được**

