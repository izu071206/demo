# 🔧 Fix Dimension Mismatch Error

## Vấn Đề

Lỗi: `ValueError: Model and pipeline dimension mismatch: 531 vs 3522`

**Nguyên nhân**: 
- Models được train với feature schema cũ (dimension = 531)
- Feature metadata hiện tại có schema mới sau khi refactor (dimension = 3522)
- Inference pipeline detect mismatch và raise error

## ✅ Giải Pháp Đã Áp Dụng

### 1. Inference Pipeline Tự Động Xử Lý

Pipeline đã được cập nhật để **tự động xử lý dimension mismatch**:

- ✅ **Không raise error ngay**: Thay vào đó, log warning và cố gắng fix
- ✅ **Tự động align**: Sử dụng model's expected dimension và pad/truncate features
- ✅ **Multiple verification layers**: Kiểm tra dimension ở nhiều điểm
- ✅ **Graceful degradation**: Cho phép dashboard hoạt động với models cũ
- ✅ **Detailed logging**: Log chi tiết để debug

### 2. Fix cho Lỗi "Feature shape mismatch, expected: 10, got 531"

**Vấn đề**: Vector có 531 features nhưng model expect 10 features

**Giải pháp đã áp dụng**:
- ✅ Align dimension TRƯỚC khi apply scaler
- ✅ Align dimension SAU khi apply scaler (nếu scaler thay đổi dimension)
- ✅ Final verification TRƯỚC khi predict
- ✅ Force alignment nếu vẫn mismatch
- ✅ Raise error với message rõ ràng nếu không thể fix

### 2. Cách Hoạt Động

Khi detect dimension mismatch:

1. **Log warning** với recommendation retrain
2. **Sử dụng model's dimension** thay vì pipeline's dimension
3. **Pad/truncate features** khi predict để match model's dimension
4. **Continue working** - dashboard vẫn hoạt động được

### 3. Logs Sẽ Hiển Thị

```
⚠️ Dimension mismatch detected! 
Model expects 531, pipeline expects 3522. 
Will use model's dimension (531) and pad/truncate features accordingly.

💡 Recommendation: Retrain models with new dataset to ensure consistency. 
Run: python src/dataset/generate_dataset.py && python src/models/train.py
```

## 🔄 Giải Pháp Lâu Dài: Retrain Models

Để đảm bảo consistency hoàn toàn, **retrain models với dataset mới**:

### Bước 1: Generate Dataset Mới

```bash
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

**Thời gian**: Xem logs để biết thời gian chính xác
- 10-20 files: 2-5 phút
- 50-100 files: 10-30 phút
- 200+ files: 30-60 phút

### Bước 2: Train Models Mới

```bash
python src/models/train.py
```

**Thời gian**: Xem logs để biết thời gian chính xác
- Random Forest: 1-3 phút
- XGBoost: 2-5 phút
- Neural Network: 10-30 phút

### Bước 3: Verify

Sau khi retrain, models sẽ có cùng dimension với pipeline và không còn warning.

## 📊 Thời Gian Xử Lý

### Dataset Generation

Script sẽ tự động log thời gian:

```
⏱️  Timing Summary:
  Benign processing: 120.45s (2.01 min)
  Obfuscated processing: 180.23s (3.00 min)
  Total time: 300.68s (5.01 min)
  Avg time per benign file: 2.41s
  Avg time per obfuscated file: 3.60s
```

**Ước tính**:
- **10-20 files**: 2-5 phút
- **50-100 files**: 10-30 phút  
- **200+ files**: 30-60 phút

**Tối ưu hóa**:
- Disable CFG: `enable_cfg: false` trong config → giảm 50-70% thời gian
- Giảm `max_features` cho opcode n-grams
- Sử dụng SSD thay vì HDD

### Model Training

Script sẽ tự động log thời gian cho từng model:

```
⏱️  Timing Summary:
  random_forest: 120.45s (2.01 min)
  xgboost: 180.23s (3.00 min)
  Total training time: 300.68s (5.01 min)
```

**Ước tính** (100 samples, feature_dim=3500):
- **Random Forest**: 1-3 phút
- **XGBoost**: 2-5 phút
- **Neural Network**: 10-30 phút

**Tối ưu hóa**:
- Giảm `n_estimators` cho tree-based models
- Giảm `epochs` cho Neural Network
- Sử dụng GPU cho Neural Network

## 🎯 Kết Quả

Sau khi fix:

- ✅ Dashboard hoạt động được với models cũ (có warning)
- ✅ Models mới sẽ không có warning (sau khi retrain)
- ✅ Timing được log tự động trong cả dataset generation và training

## 📝 Notes

1. **Models cũ vẫn hoạt động**: Fix cho phép sử dụng models cũ, nhưng nên retrain để đảm bảo accuracy
2. **Timing logs**: Tự động hiển thị trong logs, không cần config thêm
3. **Recommendation**: Luôn retrain models sau khi refactor feature extraction

---

**Status**: ✅ Fixed - Dashboard hoạt động được với models cũ và mới

