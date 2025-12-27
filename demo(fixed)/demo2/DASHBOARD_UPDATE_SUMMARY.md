# DASHBOARD UPDATE SUMMARY

## Tổng quan

Đã cập nhật toàn bộ dashboard và các file liên quan để đồng bộ với code đã refactor, đảm bảo:
- ✅ Load đúng feature metadata và schema từ training
- ✅ Hỗ trợ scaler/preprocessing
- ✅ Deterministic predictions
- ✅ Consistency checks giữa các models

---

## Files đã cập nhật

### 1. `src/dashboard/app.py` → **CẬP NHẬT HOÀN TOÀN**

**Thay đổi chính:**
- Load scaler từ `models/scaler.pkl` nếu có
- Sử dụng đúng feature metadata path
- Pass scaler_path vào InferencePipeline
- Hỗ trợ EnsembleInferencePipeline cho multi-model comparison

**Logic quan trọng:**
```python
# Load scaler if available
scaler_path = str(SCALER_PATH) if SCALER_PATH.exists() else None

# Initialize pipeline với scaler
info['pipeline'] = InferencePipeline(
    model_path=info['path'],
    model_type=info['type'],
    feature_metadata=conf_feat,
    scaler_path=conf_scaler,  # CRITICAL: Pass scaler
    enable_explainability=True,
    top_features=conf_top
)
```

---

### 2. `src/dashboard/app_with_dynamic_models.py` → **CẬP NHẬT HOÀN TOÀN**

**Thay đổi chính:**
- Tương tự app.py nhưng hỗ trợ dynamic model loading
- Consistency checks trong `/api/models/compare`
- Conflict detection giữa các models

**Logic quan trọng:**
```python
# Load model với scaler support
pipeline = InferencePipeline(
    model_path=model_info['path'],
    model_type=model_info['type'],
    feature_metadata=feature_metadata,
    scaler_path=scaler_path,  # CRITICAL
    enable_explainability=True,
    top_features=5
)
```

---

### 3. `config/inference_config.yaml` → **CẬP NHẬT**

**Thay đổi:**
- Thêm `scaler_path` field (optional)

```yaml
inference:
  model_type: "random_forest"
  model_path: "models/random_forest_model.pkl"
  feature_metadata: "data/processed/feature_metadata.json"
  scaler_path: "models/scaler.pkl"  # NEW: Optional
  model_name: "RandomForest (Obfuscation Detector)"
  enable_explainability: true
  top_features: 5
```

---

### 4. `debug.py` → **CẬP NHẬT**

**Thay đổi:**
- Thêm scaler support trong test functions
- Load scaler nếu có khi test predictions

**Logic quan trọng:**
```python
# Check for scaler
scaler_path = project_root / "models" / "scaler.pkl"
scaler_str = str(scaler_path) if scaler_path.exists() else None

pipeline = InferencePipeline(
    model_path=str(model_path),
    model_type=model_type,
    feature_metadata=str(metadata_path),
    scaler_path=scaler_str,  # NEW: Pass scaler
    enable_explainability=False
)
```

---

### 5. `src/evaluation/evaluate.py` → **CẬP NHẬT**

**Thay đổi:**
- Load và apply scaler cho test data nếu có
- Xử lý preprocessing trước khi evaluate

**Logic quan trọng:**
```python
# Check for scaler
scaler_path = Path(args.output_dir).parent / "models" / "scaler.pkl"
if scaler_path.exists():
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test = scaler.transform(X_test)  # Apply preprocessing
```

---

## Cách sử dụng Dashboard

### 1. Khởi động Dashboard

```bash
# Option 1: Basic dashboard
python src/dashboard/app.py

# Option 2: Enhanced dashboard với multi-model support
python src/dashboard/app_with_dynamic_models.py
```

### 2. Yêu cầu trước khi chạy

**Bắt buộc:**
- ✅ Models đã được train và lưu trong `models/`
- ✅ Feature metadata đã được tạo: `data/processed/feature_metadata.json`
- ✅ Dataset đã được generate với fixed schema

**Tùy chọn:**
- ⚠️ Scaler file: `models/scaler.pkl` (nếu dùng preprocessing trong training)

### 3. Kiểm tra Dashboard Status

Dashboard sẽ tự động:
1. Scan models trong `models/` directory
2. Load feature metadata từ `data/processed/feature_metadata.json`
3. Load scaler nếu có trong `models/scaler.pkl`
4. Initialize pipelines cho tất cả models

**Logs sẽ hiển thị:**
```
✅ Inference Pipeline imported successfully.
📦 Discovered: random_forest_model.pkl (random_forest)
📦 Discovered: xgboost_model.json (xgboost)
Using feature metadata: data/processed/feature_metadata.json
Using scaler: models/scaler.pkl
🔧 Initializing pipeline for random_forest_model...
✅ Pipeline ready for random_forest_model
✅ 2/2 models initialized successfully
```

---

## API Endpoints

### Basic Dashboard (`app.py`)

- `GET /` - Main dashboard page
- `POST /predict` - Upload file và predict với selected models
- `DELETE /api/history` - Clear history

### Enhanced Dashboard (`app_with_dynamic_models.py`)

- `GET /` - Main dashboard page
- `POST /predict` - Upload file và predict với current model
- `GET /api/models` - List available models
- `POST /api/models/load` - Load specific model
- `POST /api/models/compare` - Compare multiple models với conflict detection
- `GET /api/history` - Get prediction history
- `GET /api/stats` - Get dashboard statistics
- `GET /api/metrics` - Get model metrics

---

## Troubleshooting

### Lỗi: "Feature metadata not found"

**Nguyên nhân:** Chưa generate dataset hoặc metadata file không tồn tại

**Giải pháp:**
```bash
# Generate dataset trước
python src/dataset/generate_dataset.py --config config/dataset_config.yaml
```

### Lỗi: "Model expects X features, pipeline expects Y"

**Nguyên nhân:** Model được train với schema khác với metadata hiện tại

**Giải pháp:**
- Retrain models với dataset mới
- Hoặc sử dụng đúng feature_metadata từ training

### Lỗi: "Dimension mismatch" trong predictions

**Nguyên nhân:** Scaler được train với dimension khác

**Giải pháp:**
- Đảm bảo scaler được tạo cùng với models
- Hoặc retrain với preprocessing

### Models không load được

**Kiểm tra:**
1. Models có trong `models/` directory?
2. Feature metadata có tồn tại?
3. Model type có đúng? (random_forest, xgboost, neural_network)
4. File extensions đúng? (.pkl cho RF, .json cho XGBoost)

---

## Verification Checklist

Sau khi cập nhật, verify:

- [ ] Dashboard khởi động không lỗi
- [ ] Models được load thành công
- [ ] Feature metadata được load đúng
- [ ] Scaler được load (nếu có)
- [ ] Predictions hoạt động đúng
- [ ] Multi-model comparison hoạt động
- [ ] Consistency checks hoạt động
- [ ] History được lưu đúng

---

## Notes

1. **Scaler là optional**: Nếu không dùng preprocessing trong training, không cần scaler
2. **Feature metadata là bắt buộc**: Phải có để load đúng schema
3. **Model types**: Phải match với file extensions và naming convention
4. **Consistency**: Dashboard sẽ cảnh báo nếu models cho kết quả mâu thuẫn

---

**Kết quả:** Dashboard hoàn toàn đồng bộ với refactored pipeline và sẵn sàng sử dụng! ✅

