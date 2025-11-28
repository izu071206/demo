# Kiến Trúc Hệ Thống

## Tổng Quan

Hệ thống phát hiện obfuscation sử dụng machine learning với các thành phần chính:

1. **Feature Extraction**: Trích xuất features từ binary files
2. **Dataset Generation**: Tạo dataset từ samples
3. **Model Training**: Train các ML models
4. **Evaluation**: Đánh giá models
5. **Dashboard**: Giao diện web để visualize

## Kiến Trúc Chi Tiết

### 1. Feature Extraction Module

#### Static Analysis Features

- **Opcode N-grams** (`src/features/static/opcode_extractor.py`)
  - Sử dụng Capstone để disassemble
  - Trích xuất n-grams (2, 3, 4-grams)
  - Tạo feature vectors dựa trên frequency

- **CFG Properties** (`src/features/static/cfg_extractor.py`)
  - Sử dụng angr để trích xuất CFG
  - Tính toán metrics: num_nodes, num_edges, cyclomatic_complexity, etc.
  - Sử dụng NetworkX để phân tích graph

- **API Calls** (`src/features/static/api_extractor.py`)
  - Sử dụng pefile để parse PE files
  - Trích xuất imports và API calls
  - Tìm API calls trong strings

#### Dynamic Analysis Features (Future)

- Module `src/features/dynamic/` được chuẩn bị cho dynamic analysis
- Có thể tích hợp với sandbox như Cuckoo, CAPE

### 2. Models Module

#### Base Model (`src/models/base_model.py`)
- Abstract base class
- Interface chung: train, predict, predict_proba, save, load

#### Implementations

- **Random Forest** (`src/models/random_forest_model.py`)
  - Sử dụng scikit-learn
  - Feature importance analysis

- **XGBoost** (`src/models/xgboost_model.py`)
  - Gradient boosting
  - Early stopping support

- **Neural Network** (`src/models/neural_network_model.py`)
  - PyTorch implementation
  - Configurable architecture
  - Dropout regularization

### 3. Dataset Module

#### Dataset Generator (`src/dataset/generate_dataset.py`)
- Process directories của samples
- Extract features từ mỗi file
- Combine và normalize features
- Split train/val/test
- Save processed data

### 4. Evaluation Module

#### Evaluator (`src/evaluation/evaluator.py`)
- Tính toán metrics: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix analysis
- False positives/negatives reporting
- Visualization: confusion matrix, ROC curve
- Generate reports

### 5. Dashboard Module

#### Web Application (`src/dashboard/app.py`)
- Flask-based web server
- RESTful API
- File upload và prediction
- Results visualization

#### Frontend (`src/dashboard/templates/index.html`)
- Modern, responsive UI
- Real-time results display
- Charts và metrics visualization

## Data Flow

```
Binary Files
    ↓
Feature Extraction (Opcode, CFG, API)
    ↓
Feature Combination & Normalization
    ↓
Dataset (Train/Val/Test)
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Results & Dashboard
```

## File Structure

```
demo2/
├── src/
│   ├── features/          # Feature extraction
│   │   ├── static/       # Static analysis
│   │   └── dynamic/      # Dynamic analysis (future)
│   ├── models/           # ML models
│   ├── dataset/          # Dataset generation
│   ├── evaluation/       # Evaluation metrics
│   └── dashboard/        # Web dashboard
├── data/                 # Data storage
├── models/               # Trained models
├── results/              # Evaluation results
├── config/               # Configuration files
├── tests/                # Unit tests
└── scripts/              # Utility scripts
```

## Dependencies

### Core ML
- scikit-learn: Random Forest
- xgboost: Gradient Boosting
- PyTorch: Neural Networks

### Feature Extraction
- capstone: Disassembly
- angr: Binary analysis, CFG
- pefile: PE file parsing
- yara-python: Pattern matching

### Visualization
- matplotlib, seaborn: Static plots
- plotly: Interactive plots

### Web
- Flask: Web framework

## Configuration

### Dataset Config (`config/dataset_config.yaml`)
- Paths cho data
- Feature extraction settings
- Dataset split ratios

### Training Config (`config/train_config.yaml`)
- Model parameters
- Training settings
- Output paths

## Extension Points

### Thêm Features Mới
1. Tạo extractor mới trong `src/features/static/` hoặc `src/features/dynamic/`
2. Implement extraction logic
3. Thêm vào `generate_dataset.py`

### Thêm Model Mới
1. Kế thừa từ `BaseModel`
2. Implement các methods: train, predict, predict_proba, save, load
3. Thêm vào `src/models/__init__.py`
4. Cập nhật `train.py`

### Thêm Dynamic Analysis
1. Implement extractors trong `src/features/dynamic/`
2. Tích hợp với sandbox
3. Update `generate_dataset.py` để include dynamic features

