"""
Robust Flask Dashboard - Fixed Backend/Frontend Mismatch & Path Issues
"""

import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests
import yaml
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- 1. CẤU HÌNH LOGGING & ĐƯỜNG DẪN (CRITICAL FIX) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tự động xác định root project (demo2) dựa trên vị trí file app.py
# Giả định app.py nằm ở: demo2/src/dashboard/app.py
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent.parent

# Thêm root vào sys.path để import được module 'src'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Cấu hình các thư mục dữ liệu
UPLOAD_FOLDER = BASE_DIR / 'data' / 'upload'
EVAL_RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'
HISTORY_FILE = BASE_DIR / 'data' / 'dashboard_history.json'
CONFIG_FILE = BASE_DIR / 'config' / 'inference_config.yaml'
MODELS_DIR = BASE_DIR / 'models'

# Fallback: Nếu không tìm thấy models ở root, thử tìm ở thư mục hiện tại (nếu chạy sai context)
if not MODELS_DIR.exists() and Path('models').exists():
    MODELS_DIR = Path('models').resolve()

# Tạo thư mục nếu chưa có
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# --- 2. GLOBAL STATE ---
available_models = {}
pipeline_ready = False
system_error = None

# --- 3. IMPORT PIPELINE ---
INFERENCE_AVAILABLE = False
try:
    from src.pipeline.inference_pipeline import InferencePipeline
    INFERENCE_AVAILABLE = True
    logger.info("Inference Pipeline module imported successfully.")
except ImportError as e:
    logger.error(f"Inference pipeline not available: {e}")
    system_error = f"Import Error: {str(e)}"
    INFERENCE_AVAILABLE = False

# --- 4. CÁC HÀM XỬ LÝ MODEL ---

def discover_models():
    """Tìm kiếm file model với nhiều định dạng khác nhau"""
    models = {}
    if not MODELS_DIR.exists():
        return models, f"Models directory not found at {MODELS_DIR}"
    
    # Mở rộng patterns để bắt được Neural Networks
    patterns = {
        'random_forest': ['*.pkl', '*.joblib'],
        'xgboost': ['*.json', '*.model'],
        'neural_network': ['*.pt', '*.pth', '*.h5', '*.keras']
    }
    
    found_any = False
    for m_type, exts in patterns.items():
        for ext in exts:
            for f in MODELS_DIR.glob(ext):
                if f.name.startswith('.'): continue
                found_any = True
                models[f.stem] = {
                    'path': str(f),
                    'type': m_type, 
                    'display_name': f.stem.replace('_', ' ').title(),
                    'size': f.stat().st_size
                }
                logger.info(f"Discovered: {f.name} ({m_type})")
    
    if not found_any:
        return models, "No model files found in models directory."
    
    return models, None

def init_pipelines():
    """Khởi tạo pipeline cho tất cả model tìm thấy"""
    global available_models, pipeline_ready, system_error
    
    if not INFERENCE_AVAILABLE: return

    models, err = discover_models()
    if err:
        system_error = err
        logger.error(system_error)
        return

    available_models = models
    
    # Load config nếu có
    conf_feat = None
    conf_top = 5
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                c = yaml.safe_load(f).get('inference', {})
                conf_feat = c.get('feature_metadata')
                conf_top = c.get('top_features', 5)
        except: pass

    success_count = 0
    for name, info in available_models.items():
        try:
            info['pipeline'] = InferencePipeline(
                model_path=info['path'],
                model_type=info['type'],
                feature_metadata=conf_feat,
                top_features=conf_top
            )
            success_count += 1
        except Exception as e:
            info['error'] = str(e)
            logger.error(f"Failed to init {name}: {e}")

    if success_count > 0:
        pipeline_ready = True
        system_error = None
    else:
        pipeline_ready = False
        if not system_error: system_error = "All models failed to initialize."

def normalize_result(raw_res, model_info, duration):
    """Chuẩn hóa output để Frontend hiển thị đúng"""
    res = {
        'model_name': model_info['display_name'],
        'model_type': model_info['type'],
        'processing_time': round(duration, 3),
        'prediction': 'Unknown',
        'confidence': 0.0
    }
    
    # Xử lý Dictionary
    if isinstance(raw_res, dict):
        res['prediction'] = raw_res.get('prediction', raw_res.get('label', 'Unknown'))
        # Tìm confidence score trong các key phổ biến
        for key in ['confidence', 'score', 'probability', 'prob']:
            if key in raw_res:
                try: res['confidence'] = float(raw_res[key])
                except: pass
                break
    # Xử lý String
    elif isinstance(raw_res, str):
        res['prediction'] = raw_res
        res['confidence'] = 1.0 # Mặc định tin cậy tuyệt đối nếu chỉ trả về label
        
    return res

def calculate_consensus(results):
    preds = [r['prediction'] for r in results.values() if 'error' not in r]
    if not preds:
        return {'consensus': 'Unknown', 'agreement': 0, 'total': 0, 'obf': 0, 'ben': 0}
    
    obf = preds.count('Obfuscated')
    ben = preds.count('Benign')
    total = len(preds)
    
    if obf > ben: consensus = 'Obfuscated'
    elif ben > obf: consensus = 'Benign'
    else: consensus = 'Uncertain'
    
    agreement = round((max(obf, ben) / total) * 100, 1)
    confidences = [r['confidence'] for r in results.values() if 'error' not in r]
    avg_conf = sum(confidences)/len(confidences) if confidences else 0

    return {
        'consensus': consensus,
        'agreement': agreement,
        'total': total,
        'obf': obf,
        'ben': ben,
        'avg_confidence': round(avg_conf, 4)
    }

# --- 5. HELPER FUNCTIONS ---
def download_from_github(url):
    try:
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        
        fname = url.split('/')[-1].split('?')[0]
        if 'Content-Disposition' in r.headers:
            fname = re.findall("filename=(.+)", r.headers['Content-Disposition'])[0]
            
        return fname, r.content
    except Exception as e:
        raise Exception(f"GitHub Download Error: {e}")

# --- 6. ROUTES ---

@app.route('/')
def index():
    # Load history
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f: history = json.load(f)[:10]
        except: pass
        
    return render_template('index.html', 
                           available_models=available_models, 
                           pipeline_ready=pipeline_ready,
                           system_error=system_error,
                           history=history,
                           models_dir=str(MODELS_DIR))

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline_ready:
        return jsonify({'error': f'System Not Ready. {system_error or ""}'}), 503

    # 1. Lấy models
    models_selected = request.form.getlist('models[]')
    if not models_selected: return jsonify({'error': 'No models selected'}), 400

    # 2. Lấy file
    files_to_proc = []
    
    # Case A: GitHub
    if request.form.get('github_url'):
        try:
            fname, content = download_from_github(request.form['github_url'])
            files_to_proc.append((fname, content))
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    # Case B: File Upload
    elif 'file' in request.files:
        f = request.files['file']
        if f.filename:
            files_to_proc.append((secure_filename(f.filename), f.read()))
    
    if not files_to_proc:
        return jsonify({'error': 'No file provided'}), 400

    # 3. Xử lý & Dự đoán
    final_results = []
    
    for fname, content in files_to_proc:
        # Xử lý ZIP nếu cần (giản lược: chỉ save và chạy file chính)
        fpath = UPLOAD_FOLDER / fname
        with open(fpath, 'wb') as f: f.write(content)
        
        file_res = {}
        for mid in models_selected:
            if mid not in available_models: continue
            minfo = available_models[mid]
            
            if 'error' in minfo:
                file_res[mid] = {'error': minfo['error']}
                continue
                
            try:
                start = time.time()
                raw = minfo['pipeline'].predict_file(str(fpath))
                dur = time.time() - start
                file_res[mid] = normalize_result(raw, minfo, dur)
            except Exception as e:
                file_res[mid] = {'error': str(e)}
        
        consensus = calculate_consensus(file_res)
        
        # Save History
        entry = {
            'id': int(time.time()*1000),
            'filename': fname,
            'consensus': consensus,
            'results': file_res,
            'timestamp': datetime.now().isoformat(),
            'models_used': list(file_res.keys())
        }
        
        # Update History File
        hist = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE,'r') as f: hist = json.load(f)
        hist.insert(0, entry)
        with open(HISTORY_FILE, 'w') as f: json.dump(hist[:50], f, indent=2)

        final_results.append(entry)
        
        try: fpath.unlink() 
        except: pass

    return jsonify({'success': True, 'files': final_results})

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    with open(HISTORY_FILE, 'w') as f: json.dump([], f)
    return jsonify({'success': True})

# Init app
with app.app_context():
    init_pipelines()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

