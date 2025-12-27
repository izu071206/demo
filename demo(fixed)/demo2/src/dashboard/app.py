"""
Robust Flask Dashboard - REFACTORED VERSION
✅ Đồng bộ với inference pipeline mới
✅ Hỗ trợ scaler/preprocessing
✅ Load đúng feature metadata và schema
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

# --- 1. LOGGING & PATH SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Directories
UPLOAD_FOLDER = BASE_DIR / 'data' / 'upload'
EVAL_RESULTS_DIR = BASE_DIR / 'data' / 'evaluation_results'
HISTORY_FILE = BASE_DIR / 'data' / 'dashboard_history.json'
CONFIG_FILE = BASE_DIR / 'config' / 'inference_config.yaml'
MODELS_DIR = BASE_DIR / 'models'
FEATURE_METADATA = BASE_DIR / 'data' / 'processed' / 'feature_metadata.json'
SCALER_PATH = BASE_DIR / 'models' / 'scaler.pkl'

if not MODELS_DIR.exists() and Path('models').exists():
    MODELS_DIR = Path('models').resolve()

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# --- 2. GLOBAL STATE ---
available_models = {}
pipeline_ready = False
system_error = None

# --- 3. IMPORT PIPELINE ---
INFERENCE_AVAILABLE = False
try:
    from src.pipeline.inference_pipeline import InferencePipeline, EnsembleInferencePipeline
    INFERENCE_AVAILABLE = True
    logger.info("✅ Inference Pipeline imported successfully.")
except ImportError as e:
    logger.error(f"❌ Inference pipeline not available: {e}")
    system_error = f"Import Error: {str(e)}"
    INFERENCE_AVAILABLE = False

# --- 4. MODEL DISCOVERY & INIT ---

def discover_models():
    """Tìm kiếm models với nhiều định dạng"""
    models = {}
    if not MODELS_DIR.exists():
        return models, f"Models directory not found at {MODELS_DIR}"
    
    patterns = {
        'random_forest': ['random_forest_model.pkl', 'rf_model.pkl'],
        'xgboost': ['xgboost_model.json', 'xgboost_model.pkl', 'xgb_model.json'],
        'neural_network': ['neural_network_model.pt', 'nn_model.pt', 'nn_model.pth']
    }
    
    found_any = False
    for m_type, patterns_list in patterns.items():
        for pattern in patterns_list:
            for f in MODELS_DIR.glob(pattern):
                if f.name.startswith('.'): 
                    continue
                found_any = True
                model_id = f.stem
                models[model_id] = {
                    'path': str(f),
                    'type': m_type, 
                    'display_name': model_id.replace('_', ' ').title(),
                    'size': f.stat().st_size
                }
                logger.info(f"📦 Discovered: {f.name} ({m_type})")
    
    if not found_any:
        return models, "No model files found in models directory."
    
    return models, None

def init_pipelines():
    """Khởi tạo pipelines cho tất cả models với scaler support"""
    global available_models, pipeline_ready, system_error
    
    if not INFERENCE_AVAILABLE: 
        return

    models, err = discover_models()
    if err:
        system_error = err
        logger.error(f"❌ {system_error}")
        return

    available_models = models
    
    # Load config
    conf_feat = None
    conf_scaler = None
    conf_top = 5
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                c = yaml.safe_load(f).get('inference', {})
                conf_feat = c.get('feature_metadata')
                conf_scaler = c.get('scaler_path')
                conf_top = c.get('top_features', 5)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")

    # Use default paths if not in config
    if not conf_feat:
        conf_feat = str(FEATURE_METADATA) if FEATURE_METADATA.exists() else None
    
    if not conf_scaler:
        conf_scaler = str(SCALER_PATH) if SCALER_PATH.exists() else None
    
    if not conf_feat:
        system_error = f"Feature metadata not found at {FEATURE_METADATA}"
        logger.error(f"❌ {system_error}")
        return
    
    logger.info(f"Using feature metadata: {conf_feat}")
    if conf_scaler:
        logger.info(f"Using scaler: {conf_scaler}")
    else:
        logger.info("No scaler found - models may not use preprocessing")

    success_count = 0
    for name, info in available_models.items():
        try:
            logger.info(f"🔧 Initializing pipeline for {name}...")
            info['pipeline'] = InferencePipeline(
                model_path=info['path'],
                model_type=info['type'],
                feature_metadata=conf_feat,
                scaler_path=conf_scaler,  # CRITICAL: Pass scaler path
                enable_explainability=True,
                top_features=conf_top
            )
            success_count += 1
            logger.info(f"✅ Pipeline ready for {name}")
        except Exception as e:
            info['error'] = str(e)
            logger.error(f"❌ Failed to init {name}: {e}")
            import traceback
            traceback.print_exc()

    if success_count > 0:
        pipeline_ready = True
        system_error = None
        logger.info(f"✅ {success_count}/{len(available_models)} models initialized successfully")
    else:
        pipeline_ready = False
        if not system_error: 
            system_error = "All models failed to initialize."
        logger.error(f"❌ {system_error}")

def normalize_result(raw_res, model_info, duration):
    """
    CRITICAL FIX: Chuẩn hóa output từ inference pipeline
    Đảm bảo prediction và confidence đúng
    """
    # Base result structure
    res = {
        'model_name': model_info['display_name'],
        'model_type': model_info['type'],
        'processing_time': round(duration, 3),
        'prediction': 'Unknown',
        'confidence': 0.0
    }
    
    # Extract prediction from inference result
    if isinstance(raw_res, dict):
        # CRITICAL: Use the 'prediction' field from inference (already processed correctly)
        res['prediction'] = raw_res.get('prediction', 'Unknown')
        res['confidence'] = float(raw_res.get('confidence', 0.0))
        
        # Optional: Add probabilities for detailed display
        if 'probabilities' in raw_res:
            res['probabilities'] = raw_res['probabilities']
        
        # Add feature count if available
        if 'feature_count' in raw_res:
            res['feature_count'] = raw_res['feature_count']
        
        # Log for debugging
        logger.debug(
            f"Model {model_info['display_name']}: "
            f"prediction={res['prediction']}, "
            f"confidence={res['confidence']:.4f}"
        )
        
    elif isinstance(raw_res, str):
        # Fallback for string results
        res['prediction'] = raw_res
        res['confidence'] = 1.0
    
    return res

def calculate_consensus(results):
    """Tính toán consensus từ tất cả models"""
    preds = []
    confidences = []
    
    for r in results.values():
        if 'error' not in r:
            preds.append(r['prediction'])
            confidences.append(r.get('confidence', 0.0))
    
    if not preds:
        return {
            'consensus': 'Unknown', 
            'agreement': 0, 
            'total': 0, 
            'obf': 0, 
            'ben': 0,
            'avg_confidence': 0
        }
    
    obf = preds.count('Obfuscated')
    ben = preds.count('Benign')
    total = len(preds)
    
    # Determine consensus
    if obf > ben: 
        consensus = 'Obfuscated'
    elif ben > obf: 
        consensus = 'Benign'
    else: 
        consensus = 'Uncertain'
    
    agreement = round((max(obf, ben) / total) * 100, 1)
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0

    return {
        'consensus': consensus,
        'agreement': agreement,
        'total': total,
        'obf': obf,
        'ben': ben,
        'avg_confidence': avg_conf
    }

# --- 5. HELPER FUNCTIONS ---

def download_from_github(url):
    """Download file from GitHub URL"""
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
    """Main dashboard page"""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f: 
                history = json.load(f)[:10]
        except Exception as e:
            logger.error(f"Error loading history: {e}")
        
    return render_template(
        'index.html', 
        available_models=available_models, 
        pipeline_ready=pipeline_ready,
        system_error=system_error,
        history=history,
        models_dir=str(MODELS_DIR)
    )

@app.route('/predict', methods=['POST'])
def predict():
    """CRITICAL FIX: Xử lý prediction với logic đúng"""
    if not pipeline_ready:
        return jsonify({'error': f'System Not Ready. {system_error or ""}'}), 503

    # 1. Get selected models
    models_selected = request.form.getlist('models[]')
    if not models_selected: 
        return jsonify({'error': 'No models selected'}), 400

    # 2. Get file(s)
    files_to_proc = []
    
    # Case A: GitHub URL
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

    # 3. Process each file
    final_results = []
    
    for fname, content in files_to_proc:
        # Save file temporarily
        fpath = UPLOAD_FOLDER / fname
        with open(fpath, 'wb') as f: 
            f.write(content)
        
        file_res = {}
        
        # Run through each selected model
        for mid in models_selected:
            if mid not in available_models: 
                continue
            minfo = available_models[mid]
            
            if 'error' in minfo:
                file_res[mid] = {'error': minfo['error']}
                continue
            
            try:
                logger.info(f"🔍 Analyzing {fname} with {minfo['display_name']}...")
                start = time.time()
                
                # CRITICAL: Call predict_file from pipeline
                raw = minfo['pipeline'].predict_file(str(fpath))
                
                dur = time.time() - start
                
                # CRITICAL: Normalize result properly
                file_res[mid] = normalize_result(raw, minfo, dur)
                
                logger.info(
                    f"✅ {minfo['display_name']}: {file_res[mid]['prediction']} "
                    f"({file_res[mid]['confidence']*100:.1f}%)"
                )
                
            except Exception as e:
                file_res[mid] = {'error': str(e)}
                logger.error(f"❌ Error with {minfo['display_name']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate consensus
        consensus = calculate_consensus(file_res)
        
        logger.info(
            f"📊 Consensus for {fname}: {consensus['consensus']} "
            f"(agreement: {consensus['agreement']}%)"
        )
        
        # Create history entry
        entry = {
            'id': int(time.time() * 1000),
            'filename': fname,
            'consensus': consensus,
            'results': file_res,
            'timestamp': datetime.now().isoformat(),
            'models_used': list(file_res.keys())
        }
        
        # Update history file
        hist = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, 'r') as f: 
                    hist = json.load(f)
            except:
                pass
        hist.insert(0, entry)
        with open(HISTORY_FILE, 'w') as f: 
            json.dump(hist[:50], f, indent=2)

        final_results.append(entry)
        
        # Cleanup
        try: 
            fpath.unlink() 
        except: 
            pass

    return jsonify({'success': True, 'files': final_results})

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear history"""
    try:
        with open(HISTORY_FILE, 'w') as f: 
            json.dump([], f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 7. INITIALIZE APP ---
with app.app_context():
    logger.info("="*60)
    logger.info("🚀 Initializing Malware Analysis Dashboard")
    logger.info("="*60)
    init_pipelines()
    if pipeline_ready:
        logger.info("✅ Dashboard ready!")
    else:
        logger.warning("⚠️ Dashboard started but models not ready")
    logger.info("="*60)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
