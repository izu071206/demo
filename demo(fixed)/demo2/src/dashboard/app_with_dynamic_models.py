"""
Enhanced Flask Dashboard for Obfuscation Detection System
WITH ADVANCED MODEL SELECTION AND COMPARISON
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import inference pipeline
try:
    from src.pipeline.inference_pipeline import InferencePipeline
    INFERENCE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Inference pipeline not available: {e}")
    INFERENCE_AVAILABLE = False

# Initialize Flask App
app = Flask(__name__, template_folder='templates')
CORS(app)

# Configure directories
UPLOAD_FOLDER = Path('data/upload')
EVAL_RESULTS_DIR = Path('data/evaluation_results')
HISTORY_FILE = Path('data/dashboard_history.json')
CONFIG_FILE = Path('config/inference_config.yaml')
MODELS_DIR = Path('models')

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for multiple models
loaded_models = {}  # Dict: model_id -> InferencePipeline instance
current_model_id = None
pipeline_ready = False
start_time = datetime.now()
total_predictions = 0
error_count = 0


def scan_available_models():
    """
    Scan models directory and return list of available models
    Returns: List of dict with model info
    """
    available_models = []
    
    if not MODELS_DIR.exists():
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        return available_models
    
    # Define model type mapping based on file extension and name
    model_type_map = {
        'random_forest': {'extensions': ['.pkl'], 'keywords': ['random_forest', 'rf']},
        'xgboost': {'extensions': ['.json', '.pkl'], 'keywords': ['xgboost', 'xgb']},
        'neural_network': {'extensions': ['.pt', '.pth'], 'keywords': ['neural_network', 'nn', 'neuralnet']},
    }
    
    for model_file in MODELS_DIR.iterdir():
        if not model_file.is_file():
            continue
        
        file_ext = model_file.suffix.lower()
        file_stem = model_file.stem.lower()
        
        # Determine model type
        detected_type = None
        for model_type, type_info in model_type_map.items():
            if file_ext in type_info['extensions']:
                for keyword in type_info['keywords']:
                    if keyword in file_stem:
                        detected_type = model_type
                        break
            if detected_type:
                break
        
        if detected_type:
            model_id = model_file.stem
            
            available_models.append({
                'id': model_id,
                'name': model_file.stem.replace('_', ' ').title(),
                'type': detected_type,
                'path': str(model_file),
                'size': model_file.stat().st_size,
                'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2)
            })
    
    logger.info(f"Found {len(available_models)} available models")
    return available_models


def load_model_by_id(model_id: str):
    """
    Load a specific model by its ID
    Args:
        model_id: Model identifier (filename without extension)
    Returns:
        Success boolean
    """
    global loaded_models, current_model_id, pipeline_ready
    
    if not INFERENCE_AVAILABLE:
        logger.warning("Inference pipeline not available")
        return False
    
    # Check if already loaded
    if model_id in loaded_models:
        current_model_id = model_id
        pipeline_ready = True
        logger.info(f"Switched to already loaded model: {model_id}")
        return True
    
    # Find model info
    available = scan_available_models()
    model_info = next((m for m in available if m['id'] == model_id), None)
    
    if not model_info:
        logger.error(f"Model not found: {model_id}")
        return False
    
    try:
        logger.info(f"Loading model: {model_info['name']} ({model_info['type']})")
        
        # Get feature metadata path
        feature_metadata = "data/processed/feature_metadata.json"
        if not Path(feature_metadata).exists():
            logger.error(f"Feature metadata not found: {feature_metadata}")
            return False
        
        # Load model
        pipeline = InferencePipeline(
            model_path=model_info['path'],
            model_type=model_info['type'],
            feature_metadata=feature_metadata,
            enable_explainability=True,
            top_features=5
        )
        
        # Store loaded model
        loaded_models[model_id] = {
            'pipeline': pipeline,
            'info': model_info,
            'loaded_at': datetime.now().isoformat()
        }
        
        current_model_id = model_id
        pipeline_ready = True
        
        logger.info(f"Successfully loaded model: {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_current_pipeline():
    """Get currently active inference pipeline"""
    if not current_model_id or current_model_id not in loaded_models:
        return None
    return loaded_models[current_model_id]['pipeline']


def get_current_model_info():
    """Get info about currently active model"""
    if not current_model_id or current_model_id not in loaded_models:
        return None
    return loaded_models[current_model_id]['info']


def init_default_model():
    """Initialize default model from config or first available"""
    global current_model_id
    
    # Try to load from config
    try:
        import yaml
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f)
            
            inference_config = config.get('inference', {})
            model_path = inference_config.get('model_path')
            
            if model_path:
                model_id = Path(model_path).stem
                if load_model_by_id(model_id):
                    logger.info(f"Loaded default model from config: {model_id}")
                    return True
    except Exception as e:
        logger.warning(f"Could not load default from config: {e}")
    
    # Fallback: load first available model
    available = scan_available_models()
    if available:
        first_model = available[0]
        if load_model_by_id(first_model['id']):
            logger.info(f"Loaded first available model: {first_model['id']}")
            return True
    
    logger.warning("No models could be loaded")
    return False


def load_latest_metrics():
    """Load latest evaluation metrics"""
    if not EVAL_RESULTS_DIR.exists():
        return None
    
    files = [f for f in EVAL_RESULTS_DIR.iterdir() if f.suffix == '.json']
    if not files:
        return None
    
    target_file = next((f for f in files if 'random_forest' in f.name), files[0])
    
    try:
        with open(target_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return None


def load_all_metrics():
    """Load metrics for all available models"""
    all_metrics = {}
    
    if not EVAL_RESULTS_DIR.exists():
        return all_metrics
    
    for metrics_file in EVAL_RESULTS_DIR.iterdir():
        if metrics_file.suffix == '.json' and 'metrics' in metrics_file.name:
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                model_name = metrics_file.stem.replace('_metrics', '')
                all_metrics[model_name] = data
            except Exception as e:
                logger.error(f"Error loading {metrics_file}: {e}")
    
    return all_metrics


def load_history():
    """Load prediction history"""
    if not HISTORY_FILE.exists():
        return []
    
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []


def save_history(history):
    """Save prediction history"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving history: {e}")


def add_to_history(prediction_result):
    """Add prediction to history"""
    history = load_history()
    
    entry = {
        'id': int(time.time() * 1000),
        'filename': prediction_result.get('filename'),
        'is_obfuscated': prediction_result.get('label') == 1,
        'prediction': prediction_result.get('prediction'),
        'label': prediction_result.get('label'),
        'confidence': prediction_result.get('confidence'),
        'probabilities': prediction_result.get('probabilities'),
        'model': prediction_result.get('model'),
        'model_id': prediction_result.get('model_id'),
        'feature_count': prediction_result.get('feature_count'),
        'processing_time': prediction_result.get('processing_time'),
        'timestamp': datetime.now().isoformat()
    }
    
    history.insert(0, entry)
    history = history[:100]  # Keep only last 100
    
    save_history(history)
    return entry


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'.exe', '.dll', '.bin', '.so', '.elf', ''}
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ==================== WEB ROUTES ====================

@app.route('/')
def index():
    """Main dashboard page"""
    metrics_data = load_latest_metrics()
    all_metrics = load_all_metrics()
    history = load_history()[:10]
    available_models_list = scan_available_models()
    current_model = get_current_model_info()
    
    return render_template(
        'enhanced_index.html',
        data=metrics_data,
        all_metrics=all_metrics,
        history=history,
        pipeline_ready=pipeline_ready,
        available_models=available_models_list,
        current_model=current_model,
        current_model_id=current_model_id
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    global total_predictions, error_count
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    if not pipeline_ready:
        return jsonify({'error': 'Inference pipeline not ready'}), 503
    
    # Get model to use (from form or use current)
    model_id = request.form.get('model_id', current_model_id)
    
    # Load model if different from current
    if model_id != current_model_id:
        if not load_model_by_id(model_id):
            return jsonify({'error': f'Failed to load model: {model_id}'}), 500
    
    pipeline = get_current_pipeline()
    if not pipeline:
        return jsonify({'error': 'No pipeline available'}), 503
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Predict
        start_time_pred = time.time()
        result = pipeline.predict_file(str(filepath))
        processing_time = time.time() - start_time_pred
        
        # Add metadata
        model_info = get_current_model_info()
        result['filename'] = filename
        result['processing_time'] = round(processing_time, 3)
        result['model'] = model_info['name'] if model_info else 'Unknown'
        result['model_id'] = current_model_id
        result['model_type'] = model_info['type'] if model_info else 'unknown'
        
        # Add to history
        entry = add_to_history(result)
        result['id'] = entry['id']
        
        # Update stats
        total_predictions += 1
        
        # Clean up uploaded file
        try:
            filepath.unlink()
        except:
            pass
        
        result['is_obfuscated'] = result.get('label') == 1
        
        return jsonify(result)
        
    except Exception as e:
        error_count += 1
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


# ==================== API ROUTES ====================

@app.route('/api/models')
def api_models():
    """Get list of available models"""
    available_models_list = scan_available_models()
    
    for model in available_models_list:
        model['loaded'] = model['id'] in loaded_models
        model['active'] = model['id'] == current_model_id
    
    return jsonify({
        'models': available_models_list,
        'current': current_model_id,
        'loaded_count': len(loaded_models)
    })


@app.route('/api/models/load', methods=['POST'])
def api_load_model():
    """Load a specific model"""
    data = request.get_json()
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({'error': 'model_id required'}), 400
    
    if load_model_by_id(model_id):
        model_info = get_current_model_info()
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_id}',
            'model': model_info
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Failed to load model: {model_id}'
        }), 500


@app.route('/api/models/compare', methods=['POST'])
def api_compare_models():
    """Compare predictions from multiple models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get models to compare
    model_ids = request.form.get('model_ids', '').split(',')
    model_ids = [m.strip() for m in model_ids if m.strip()]
    
    if not model_ids:
        # Use all available models
        available = scan_available_models()
        model_ids = [m['id'] for m in available]
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        results = {}
        
        # Test with each model
        for model_id in model_ids:
            try:
                # Load model
                if not load_model_by_id(model_id):
                    results[model_id] = {'error': 'Failed to load model'}
                    continue
                
                pipeline = get_current_pipeline()
                model_info = get_current_model_info()
                
                # Predict
                start_time_pred = time.time()
                result = pipeline.predict_file(str(filepath))
                processing_time = time.time() - start_time_pred
                
                result['processing_time'] = round(processing_time, 3)
                result['model_name'] = model_info['name'] if model_info else 'Unknown'
                result['model_type'] = model_info['type'] if model_info else 'unknown'
                
                results[model_id] = result
                
            except Exception as e:
                logger.error(f"Error with model {model_id}: {e}")
                results[model_id] = {'error': str(e)}
        
        # Clean up
        try:
            filepath.unlink()
        except:
            pass
        
        # Calculate consensus
        predictions = [r.get('prediction') for r in results.values() if 'error' not in r]
        obf_count = sum(1 for p in predictions if p == 'Obfuscated')
        ben_count = len(predictions) - obf_count
        
        consensus = {
            'obfuscated_votes': obf_count,
            'benign_votes': ben_count,
            'total_models': len(predictions),
            'consensus': 'Obfuscated' if obf_count > ben_count else 'Benign' if ben_count > obf_count else 'Split'
        }
        
        return jsonify({
            'filename': filename,
            'results': results,
            'consensus': consensus
        })
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({
            'error': 'Comparison failed',
            'message': str(e)
        }), 500


@app.route('/api/history')
def api_history():
    """Get prediction history"""
    limit = request.args.get('limit', 50, type=int)
    model_filter = request.args.get('model_id', None)
    
    history = load_history()
    
    if model_filter:
        history = [h for h in history if h.get('model_id') == model_filter]
    
    history = history[:limit]
    
    return jsonify({
        'history': history,
        'count': len(history),
        'filtered_by_model': model_filter
    })


@app.route('/api/stats')
def api_stats():
    """Get dashboard statistics"""
    history = load_history()
    
    obfuscated_count = sum(1 for h in history if h.get('is_obfuscated'))
    benign_count = len(history) - obfuscated_count
    
    # Stats per model
    model_stats = {}
    for entry in history:
        model_id = entry.get('model_id', 'unknown')
        if model_id not in model_stats:
            model_stats[model_id] = {'total': 0, 'obfuscated': 0, 'benign': 0}
        
        model_stats[model_id]['total'] += 1
        if entry.get('is_obfuscated'):
            model_stats[model_id]['obfuscated'] += 1
        else:
            model_stats[model_id]['benign'] += 1
    
    model_info = get_current_model_info()
    
    return jsonify({
        'total_predictions': total_predictions,
        'obfuscated_count': obfuscated_count,
        'benign_count': benign_count,
        'errors': error_count,
        'start_time': start_time.isoformat(),
        'model_info': {
            'id': current_model_id,
            'name': model_info['name'] if model_info else None,
            'type': model_info['type'] if model_info else None,
            'loaded': pipeline_ready
        },
        'models_loaded': len(loaded_models),
        'model_stats': model_stats,
        'recent_predictions': len(history)
    })


@app.route('/api/metrics')
def api_metrics():
    """Get detailed metrics for all models"""
    all_metrics = load_all_metrics()
    
    for model_name in all_metrics:
        cm_file = EVAL_RESULTS_DIR / f"{model_name}_confusion_matrix.png"
        roc_file = EVAL_RESULTS_DIR / f"{model_name}_roc_curve.png"
        
        if 'charts' not in all_metrics[model_name]:
            all_metrics[model_name]['charts'] = {}
        
        all_metrics[model_name]['charts']['confusion_matrix'] = cm_file.exists()
        all_metrics[model_name]['charts']['roc_curve'] = roc_file.exists()
    
    return jsonify(all_metrics)


# Initialize default model on startup
with app.app_context():
    logger.info("="*60)
    logger.info("Initializing Enhanced Obfuscation Detection Dashboard")
    logger.info("="*60)
    
    available = scan_available_models()
    logger.info(f"Found {len(available)} available models:")
    for model in available:
        logger.info(f"  - {model['name']} ({model['type']}) - {model['size_mb']} MB")
    
    if available:
        init_default_model()
        if pipeline_ready:
            logger.info(f"✓ Dashboard ready with model: {current_model_id}")
        else:
            logger.warning("✗ Failed to initialize default model")
    else:
        logger.warning("✗ No models found. Please train models first.")
    
    logger.info("="*60)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)