"""
Dashboard Web Application
Giao diện gọn nhẹ để visualize kết quả với backend chuyên nghiệp
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import InferencePipeline  # noqa: E402

app = Flask(__name__)
CORS(app)  # Enable CORS for API access
app.config['UPLOAD_FOLDER'] = 'data/upload/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['PREDICTION_HISTORY'] = 'data/dashboard_history.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFERENCE_PIPELINE = None
INFERENCE_CONFIG = {}
PREDICTION_CACHE = {}
STATS = {
    'total_predictions': 0,
    'obfuscated_count': 0,
    'benign_count': 0,
    'errors': 0,
    'start_time': datetime.now().isoformat()
}


def load_inference_pipeline():
    global INFERENCE_PIPELINE, INFERENCE_CONFIG
    config_path = os.getenv('INFERENCE_CONFIG', 'config/inference_config.yaml')
    config_full = Path(config_path)
    if not config_full.exists():
        logger.warning("Inference config %s not found. Dashboard will serve mock data.", config_path)
        return
    with config_full.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f).get('inference', {})
    try:
        INFERENCE_PIPELINE = InferencePipeline(
            model_path=cfg['model_path'],
            model_type=cfg['model_type'],
            feature_metadata=cfg['feature_metadata'],
            enable_explainability=cfg.get('enable_explainability', False),
            top_features=cfg.get('top_features', 5),
        )
        INFERENCE_CONFIG = cfg
        logger.info("Inference pipeline loaded (%s).", cfg['model_type'])
    except Exception as exc:
        logger.error("Failed to initialize inference pipeline: %s", exc)
        INFERENCE_PIPELINE = None


load_inference_pipeline()


# Middleware for request logging
@app.before_request
def log_request_info():
    """Log request information"""
    if request.path.startswith('/api/'):
        logger.info(f"API Request: {request.method} {request.path}")


@app.after_request
def after_request(response):
    """Add response headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error", "message": "An unexpected error occurred"}), 500


def save_prediction_history(prediction_data: dict):
    """Save prediction to history"""
    history_path = Path(app.config['PREDICTION_HISTORY'])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    history = []
    if history_path.exists():
        try:
            with history_path.open('r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    history.insert(0, {
        **prediction_data,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 100 predictions
    history = history[:100]
    
    with history_path.open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def load_prediction_history(limit: int = 50) -> List[dict]:
    """Load prediction history"""
    history_path = Path(app.config['PREDICTION_HISTORY'])
    if not history_path.exists():
        return []
    
    try:
        with history_path.open('r', encoding='utf-8') as f:
            history = json.load(f)
        return history[:limit]
    except:
        return []


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def status():
    """Check if inference pipeline is ready"""
    if INFERENCE_PIPELINE is None:
        return jsonify({"status": "not_ready", "message": "Inference pipeline not initialized"}), 503
    return jsonify({"status": "ready", "model": INFERENCE_CONFIG.get('model_name', 'Unknown')}), 200


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get evaluation results"""
    results_dir = Path("results/")
    
    if not results_dir.exists():
        return jsonify({"error": "No results found"}), 404
    
    results = {}
    
    # Load metrics for each model
    for metrics_file in results_dir.glob("*_metrics.csv"):
        model_name = metrics_file.stem.replace("_metrics", "")
        df = pd.read_csv(metrics_file)
        results[model_name] = df.to_dict('records')[0]
    
    return jsonify(results)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict obfuscation for uploaded file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save uploaded file
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = upload_dir / file.filename
    filepath_str = str(filepath.resolve())
    file.save(filepath_str)
    
    try:
        if INFERENCE_PIPELINE is None:
            load_inference_pipeline()
        if INFERENCE_PIPELINE is None:
            return jsonify({"error": "Inference pipeline not configured."}), 503
        
        # Use absolute path for feature extraction
        start_time = time.time()
        result = INFERENCE_PIPELINE.predict_file(filepath_str)
        processing_time = time.time() - start_time
        
        # Get values from result
        label = result.get('label', 0)  # This is already fixed in pipeline
        label_raw = result.get('label_raw', label)  # Raw label from model for debugging
        prediction_str = result.get('prediction', 'Benign')
        probabilities = result.get('probabilities', {})
        
        # Use probability-based decision as primary source (most reliable)
        prob_obf = probabilities.get('obfuscated', 0.0)
        prob_ben = probabilities.get('benign', 0.0)
        is_obfuscated = prob_obf > prob_ben
        
        # Double-check with prediction string and label
        if prediction_str.lower() == 'obfuscated':
            is_obfuscated = True
        elif prediction_str.lower() == 'benign':
            is_obfuscated = False
        
        # Log for debugging
        logger.info(f"Prediction result - Label (fixed): {label}, Label (raw): {label_raw}, "
                   f"Prediction: {prediction_str}, Probs: benign={prob_ben:.4f}, obfuscated={prob_obf:.4f}, "
                   f"Final IsObfuscated: {is_obfuscated}")
        
        payload = {
            "id": int(time.time() * 1000),  # Unique ID
            "filename": file.filename,
            "is_obfuscated": is_obfuscated,
            "prediction": prediction_str,
            "label": int(label),
            "confidence": result['confidence'],
            "probabilities": probabilities,
            "model": INFERENCE_CONFIG.get('model_name', result['model_type']),
            "feature_count": result['feature_count'],
            "processing_time": round(processing_time, 3)
        }
        if 'top_contributors' in result:
            payload['top_contributors'] = result['top_contributors']
        
        # Update statistics
        STATS['total_predictions'] += 1
        if payload['is_obfuscated']:
            STATS['obfuscated_count'] += 1
        else:
            STATS['benign_count'] += 1
        
        # Save to history
        save_prediction_history(payload)
        
        return jsonify(payload)
    
    except Exception as e:
        logger.error(f"Error predicting: {e}", exc_info=True)
        STATS['errors'] += 1
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up file {filepath}: {cleanup_error}")


@app.route('/api/charts/confusion_matrix/<model_name>')
def get_confusion_matrix(model_name):
    """Get confusion matrix image"""
    img_path = Path(f"results/{model_name}_confusion_matrix.png")
    
    if img_path.exists():
        return send_file(str(img_path), mimetype='image/png')
    else:
        return jsonify({"error": "Image not found"}), 404


@app.route('/api/charts/roc_curve/<model_name>')
def get_roc_curve(model_name):
    """Get ROC curve image"""
    img_path = Path(f"results/{model_name}_roc_curve.png")
    
    if img_path.exists():
        return send_file(str(img_path), mimetype='image/png')
    else:
        return jsonify({"error": "Image not found"}), 404


# New API endpoints for enhanced functionality

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_dir = Path("models/")
    models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.pkl"):
            models.append({
                "name": model_file.stem.replace("_model", ""),
                "type": "random_forest",
                "path": str(model_file),
                "size": model_file.stat().st_size
            })
        for model_file in models_dir.glob("*.json"):
            models.append({
                "name": model_file.stem.replace("_model", ""),
                "type": "xgboost",
                "path": str(model_file),
                "size": model_file.stat().st_size
            })
        for model_file in models_dir.glob("*.pt"):
            models.append({
                "name": model_file.stem.replace("_model", ""),
                "type": "neural_network",
                "path": str(model_file),
                "size": model_file.stat().st_size
            })
    
    return jsonify({"models": models, "current": INFERENCE_CONFIG.get('model_name', 'None')})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    global STATS
    history = load_prediction_history(limit=1000)
    
    # Update stats from history
    if history:
        STATS['total_predictions'] = len(history)
        STATS['obfuscated_count'] = sum(1 for h in history if h.get('is_obfuscated', False))
        STATS['benign_count'] = STATS['total_predictions'] - STATS['obfuscated_count']
    
    return jsonify({
        **STATS,
        "model_info": {
            "type": INFERENCE_CONFIG.get('model_type', 'unknown'),
            "name": INFERENCE_CONFIG.get('model_name', 'unknown'),
            "loaded": INFERENCE_PIPELINE is not None
        },
        "recent_predictions": len(history)
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    limit = request.args.get('limit', 50, type=int)
    history = load_prediction_history(limit=limit)
    return jsonify({"history": history, "count": len(history)})


@app.route('/api/history/<prediction_id>', methods=['GET'])
def get_prediction_detail(prediction_id):
    """Get detailed information about a specific prediction"""
    history = load_prediction_history(limit=1000)
    for pred in history:
        if str(pred.get('id')) == str(prediction_id):
            return jsonify(pred)
    return jsonify({"error": "Prediction not found"}), 404


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear prediction history"""
    history_path = Path(app.config['PREDICTION_HISTORY'])
    if history_path.exists():
        history_path.unlink()
    global STATS
    STATS['total_predictions'] = 0
    STATS['obfuscated_count'] = 0
    STATS['benign_count'] = 0
    return jsonify({"message": "History cleared"})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current inference configuration"""
    return jsonify({
        "inference_config": INFERENCE_CONFIG,
        "pipeline_ready": INFERENCE_PIPELINE is not None,
        "feature_dim": INFERENCE_PIPELINE.expected_dim if INFERENCE_PIPELINE else None
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update inference configuration"""
    data = request.json
    # This would require reloading the pipeline
    # For now, just return current config
    return jsonify({
        "message": "Config update requires server restart",
        "current_config": INFERENCE_CONFIG
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline": "ready" if INFERENCE_PIPELINE is not None else "not_ready",
        "models_available": len(list(Path("models/").glob("*.pkl")) + list(Path("models/").glob("*.json"))) if Path("models/").exists() else 0
    }
    status_code = 200 if INFERENCE_PIPELINE is not None else 503
    return jsonify(health), status_code


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get detailed metrics for all models"""
    results_dir = Path("results/")
    metrics = {}
    
    if results_dir.exists():
        for metrics_file in results_dir.glob("*_metrics.csv"):
            model_name = metrics_file.stem.replace("_metrics", "")
            df = pd.read_csv(metrics_file)
            metrics[model_name] = df.to_dict('records')[0]
            
            # Add chart availability
            metrics[model_name]['charts'] = {
                'confusion_matrix': (results_dir / f"{model_name}_confusion_matrix.png").exists(),
                'roc_curve': (results_dir / f"{model_name}_roc_curve.png").exists()
            }
    
    return jsonify(metrics)


@app.route('/api/features/info', methods=['GET'])
def get_feature_info():
    """Get information about feature extraction"""
    if INFERENCE_PIPELINE is None:
        return jsonify({"error": "Pipeline not loaded"}), 503
    
    feature_metadata_path = INFERENCE_CONFIG.get('feature_metadata')
    if feature_metadata_path and Path(feature_metadata_path).exists():
        with open(feature_metadata_path, 'r') as f:
            metadata = json.load(f)
        return jsonify({
            "feature_dim": metadata.get('feature_dim'),
            "opcode_ngrams": metadata.get('opcode_ngrams'),
            "opcode_max_features": metadata.get('opcode_max_features'),
            "api_max_features": metadata.get('api_max_features'),
            "enable_cfg": metadata.get('enable_cfg')
        })
    
    return jsonify({"error": "Feature metadata not available"}), 404


@app.route('/api/predict/batch', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple files"""
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files selected"}), 400
    
    if INFERENCE_PIPELINE is None:
        return jsonify({"error": "Inference pipeline not configured."}), 503
    
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for file in files:
        if file.filename == '':
            continue
        
        filepath = upload_dir / f"{int(time.time())}_{file.filename}"
        filepath_str = str(filepath.resolve())
        file.save(filepath_str)
        
        try:
            result = INFERENCE_PIPELINE.predict_file(filepath_str)
            results.append({
                "filename": file.filename,
                "is_obfuscated": result['label'] == 1,
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities'],
                "feature_count": result['feature_count']
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            if filepath.exists():
                try:
                    filepath.unlink()
                except:
                    pass
    
    return jsonify({"results": results, "count": len(results)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

