"""
Dashboard Web Application
Giao diện gọn nhẹ để visualize kết quả
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request, send_file

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import InferencePipeline  # noqa: E402

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/upload/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFERENCE_PIPELINE = None
INFERENCE_CONFIG = {}


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


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


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
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        if INFERENCE_PIPELINE is None:
            load_inference_pipeline()
        if INFERENCE_PIPELINE is None:
            return jsonify({"error": "Inference pipeline not configured."}), 503
        
        result = INFERENCE_PIPELINE.predict_file(filepath)
        payload = {
            "is_obfuscated": result['label'] == 1,
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "model": INFERENCE_CONFIG.get('model_name', result['model_type']),
            "feature_count": result['feature_count'],
        }
        if 'top_contributors' in result:
            payload['top_contributors'] = result['top_contributors']
        
        return jsonify(payload)
    
    except Exception as e:
        logger.error(f"Error predicting: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)


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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

