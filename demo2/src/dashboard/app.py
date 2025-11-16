"""
Dashboard Web Application
Giao diện gọn nhẹ để visualize kết quả
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
from pathlib import Path
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/upload/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # Load best model (simplified - should load from config)
        # For now, return mock prediction
        prediction = {
            "is_obfuscated": True,
            "confidence": 0.85,
            "model": "RandomForest"
        }
        
        return jsonify(prediction)
    
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

