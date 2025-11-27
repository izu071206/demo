import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Khởi tạo Flask App
app = Flask(__name__, template_folder='templates')

# Cấu hình thư mục
UPLOAD_FOLDER = os.path.join('data', 'uploads')
EVAL_RESULTS_DIR = os.path.join('data', 'evaluation_results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_latest_metrics():
    """
    Hàm tìm file JSON kết quả đánh giá mới nhất để hiển thị.
    """
    if not os.path.exists(EVAL_RESULTS_DIR):
        return None
    
    # Lấy tất cả file .json
    files = [f for f in os.listdir(EVAL_RESULTS_DIR) if f.endswith('.json')]
    
    if not files:
        return None
    
    # Ưu tiên load file 'random_forest' hoặc lấy file đầu tiên tìm thấy
    target_file = next((f for f in files if 'random_forest' in f), files[0])
    
    full_path = os.path.join(EVAL_RESULTS_DIR, target_file)
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

@app.route('/')
def index():
    # Load metrics từ file JSON
    metrics_data = load_latest_metrics()
    return render_template('index.html', data=metrics_data)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Xử lý upload file và dự đoán (Logic giữ nguyên hoặc placeholder)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # --- LOGIC GỌI MODEL DỰ ĐOÁN Ở ĐÂY ---
        # (Để đơn giản cho Dashboard, ở đây trả về kết quả giả lập
        #  Bạn cần import inference_pipeline để chạy thật)
        
        result = {
            "filename": filename,
            "prediction": "Obfuscated", # Hoặc "Clean"
            "confidence": 0.95,
            "details": "Detected high entropy and abnormal control flow."
        }
        
        # Nếu muốn hiển thị kết quả trên trang mới hoặc popup, 
        # ở đây trả về JSON để JS xử lý alert
        return render_template('index.html', data=load_latest_metrics(), prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)