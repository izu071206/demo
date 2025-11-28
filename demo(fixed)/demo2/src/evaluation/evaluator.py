import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

class ModelEvaluator:
    def __init__(self, output_dir="data/evaluation_results"):
        """
        Khởi tạo bộ đánh giá, tự động tạo thư mục lưu kết quả.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, model, X_test, y_test, model_name="model"):
        """
        Đánh giá model và lưu metrics ra file JSON.
        """
        print(f"[*] Starting evaluation for {model_name}...")
        
        # 1. Dự đoán nhãn
        y_pred = model.predict(X_test)
        
        # 2. Dự đoán xác suất (để vẽ ROC curve)
        # Kiểm tra xem model có hỗ trợ predict_proba không
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            # Lấy cột xác suất của lớp 1 (Malware/Obfuscated)
            y_score = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        else:
            # Fallback nếu không có predict_proba (dùng chính y_pred)
            y_score = y_pred

        # 3. Tính toán các chỉ số cơ bản
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
        }

        # 4. Tính Confusion Matrix (Chuyển về list để lưu JSON)
        cm = confusion_matrix(y_test, y_pred)
        cm_list = cm.tolist()

        # 5. Tính ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = float(auc(fpr, tpr))

        # 6. Cấu trúc dữ liệu để lưu
        eval_result = {
            "model_name": model_name,
            "metrics": metrics,
            "confusion_matrix": cm_list,
            "roc_data": {
                "fpr": fpr.tolist(), # Chuyển numpy array thành list
                "tpr": tpr.tolist(),
                "auc": round(roc_auc, 4)
            }
        }

        # 7. Lưu file JSON
        # Tên file ví dụ: random_forest_metrics.json
        filename = f"{model_name.lower().replace(' ', '_')}_metrics.json"
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(eval_result, f, indent=4)
            
        print(f"[+] Evaluation results saved to: {file_path}")
        return eval_result