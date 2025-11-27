import sys
import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- FIX PATH SETUP ---
# Đảm bảo python tìm thấy module src dù chạy từ đâu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------------

from src.features.feature_combiner import FeatureCombiner
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network_model import NeuralNetworkModel
from src.evaluation.evaluator import ModelEvaluator

def load_config(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        print(f"[!] Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        
    return full_config.get('training', full_config)

def train_model(config_rel_path="config/train_config.yaml"):
    print("[*] Loading configuration...")
    config = load_config(config_rel_path)
    
    # 1. Load và chuẩn bị dữ liệu
    print("[*] Preparing data...")
    data_path = os.path.join(project_root, "data", "processed", "features.csv")
    
    if not os.path.exists(data_path):
        print(f"[!] Error: Feature file not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['label', 'filename', 'filepath'], errors='ignore')
    y = df['label']

    # Split dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # --- LOGIC TÍNH IMBALANCE RATIO (Fix lỗi XGBoost 'auto') ---
    # Tính toán tỷ lệ mẫu Positive/Negative để gán cho scale_pos_weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"[*] Data Imbalance Ratio (Neg/Pos): {imbalance_ratio:.2f}")
    # -----------------------------------------------------------

    models_to_train = config.get('models', ['random_forest'])
    
    eval_dir = os.path.join(project_root, "data", "evaluation_results")
    save_dir = os.path.join(project_root, "models")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 2. Loop qua từng model để train
    for model_type in models_to_train:
        print(f"\n{'='*40}")
        print(f"[*] Processing Model: {model_type.upper()}")
        print(f"{'='*40}")

        try:
            model_wrapper = None
            
            if model_type == 'random_forest':
                rf_params = config.get('random_forest', {})
                model_wrapper = RandomForestModel(**rf_params)
                
            elif model_type == 'xgboost':
                # Copy config để không ảnh hưởng vòng lặp sau
                xgb_params = config.get('xgboost', {}).copy()
                
                # --- FIX: Thay thế 'auto' bằng số thực ---
                if xgb_params.get('scale_pos_weight') == 'auto':
                    print(f"    -> Replacing scale_pos_weight='auto' with {imbalance_ratio:.2f}")
                    xgb_params['scale_pos_weight'] = imbalance_ratio
                # ----------------------------------------
                
                model_wrapper = XGBoostModel(**xgb_params)
                
            elif model_type == 'neural_network':
                nn_params = config.get('neural_network', {})
                model_wrapper = NeuralNetworkModel(**nn_params)
            
            else:
                print(f"[!] Warning: Unknown model type '{model_type}'. Skipping.")
                continue

            # 3. Huấn luyện
            print(f"[*] Training {model_type}...")
            model_wrapper.train(X_train, y_train)

            # 4. Lưu model
            model_filename = f"{model_type}_model.pkl"
            model_path = os.path.join(save_dir, model_filename)
            
            if hasattr(model_wrapper, 'save'):
                model_wrapper.save(model_path)
            else:
                joblib.dump(model_wrapper.model, model_path)
            print(f"[+] Model saved to {model_path}")

            # 5. Đánh giá
            evaluator = ModelEvaluator(output_dir=eval_dir)
            real_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
            metrics = evaluator.evaluate(real_model, X_test, y_test, model_name=model_type)
            
            print(f"    -> Accuracy: {metrics['metrics']['accuracy']}")
            print(f"    -> F1 Score: {metrics['metrics']['f1_score']}")

        except Exception as e:
            print(f"[!] Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    train_model()