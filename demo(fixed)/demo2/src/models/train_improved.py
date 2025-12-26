import sys
import os
import yaml
import pickle
import numpy as np

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

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
    
    # 1. Load data từ .pkl files (FIXED: đổi từ .csv sang .pkl)
    print("[*] Loading training data...")
    
    # Load train data
    train_path = os.path.join(project_root, "data", "processed", "train_features.pkl")
    if not os.path.exists(train_path):
        print(f"[!] Error: Training data not found at {train_path}")
        print(f"[!] Please run: python main.py generate-dataset")
        return
    
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # Load validation data (if exists)
    val_path = os.path.join(project_root, "data", "processed", "val_features.pkl")
    X_val, y_val = None, None
    if os.path.exists(val_path):
        with open(val_path, 'rb') as f:
            X_val, y_val = pickle.load(f)
    
    # Load test data
    test_path = os.path.join(project_root, "data", "processed", "test_features.pkl")
    if not os.path.exists(test_path):
        print(f"[!] Warning: Test data not found at {test_path}")
        X_test, y_test = None, None
    else:
        with open(test_path, 'rb') as f:
            X_test, y_test = pickle.load(f)
    
    print(f"[+] Loaded {len(X_train)} training samples")
    if X_val is not None:
        print(f"[+] Loaded {len(X_val)} validation samples")
    if X_test is not None:
        print(f"[+] Loaded {len(X_test)} test samples")
    
    # Calculate imbalance ratio for XGBoost
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"[*] Data Imbalance Ratio (Neg/Pos): {imbalance_ratio:.2f}")

    # Prepare output directories
    eval_dir = os.path.join(project_root, "data", "evaluation_results")
    save_dir = os.path.join(project_root, "models")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    models_to_train = config.get('models', ['random_forest', 'xgboost'])
    
    # 2. Train each model
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"[*] Training {model_type.upper()}")
        print(f"{'='*60}")

        try:
            model_wrapper = None
            
            if model_type == 'random_forest':
                rf_params = config.get('random_forest', {})
                model_wrapper = RandomForestModel(**rf_params)
                
            elif model_type == 'xgboost':
                xgb_params = config.get('xgboost', {}).copy()
                if xgb_params.get('scale_pos_weight') == 'auto':
                    xgb_params['scale_pos_weight'] = imbalance_ratio
                model_wrapper = XGBoostModel(**xgb_params)
                
            elif model_type == 'neural_network':
                nn_params = config.get('neural_network', {})
                model_wrapper = NeuralNetworkModel(**nn_params)
            else:
                print(f"[!] Unknown model type: {model_type}")
                continue

            # Train
            print(f"[*] Training {model_type}...")
            history = model_wrapper.train(X_train, y_train, X_val, y_val)
            
            # Save model
            if model_type == 'xgboost':
                model_filename = f"{model_type}_model.json"
            elif model_type == 'neural_network':
                model_filename = f"{model_type}_model.pt"
            else:
                model_filename = f"{model_type}_model.pkl"
            
            model_path = os.path.join(save_dir, model_filename)
            model_wrapper.save(model_path)
            print(f"[+] Model saved to {model_path}")

            # Evaluate on test set (if exists)
            if X_test is not None and y_test is not None:
                print(f"[*] Evaluating {model_type} on test set...")
                evaluator = ModelEvaluator(output_dir=eval_dir)
                metrics = evaluator.evaluate(
                    model_wrapper, 
                    X_test, 
                    y_test, 
                    model_name=model_type
                )
                print(f"    -> Test Accuracy: {metrics['metrics']['accuracy']:.4f}")
                print(f"    -> Test F1 Score: {metrics['metrics']['f1_score']:.4f}")
            else:
                print(f"[!] Skipping evaluation (no test set)")

        except Exception as e:
            print(f"[!] Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("[+] Training completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_model()