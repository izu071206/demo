"""
Training Script
Train tất cả models
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

from src.evaluation.evaluator import ModelEvaluator
from src.models import NeuralNetworkModel, RandomForestModel, XGBoostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                       help="Path to training config file")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", args.config)
        return
    
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)['training']
    
    def load_split(path: str):
        split_path = Path(path)
        if not split_path.exists():
            raise FileNotFoundError(f"Data split not found: {path}")
        with split_path.open('rb') as handle:
            return pickle.load(handle)
    
    try:
        logger.info("Loading training data...")
        X_train, y_train = load_split(config['train_data'])
        X_val, y_val = load_split(config['val_data'])
        X_test, y_test = load_split(config['test_data'])
    except FileNotFoundError as exc:
        logger.error(exc)
        return
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Train models
    models_to_train = config['models']
    Path(config['model_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    best_model = None
    best_score = 0.0
    best_model_name = None
    
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(len(y_train) - pos_count)
    imbalance_ratio = (neg_count / max(pos_count, 1)) if pos_count > 0 else 1.0
    
    for model_name in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*50}")
        
        # Create model
        if model_name == 'random_forest':
            rf_params = config['random_forest'].copy()
            model = RandomForestModel(**rf_params)
        elif model_name == 'xgboost':
            xgb_params = config['xgboost'].copy()
            if xgb_params.get('scale_pos_weight') == 'auto':
                xgb_params['scale_pos_weight'] = imbalance_ratio
            model = XGBoostModel(**xgb_params)
        elif model_name == 'neural_network':
            model = NeuralNetworkModel(**config['neural_network'])
        else:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue
        
        # Train
        try:
            history = model.train(X_train, y_train, X_val, y_val)
        except Exception as exc:
            logger.error("Training failed for %s: %s", model_name, exc)
            continue
        
        # Save model
        model_path = f"{config['model_save_dir']}/{model_name}_model"
        if model_name == 'neural_network':
            model_path += ".pt"
        elif model_name == 'xgboost':
            model_path += ".json"
        else:
            model_path += ".pkl"
        
        model.save(model_path)
        
        # Evaluate on test set
        evaluator = ModelEvaluator(model, model_name)
        metrics = evaluator.evaluate(X_test, y_test, config['results_dir'])
        
        # Track best model
        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model = model
            best_model_name = model_name
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Best model: {best_model_name} with F1-score: {best_score:.4f}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

