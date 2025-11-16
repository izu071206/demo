"""
Training Script
Train tất cả models
"""

import argparse
import pickle
import yaml
import logging
from pathlib import Path

from src.models import RandomForestModel, XGBoostModel, NeuralNetworkModel
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                       help="Path to training config file")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['training']
    
    # Load data
    logger.info("Loading training data...")
    with open(config['train_data'], 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    with open(config['val_data'], 'rb') as f:
        X_val, y_val = pickle.load(f)
    
    with open(config['test_data'], 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Train models
    models_to_train = config['models']
    Path(config['model_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    best_model = None
    best_score = 0.0
    best_model_name = None
    
    for model_name in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*50}")
        
        # Create model
        if model_name == 'random_forest':
            model = RandomForestModel(**config['random_forest'])
        elif model_name == 'xgboost':
            model = XGBoostModel(**config['xgboost'])
        elif model_name == 'neural_network':
            model = NeuralNetworkModel(**config['neural_network'])
        else:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue
        
        # Train
        history = model.train(X_train, y_train, X_val, y_val)
        
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

