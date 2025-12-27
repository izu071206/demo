"""
Training Script - REFACTORED VERSION
Shared preprocessing, fixed random states, feature importance logging
"""

import sys
import os
import yaml
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Fix path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network_model import NeuralNetworkModel
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    return full_config.get('training', full_config)


def load_pickle_split(data_path: str) -> tuple:
    """
    Load train/val/test split from pickle file.
    
    Args:
        data_path: Path to pickle file
        
    Returns:
        Tuple of (features, labels)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        features, labels = pickle.load(f)
    
    logger.info(f"Loaded {len(features)} samples from {data_path}")
    return features, labels


def check_class_imbalance(y: np.ndarray, split_name: str) -> Dict[str, float]:
    """
    Check class distribution and return imbalance metrics.
    
    Args:
        y: Labels array
        split_name: Name of split (train/val/test)
        
    Returns:
        Dictionary with imbalance metrics
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    metrics = {}
    for label, count in zip(unique, counts):
        percentage = count / total * 100
        metrics[f'class_{label}_count'] = int(count)
        metrics[f'class_{label}_percentage'] = float(percentage)
        logger.info(f"{split_name} - Class {label}: {count} samples ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    if len(unique) == 2:
        pos_count = counts[unique == 1][0] if 1 in unique else 0
        neg_count = counts[unique == 0][0] if 0 in unique else 0
        if pos_count > 0:
            imbalance_ratio = neg_count / pos_count
            metrics['imbalance_ratio'] = float(imbalance_ratio)
            logger.info(f"{split_name} - Imbalance ratio (neg/pos): {imbalance_ratio:.2f}")
        else:
            metrics['imbalance_ratio'] = float('inf')
            logger.warning(f"{split_name} - No positive samples!")
    
    return metrics


def apply_preprocessing(X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                       X_test: Optional[np.ndarray] = None,
                       method: str = 'standard') -> tuple:
    """
    Apply shared preprocessing to all splits.
    CRITICAL: Same preprocessing must be applied to train, val, and test.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        method: Preprocessing method ('standard', 'minmax', 'none')
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    if method == 'none':
        logger.info("Skipping preprocessing (method='none')")
        return X_train, X_val, X_test, None
    
    if method == 'standard':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        logger.info("Applied StandardScaler to training data")
        
        X_val_scaled = None
        if X_val is not None and len(X_val) > 0:
            X_val_scaled = scaler.transform(X_val)
            logger.info("Applied StandardScaler to validation data")
        
        X_test_scaled = None
        if X_test is not None and len(X_test) > 0:
            X_test_scaled = scaler.transform(X_test)
            logger.info("Applied StandardScaler to test data")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    else:
        logger.warning(f"Unknown preprocessing method: {method}, skipping")
        return X_train, X_val, X_test, None


def log_feature_importance(model_wrapper, model_name: str, top_n: int = 20):
    """
    Log feature importance for tree-based models.
    
    Args:
        model_wrapper: Model wrapper with get_feature_importance method
        model_name: Name of model
        top_n: Number of top features to log
    """
    try:
        if hasattr(model_wrapper, 'get_feature_importance'):
            importances = model_wrapper.get_feature_importance()
            
            if importances is not None and len(importances) > 0:
                # Get top N features
                top_indices = np.argsort(importances)[::-1][:top_n]
                top_importances = importances[top_indices]
                
                logger.info(f"{model_name} - Top {top_n} most important features:")
                for i, (idx, imp) in enumerate(zip(top_indices, top_importances)):
                    logger.info(f"  {i+1}. Feature {idx}: {imp:.6f}")
                
                # Log statistics
                logger.info(f"{model_name} - Feature importance stats:")
                logger.info(f"  Mean: {np.mean(importances):.6f}")
                logger.info(f"  Std: {np.std(importances):.6f}")
                logger.info(f"  Max: {np.max(importances):.6f}")
                logger.info(f"  Min: {np.min(importances):.6f}")
            else:
                logger.warning(f"{model_name} - No feature importance available")
    except Exception as e:
        logger.warning(f"{model_name} - Failed to get feature importance: {e}")


def check_overfitting(train_score: float, val_score: float, threshold: float = 0.1) -> bool:
    """
    Check for overfitting by comparing train and validation scores.
    
    Args:
        train_score: Training accuracy/score
        val_score: Validation accuracy/score
        threshold: Threshold for overfitting detection
        
    Returns:
        True if overfitting detected
    """
    gap = train_score - val_score
    if gap > threshold:
        logger.warning(
            f"⚠️ Potential overfitting detected! "
            f"Train score: {train_score:.4f}, Val score: {val_score:.4f}, Gap: {gap:.4f}"
        )
        return True
    else:
        logger.info(
            f"✓ No overfitting detected. "
            f"Train score: {train_score:.4f}, Val score: {val_score:.4f}, Gap: {gap:.4f}"
        )
        return False


def train_model(config_rel_path: str = "config/train_config.yaml"):
    """
    Main training function.
    REFACTORED: Shared preprocessing, fixed random states, comprehensive logging.
    """
    training_start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting model training")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(config_rel_path)
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Load data from pickle files (from dataset generation)
    train_data_path = os.path.join(project_root, config.get('train_data', 'data/processed/train_features.pkl'))
    val_data_path = os.path.join(project_root, config.get('val_data', 'data/processed/val_features.pkl'))
    test_data_path = os.path.join(project_root, config.get('test_data', 'data/processed/test_features.pkl'))
    
    logger.info("Loading data splits...")
    try:
        X_train, y_train = load_pickle_split(train_data_path)
        X_val, y_val = load_pickle_split(val_data_path)
        X_test, y_test = load_pickle_split(test_data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Verify feature dimensions are consistent
    dims = [X_train.shape[1], X_val.shape[1], X_test.shape[1]]
    if len(set(dims)) > 1:
        logger.error(f"CRITICAL: Feature dimension mismatch! Train: {dims[0]}, Val: {dims[1]}, Test: {dims[2]}")
        return
    
    feature_dim = dims[0]
    logger.info(f"Feature dimension: {feature_dim}")
    
    # Check class imbalance
    logger.info("\n" + "=" * 60)
    logger.info("Class Distribution Analysis")
    logger.info("=" * 60)
    train_imbalance = check_class_imbalance(y_train, "Train")
    val_imbalance = check_class_imbalance(y_val, "Val")
    test_imbalance = check_class_imbalance(y_test, "Test")
    
    # Calculate imbalance ratio for XGBoost
    pos_count = int(train_imbalance.get('class_1_count', 0))
    neg_count = int(train_imbalance.get('class_0_count', 0))
    imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
    logger.info(f"\nImbalance ratio (neg/pos) for XGBoost: {imbalance_ratio:.2f}")
    
    # Apply shared preprocessing
    logger.info("\n" + "=" * 60)
    logger.info("Applying Preprocessing")
    logger.info("=" * 60)
    preprocessing_method = config.get('preprocessing', 'standard')
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_preprocessing(
        X_train, X_val, X_test, method=preprocessing_method
    )
    
    # Setup output directories
    eval_dir = os.path.join(project_root, config.get('results_dir', 'data/evaluation_results'))
    save_dir = os.path.join(project_root, config.get('model_save_dir', 'models'))
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save scaler if used
    if scaler is not None:
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
    
    # Train models
    models_to_train = config.get('models', ['random_forest'])
    model_timings = {}
    
    for model_type in models_to_train:
        model_start_time = time.time()
        logger.info("\n" + "=" * 60)
        logger.info(f"Training Model: {model_type.upper()}")
        logger.info("=" * 60)
        
        try:
            model_wrapper = None
            
            if model_type == 'random_forest':
                rf_params = config.get('random_forest', {})
                # Ensure random_state is set
                if 'random_state' not in rf_params:
                    rf_params['random_state'] = 42
                model_wrapper = RandomForestModel(**rf_params)
                
            elif model_type == 'xgboost':
                xgb_params = config.get('xgboost', {}).copy()
                # Fix scale_pos_weight
                if xgb_params.get('scale_pos_weight') == 'auto':
                    xgb_params['scale_pos_weight'] = imbalance_ratio
                    logger.info(f"Set scale_pos_weight to {imbalance_ratio:.2f}")
                # Ensure random_state is set
                if 'random_state' not in xgb_params:
                    xgb_params['random_state'] = 42
                model_wrapper = XGBoostModel(**xgb_params)
                
            elif model_type == 'neural_network':
                nn_params = config.get('neural_network', {})
                model_wrapper = NeuralNetworkModel(**nn_params)
            
            else:
                logger.warning(f"Unknown model type '{model_type}'. Skipping.")
                continue
            
            # Train model
            logger.info(f"Training {model_type}...")
            history = model_wrapper.train(
                X_train_scaled, y_train,
                X_val=X_val_scaled, y_val=y_val
            )
            
            # Check for overfitting
            if 'train_accuracy' in history and 'val_accuracy' in history:
                check_overfitting(history['train_accuracy'], history['val_accuracy'])
            
            # Log feature importance
            log_feature_importance(model_wrapper, model_type)
            
            # Save model
            model_filename = f"{model_type}_model.pkl" if model_type != 'xgboost' else f"{model_type}_model.json"
            model_path = os.path.join(save_dir, model_filename)
            model_wrapper.save(model_path)
            logger.info(f"✓ Model saved to {model_path}")
            
            # Evaluate on test set
            logger.info(f"Evaluating {model_type} on test set...")
            evaluator = ModelEvaluator(output_dir=eval_dir)
            real_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
            metrics = evaluator.evaluate(real_model, X_test_scaled, y_test, model_name=model_type)
            
            logger.info(f"Test Results:")
            logger.info(f"  Accuracy: {metrics['metrics']['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['metrics'].get('precision', 'N/A')}")
            logger.info(f"  Recall: {metrics['metrics'].get('recall', 'N/A')}")
            logger.info(f"  F1 Score: {metrics['metrics'].get('f1_score', 'N/A')}")
            
            # Record timing
            model_time = time.time() - model_start_time
            model_timings[model_type] = model_time
            logger.info(f"⏱️  {model_type} training time: {model_time:.2f}s ({model_time/60:.2f} min)")
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            model_time = time.time() - model_start_time
            model_timings[model_type] = model_time
    
    # Calculate total time
    total_training_time = time.time() - training_start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)
    logger.info("⏱️  Timing Summary:")
    for model_type, model_time in model_timings.items():
        logger.info(f"  {model_type}: {model_time:.2f}s ({model_time/60:.2f} min)")
    logger.info(f"  Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_model()
