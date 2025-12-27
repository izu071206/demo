"""
Retrain Models with Extensive Debugging
Use this to identify and fix training issues
"""

import sys
import os
from pathlib import Path
import logging
import pickle
import numpy as np
import yaml

# Setup path
current_dir = Path(__file__).parent
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.evaluation.evaluator import ModelEvaluator

# Detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('retrain_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def load_and_verify_data():
    """Load training data and verify it's correct."""
    logger.info("="*80)
    logger.info("LOADING AND VERIFYING DATASET")
    logger.info("="*80)
    
    processed_dir = project_root / "data" / "processed"
    
    # Load train
    train_path = processed_dir / "train_features.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    logger.info(f"\nTRAIN SET:")
    logger.info(f"  X shape: {X_train.shape}")
    logger.info(f"  y shape: {y_train.shape}")
    logger.info(f"  y dtype: {y_train.dtype}")
    
    # Verify labels
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    logger.info(f"  Unique labels: {unique_labels}")
    logger.info(f"  Label distribution:")
    for label, count in zip(unique_labels, label_counts):
        logger.info(f"    Label {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Check for label issues
    if len(unique_labels) != 2:
        logger.error(f"  ERROR: Expected 2 classes, found {len(unique_labels)}")
    
    if not np.array_equal(unique_labels, [0, 1]):
        logger.warning(f"  WARNING: Labels are not [0, 1]: {unique_labels}")
    
    # Verify features
    logger.info(f"\n  Feature statistics:")
    logger.info(f"    Non-zero features per sample: {np.count_nonzero(X_train, axis=1).mean():.1f} ± {np.count_nonzero(X_train, axis=1).std():.1f}")
    logger.info(f"    Feature sum per sample: {np.sum(X_train, axis=1).mean():.4f} ± {np.sum(X_train, axis=1).std():.4f}")
    logger.info(f"    Feature mean: {np.mean(X_train):.6f}")
    logger.info(f"    Feature std: {np.std(X_train):.6f}")
    
    # Check if all samples are identical (major red flag!)
    if len(X_train) > 1:
        first_sample = X_train[0]
        all_identical = True
        for i in range(1, min(10, len(X_train))):
            if not np.array_equal(first_sample, X_train[i]):
                all_identical = False
                break
        
        if all_identical:
            logger.error("  ❌ CRITICAL: First 10 samples are IDENTICAL!")
            logger.error("  This means feature extraction is broken!")
        else:
            # Check diversity
            unique_samples = len(np.unique(X_train, axis=0))
            logger.info(f"  ✅ Unique feature vectors: {unique_samples}/{len(X_train)} ({unique_samples/len(X_train)*100:.1f}%)")
    
    # Sample some features
    logger.info(f"\n  Sample features (first 3 samples, first 10 dimensions):")
    for i in range(min(3, len(X_train))):
        logger.info(f"    Sample {i} (label={y_train[i]}): {X_train[i][:10]}")
    
    # Load validation (if exists)
    val_path = processed_dir / "val_features.pkl"
    X_val, y_val = None, None
    if val_path.exists():
        with open(val_path, 'rb') as f:
            X_val, y_val = pickle.load(f)
        logger.info(f"\nVAL SET: X shape={X_val.shape}, y shape={y_val.shape}")
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        logger.info(f"  Val labels: {unique_val}, counts: {counts_val}")
    
    # Load test
    test_path = processed_dir / "test_features.pkl"
    X_test, y_test = None, None
    if test_path.exists():
        with open(test_path, 'rb') as f:
            X_test, y_test = pickle.load(f)
        logger.info(f"\nTEST SET: X shape={X_test.shape}, y shape={y_test.shape}")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        logger.info(f"  Test labels: {unique_test}, counts: {counts_test}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model_with_verification(model_wrapper, X_train, y_train, X_val, y_val, 
                                  X_test, y_test, model_name):
    """Train a model with extensive verification."""
    logger.info("\n" + "="*80)
    logger.info(f"TRAINING {model_name.upper()}")
    logger.info("="*80)
    
    # Train
    logger.info("\nStarting training...")
    history = model_wrapper.train(X_train, y_train, X_val, y_val)
    
    logger.info(f"\nTraining completed:")
    for key, val in history.items():
        logger.info(f"  {key}: {val:.4f}")
    
    # Test on training data (should be high accuracy)
    logger.info(f"\nVerifying on training data:")
    train_preds = model_wrapper.predict(X_train)
    train_probs = model_wrapper.predict_proba(X_train)
    
    train_acc = np.mean(train_preds == y_train)
    logger.info(f"  Training accuracy: {train_acc:.4f}")
    
    # Check prediction distribution
    unique_preds, pred_counts = np.unique(train_preds, return_counts=True)
    logger.info(f"  Prediction distribution:")
    for label, count in zip(unique_preds, pred_counts):
        logger.info(f"    Predicted {label}: {count} ({count/len(train_preds)*100:.1f}%)")
    
    # Check if model is predicting only one class
    if len(unique_preds) == 1:
        logger.error(f"  ❌ CRITICAL: Model predicts only class {unique_preds[0]}!")
        logger.error(f"  This means the model didn't learn anything useful!")
    
    # Sample predictions
    logger.info(f"\n  Sample predictions (first 10):")
    for i in range(min(10, len(X_train))):
        logger.info(f"    Sample {i}: true={y_train[i]}, pred={train_preds[i]}, " +
                   f"probs=[{train_probs[i][0]:.4f}, {train_probs[i][1]:.4f}]")
    
    # Test on test set
    if X_test is not None and y_test is not None:
        logger.info(f"\nEvaluating on test set:")
        test_preds = model_wrapper.predict(X_test)
        test_probs = model_wrapper.predict_proba(X_test)
        test_acc = np.mean(test_preds == y_test)
        logger.info(f"  Test accuracy: {test_acc:.4f}")
        
        # Check if predictions vary
        unique_test_preds, test_pred_counts = np.unique(test_preds, return_counts=True)
        logger.info(f"  Test prediction distribution:")
        for label, count in zip(unique_test_preds, test_pred_counts):
            logger.info(f"    Predicted {label}: {count} ({count/len(test_preds)*100:.1f}%)")
        
        # Sample test predictions
        logger.info(f"\n  Sample test predictions (first 10):")
        for i in range(min(10, len(X_test))):
            logger.info(f"    Sample {i}: true={y_test[i]}, pred={test_preds[i]}, " +
                       f"probs=[{test_probs[i][0]:.4f}, {test_probs[i][1]:.4f}]")
    
    return history


def main():
    """Main retraining function."""
    try:
        # Load and verify data
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_verify_data()
        
        # Load config
        config_path = project_root / "config" / "train_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        train_config = config.get('training', config)
        
        # Calculate imbalance ratio for XGBoost
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
        logger.info(f"\nClass imbalance ratio (Neg/Pos): {imbalance_ratio:.2f}")
        
        # Prepare output directories
        models_dir = project_root / "models"
        eval_dir = project_root / "data" / "evaluation_results"
        models_dir.mkdir(exist_ok=True)
        eval_dir.mkdir(exist_ok=True)
        
        # Train Random Forest
        logger.info("\n\n" + "="*80)
        logger.info("RANDOM FOREST")
        logger.info("="*80)
        
        rf_params = train_config.get('random_forest', {})
        rf_model = RandomForestModel(**rf_params)
        
        train_model_with_verification(
            rf_model, X_train, y_train, X_val, y_val, X_test, y_test,
            "Random Forest"
        )
        
        # Save
        rf_path = models_dir / "random_forest_model.pkl"
        rf_model.save(str(rf_path))
        logger.info(f"\n✅ Random Forest saved to {rf_path}")
        
        # Evaluate
        if X_test is not None and y_test is not None:
            evaluator = ModelEvaluator(output_dir=str(eval_dir))
            evaluator.evaluate(rf_model, X_test, y_test, model_name="random_forest")
        
        # Train XGBoost
        logger.info("\n\n" + "="*80)
        logger.info("XGBOOST")
        logger.info("="*80)
        
        xgb_params = train_config.get('xgboost', {}).copy()
        if xgb_params.get('scale_pos_weight') == 'auto':
            xgb_params['scale_pos_weight'] = imbalance_ratio
            logger.info(f"Using scale_pos_weight={imbalance_ratio:.2f}")
        
        xgb_model = XGBoostModel(**xgb_params)
        
        train_model_with_verification(
            xgb_model, X_train, y_train, X_val, y_val, X_test, y_test,
            "XGBoost"
        )
        
        # Save
        xgb_path = models_dir / "xgboost_model.json"
        xgb_model.save(str(xgb_path))
        logger.info(f"\n✅ XGBoost saved to {xgb_path}")
        
        # Evaluate
        if X_test is not None and y_test is not None:
            evaluator = ModelEvaluator(output_dir=str(eval_dir))
            evaluator.evaluate(xgb_model, X_test, y_test, model_name="xgboost")
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info("\nCheck retrain_debug.log for detailed information")
        
    except Exception as e:
        logger.error(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()