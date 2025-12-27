"""
XGBoost Model - FIXED VERSION
Fixes:
1. Base score validation
2. Probability alignment
3. Label consistency (0=Benign, 1=Obfuscated)
"""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier with fixed probability handling"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42,
                 base_score: float = 0.5, objective: str = "binary:logistic",
                 scale_pos_weight: float = 1.0,
                 **extra_params):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random seed
            base_score: Initial prediction score
            scale_pos_weight: Balancing of positive and negative weights
        """
        super().__init__("XGBoost")
        
        # Validate base_score for logistic objective
        validated_base_score = self._validate_base_score(base_score)
        
        # Validate scale_pos_weight
        if isinstance(scale_pos_weight, str) and scale_pos_weight == 'auto':
            logger.warning("scale_pos_weight='auto' not valid for XGBoost, using 1.0")
            scale_pos_weight = 1.0
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            objective=objective,
            base_score=validated_base_score,
            scale_pos_weight=scale_pos_weight,
            **extra_params
        )
        
        if validated_base_score != base_score:
            logger.warning(
                "Adjusted base_score from %.4f to %.4f to satisfy logistic objective constraints",
                base_score, validated_base_score
            )
        
        logger.info(f"XGBoost initialized with scale_pos_weight={scale_pos_weight}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train XGBoost model
        """
        logger.info(f"Training {self.model_name}...")
        
        # Log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Training class distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
        # Setup evaluation
        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
            logger.info(f"Using validation set with {len(X_val)} samples")
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_trained = True
        
        # Log training results
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Test predictions on a few samples
        if len(X_train) > 0:
            test_preds = self.model.predict(X_train[:5])
            test_probs = self.model.predict_proba(X_train[:5])
            logger.info(f"Sample predictions on training data:")
            for i in range(min(5, len(X_train))):
                logger.info(f"  Sample {i}: true={y_train[i]}, pred={test_preds[i]}, " +
                          f"probs=[{test_probs[i][0]:.4f}, {test_probs[i][1]:.4f}]")
        
        history = {
            'train_accuracy': train_score
        }
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_score = self.model.score(X_val, y_val)
            history['val_accuracy'] = val_score
            logger.info(f"Validation accuracy: {val_score:.4f}")
        elif X_val is not None and y_val is not None and len(X_val) == 0:
            logger.warning("Validation set is empty, skipping validation evaluation")
        
        # Log class predictions on training data
        train_preds = self.model.predict(X_train)
        pred_unique, pred_counts = np.unique(train_preds, return_counts=True)
        logger.info(f"Training predictions distribution:")
        for label, count in zip(pred_unique, pred_counts):
            logger.info(f"  Class {label}: {count} predictions ({count/len(train_preds)*100:.1f}%)")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels with verification."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        logger.debug(f"XGBoost predictions: {np.unique(predictions, return_counts=True)}")
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities with explicit verification.
        Returns: Array of shape (n_samples, 2) where:
        - Column 0 = P(Benign) = P(y=0)
        - Column 1 = P(Obfuscated) = P(y=1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        probs = self.model.predict_proba(X)
        
        # Verify probability format
        if probs.shape[1] != 2:
            logger.warning(f"Unexpected probability shape: {probs.shape}, expected (n, 2)")
        
        # Log probabilities for debugging
        if len(probs) <= 5:
            logger.debug(f"XGBoost probabilities:")
            for i, prob in enumerate(probs):
                logger.debug(f"  Sample {i}: benign={prob[0]:.4f}, obf={prob[1]:.4f}")
        
        # Verify probabilities sum to 1
        prob_sums = np.sum(probs, axis=1)
        if not np.allclose(prob_sums, 1.0):
            logger.warning(f"Probabilities don't sum to 1.0: {prob_sums}")
        
        return probs
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """Save model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Verify classes are stored
        if hasattr(self.model, 'classes_'):
            logger.info(f"Model classes: {self.model.classes_}")
    
    def load(self, filepath: str):
        """Load model with verification"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        
        # Log model info
        try:
            num_features = self.model.get_booster().num_feature()
            logger.info(f"Model expects {num_features} features")
        except:
            pass
        
        # Check if classes are available
        if hasattr(self.model, 'classes_'):
            logger.info(f"Model classes: {self.model.classes_}")
        else:
            logger.warning("Model classes not available - may cause issues")
            # Set default classes
            self.model.classes_ = np.array([0, 1])
            logger.info(f"Set default classes: {self.model.classes_}")

    def _validate_base_score(self, base_score: float) -> float:
        """
        Ensure base_score satisfies logistic objective requirements.
        XGBoost expects base_score to be strictly within (0, 1) for logistic loss.
        """
        if np.isnan(base_score):
            logger.warning("base_score is NaN; resetting to 0.5")
            return 0.5
        
        eps = 1e-6
        if base_score <= 0.0:
            logger.warning(f"base_score {base_score} <= 0, adjusting to {eps}")
            return eps
        if base_score >= 1.0:
            logger.warning(f"base_score {base_score} >= 1, adjusting to {1.0 - eps}")
            return 1.0 - eps
        
        return base_score