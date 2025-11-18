"""
XGBoost Model
"""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict
import logging
import pickle

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42,
                 base_score: float = 0.5, objective: str = "binary:logistic",
                 **extra_params):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random seed
        """
        super().__init__("XGBoost")
        
        validated_base_score = self._validate_base_score(base_score)
        
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
            **extra_params
        )
        
        if validated_base_score != base_score:
            logger.warning(
                "Adjusted base_score from %.4f to %.4f to satisfy logistic objective constraints",
                base_score, validated_base_score
            )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train XGBoost model
        """
        logger.info(f"Training {self.model_name}...")
        
        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_trained = True
        
        # Calculate training accuracy
        train_score = self.model.score(X_train, y_train)
        
        history = {
            'train_accuracy': train_score
        }
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_score = self.model.score(X_val, y_val)
            history['val_accuracy'] = val_score
            logger.info(f"Validation accuracy: {val_score:.4f}")
        elif X_val is not None and y_val is not None and len(X_val) == 0:
            logger.warning("Validation set is empty, skipping validation evaluation")
            
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
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
    
    def load(self, filepath: str):
        """Load model"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

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
            return eps
        if base_score >= 1.0:
            return 1.0 - eps
        return base_score

