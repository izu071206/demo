"""
Random Forest Model
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
from typing import Optional, Dict
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 min_samples_split: int = 5, min_samples_leaf: int = 2,
                 random_state: int = 42, class_weight: Optional[str] = None):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
        """
        super().__init__("RandomForest")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train Random Forest model
        """
        logger.info(f"Training {self.model_name}...")
        
        self.model.fit(X_train, y_train)
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
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

