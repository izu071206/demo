"""
Base Model Class
Abstract base class cho tất cả ML models
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class cho ML models"""
    
    def __init__(self, model_name: str):
        """
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model
        """
        pass

