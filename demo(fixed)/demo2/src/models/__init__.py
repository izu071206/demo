"""
Machine Learning Models
Random Forest, XGBoost, và Neural Network
"""

from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

# Conditional import for Neural Network (requires torch)
try:
    from .neural_network_model import NeuralNetworkModel
    __all__ = ['RandomForestModel', 'XGBoostModel', 'NeuralNetworkModel']
except (ImportError, OSError) as e:
    # Skip Neural Network if torch is not available or has DLL issues
    NeuralNetworkModel = None
    __all__ = ['RandomForestModel', 'XGBoostModel']

