"""
Evaluation Script
"""

import argparse
import pickle
import yaml
import logging
from pathlib import Path

from src.models import RandomForestModel, XGBoostModel
try:
    from src.models import NeuralNetworkModel
except (ImportError, AttributeError):
    NeuralNetworkModel = None
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_type: str, model_path: str):
    """Load trained model"""
    if model_type == 'random_forest':
        model = RandomForestModel()
    elif model_type == 'xgboost':
        model = XGBoostModel()
    elif model_type == 'neural_network':
        if NeuralNetworkModel is None:
            raise ValueError("Neural Network model is not available (torch DLL issue)")
        model = NeuralNetworkModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=['random_forest', 'xgboost', 'neural_network'],
                       help="Type of model")
    parser.add_argument("--test_data", type=str, default="data/processed/test_features.pkl",
                       help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="results/",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # Load model
    logger.info(f"Loading {args.model_type} model from {args.model}")
    model = load_model(args.model_type, args.model)
    
    # Evaluate
    evaluator = ModelEvaluator(model, args.model_type)
    metrics = evaluator.evaluate(X_test, y_test, args.output_dir)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

