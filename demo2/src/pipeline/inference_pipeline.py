"""
Unified inference pipeline for the dashboard and CLI utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.features.feature_pipeline import FeaturePipeline
from src.models import NeuralNetworkModel, RandomForestModel, XGBoostModel

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Load trained model + feature pipeline and serve predictions."""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        feature_metadata: str,
        enable_explainability: bool = False,
        top_features: int = 5,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.feature_pipeline = FeaturePipeline.from_metadata(feature_metadata)
        self.expected_dim = self.feature_pipeline.expected_dim
        self.enable_explainability = enable_explainability
        self.top_features = top_features
        self.model = self._load_model()
        self.explainer = self._init_explainer() if enable_explainability else None

    def _load_model(self):
        loader = None
        if self.model_type == 'random_forest':
            loader = RandomForestModel()
        elif self.model_type == 'xgboost':
            loader = XGBoostModel()
        elif self.model_type == 'neural_network':
            loader = NeuralNetworkModel()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        loader.load(self.model_path)
        return loader

    def _init_explainer(self):
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Disable enable_explainability or install shap>=0.43.")
            return None

        if self.model_type in {'random_forest', 'xgboost'}:
            try:
                return shap.TreeExplainer(self.model.model)
            except Exception as exc:
                logger.warning("Failed to initialize SHAP explainer: %s", exc)
                return None
        logger.info("Explainability not supported for model_type=%s yet.", self.model_type)
        return None

    def _build_feature_vector(self, file_path: str) -> Dict:
        vector = self.feature_pipeline.build_feature_vector(file_path)
        feature_names = self.feature_pipeline.combiner.get_feature_names()

        if vector.size == 0:
            raise ValueError("No features extracted from file.")

        expected_dim = self.expected_dim or len(vector)
        padded_vector = self.feature_pipeline.pad_vector(vector, expected_dim)

        if len(feature_names) < expected_dim:
            padding_names = [f"_pad_{i}" for i in range(expected_dim - len(feature_names))]
            feature_names = feature_names + padding_names
        elif len(feature_names) > expected_dim:
            feature_names = feature_names[:expected_dim]

        return {
            'vector': padded_vector.reshape(1, -1),
            'feature_names': feature_names,
            'raw_dim': len(vector)
        }

    def _format_probabilities(self, probs: np.ndarray) -> Dict[str, float]:
        if probs.ndim == 2:
            probs = probs[0]
        if probs.size == 1:
            prob_obf = float(probs[0])
            prob_benign = 1.0 - prob_obf
        else:
            prob_benign = float(probs[0])
            prob_obf = float(probs[1])
        return {'benign': prob_benign, 'obfuscated': prob_obf}

    def _explain(self, feature_vector: np.ndarray, feature_names: list) -> Optional[list]:
        if self.explainer is None:
            return None
        try:
            shap_values = self.explainer.shap_values(feature_vector)
            if isinstance(shap_values, list):
                shap_vector = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vector = shap_values
            shap_scores = shap_vector[0]
            pairs = list(zip(feature_names, shap_scores))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            top_pairs = pairs[:self.top_features]
            return [{'feature': name, 'impact': float(score)} for name, score in top_pairs]
        except Exception as exc:
            logger.warning("Explainability calculation failed: %s", exc)
            return None

    def predict_file(self, file_path: str) -> Dict:
        features = self._build_feature_vector(file_path)
        vector = features['vector']

        prediction = self.model.predict(vector)[0]
        probabilities = self.model.predict_proba(vector)
        prob_map = self._format_probabilities(probabilities)
        confidence = max(prob_map.values())

        result = {
            'prediction': 'Obfuscated' if prediction == 1 else 'Benign',
            'label': int(prediction),
            'confidence': confidence,
            'probabilities': prob_map,
            'feature_count': features['raw_dim'],
            'model_type': self.model_type,
        }

        explanations = self._explain(vector, features['feature_names'])
        if explanations:
            result['top_contributors'] = explanations

        return result

