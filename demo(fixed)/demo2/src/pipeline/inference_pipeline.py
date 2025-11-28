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
        self.enable_explainability = enable_explainability
        self.top_features = top_features
        self.model = self._load_model()
        # Get expected dimension from model (more reliable than metadata)
        self.expected_dim = self._get_model_expected_dim()
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
    
    def _get_model_expected_dim(self) -> int:
        """Get expected feature dimension from the loaded model."""
        try:
            if self.model_type == 'random_forest':
                if hasattr(self.model.model, 'n_features_in_'):
                    return int(self.model.model.n_features_in_)
                elif hasattr(self.model.model, 'feature_importances_'):
                    return len(self.model.model.feature_importances_)
            elif self.model_type == 'xgboost':
                try:
                    return int(self.model.model.get_booster().num_feature())
                except:
                    if hasattr(self.model.model, 'feature_importances_'):
                        return len(self.model.model.feature_importances_)
            elif self.model_type == 'neural_network':
                if hasattr(self.model, 'input_size'):
                    return int(self.model.input_size)
                # Try to get from first layer
                try:
                    first_layer = list(self.model.model.modules())[1]
                    if hasattr(first_layer, 'in_features'):
                        return int(first_layer.in_features)
                except:
                    pass
        except Exception as exc:
            logger.warning(f"Could not get expected dimension from model: {exc}")
        
        # Fall back to metadata
        if self.feature_pipeline.expected_dim:
            logger.info(f"Using expected dimension from metadata: {self.feature_pipeline.expected_dim}")
            return self.feature_pipeline.expected_dim
        
        logger.warning("Could not determine expected dimension. Using extracted features as-is.")
        return None

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

        # Use model's expected dimension
        expected_dim = self.expected_dim
        if expected_dim is None:
            expected_dim = len(vector)
            logger.warning(f"No expected dimension set. Using extracted dimension: {expected_dim}")
        else:
            logger.info(f"Aligning features: extracted {len(vector)} features, model expects {expected_dim}")
        
        # Align features to match model's expected dimension
        padded_vector = self.feature_pipeline.pad_vector(vector, expected_dim)

        if len(feature_names) < expected_dim:
            padding_names = [f"_pad_{i}" for i in range(expected_dim - len(feature_names))]
            feature_names = feature_names + padding_names
        elif len(feature_names) > expected_dim:
            feature_names = feature_names[:expected_dim]

        return {
            'vector': padded_vector.reshape(1, -1),
            'feature_names': feature_names,
            'raw_dim': len(vector),
            'aligned_dim': len(padded_vector)
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
        
        # Determine final prediction based on probabilities (more reliable than label)
        # This handles cases where label might be wrong due to class order issues
        prob_obf = prob_map.get('obfuscated', 0.0)
        prob_ben = prob_map.get('benign', 0.0)
        
        # Use probabilities to determine the actual prediction
        is_obfuscated_from_probs = prob_obf > prob_ben
        is_obfuscated_from_label = (prediction == 1)
        
        # Prefer probability-based decision, but log if there's a mismatch
        if is_obfuscated_from_probs != is_obfuscated_from_label:
            logger.warning(
                f"Label/Probability mismatch! Label={prediction} (label says {'obfuscated' if is_obfuscated_from_label else 'benign'}), "
                f"but probabilities: benign={prob_ben:.4f}, obfuscated={prob_obf:.4f}. Using probability-based decision."
            )
        
        # Use probability-based decision as the final truth
        final_is_obfuscated = is_obfuscated_from_probs
        final_prediction = 'Obfuscated' if final_is_obfuscated else 'Benign'
        final_label = 1 if final_is_obfuscated else 0

        result = {
            'prediction': final_prediction,
            'label': int(final_label),
            'label_raw': int(prediction),  # Keep raw label for debugging
            'confidence': confidence,
            'probabilities': prob_map,
            'feature_count': features['raw_dim'],
            'model_type': self.model_type,
        }

        explanations = self._explain(vector, features['feature_names'])
        if explanations:
            result['top_contributors'] = explanations

        return result

