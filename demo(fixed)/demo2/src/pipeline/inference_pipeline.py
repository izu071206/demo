"""
Unified inference pipeline - REFACTORED VERSION
Load exact schema from training, deterministic predictions, consistency checks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.feature_pipeline import FeaturePipeline
from src.models import NeuralNetworkModel, RandomForestModel, XGBoostModel

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Load trained model + feature pipeline and serve predictions.
    REFACTORED: Load exact schema from training, deterministic predictions, consistency checks.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str,
        feature_metadata: str,
        scaler_path: Optional[str] = None,
        enable_explainability: bool = False,
        top_features: int = 5,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model ('random_forest', 'xgboost', 'neural_network')
            feature_metadata: Path to feature metadata file (from training)
            scaler_path: Path to scaler file (if preprocessing was used)
            enable_explainability: Enable SHAP explanations
            top_features: Number of top features to show in explanations
        """
        self.model_path = model_path
        self.model_type = model_type
        
        # CRITICAL: Load feature pipeline with exact schema from training
        logger.info(f"Loading feature pipeline from metadata: {feature_metadata}")
        self.feature_pipeline = FeaturePipeline.from_metadata(feature_metadata)
        self.expected_dim = self.feature_pipeline.expected_dim
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        self.enable_explainability = enable_explainability
        self.top_features = top_features
        
        # Load model
        self.model = self._load_model()
        
        # Get model's expected dimension
        model_expected_dim = self._get_model_expected_dim()
        
        # Store original pipeline dimension for reference
        self.pipeline_original_dim = self.expected_dim
        
        # Handle dimension mismatch gracefully
        if model_expected_dim and model_expected_dim != self.expected_dim:
            logger.warning(
                f"⚠️ Dimension mismatch detected! "
                f"Model expects {model_expected_dim}, pipeline expects {self.expected_dim}. "
                f"Will use model's dimension ({model_expected_dim}) and pad/truncate features accordingly."
            )
            logger.warning(
                f"💡 Recommendation: Retrain models with new dataset to ensure consistency. "
                f"Run: python src/dataset/generate_dataset.py && python src/models/train.py"
            )
            # Use model's dimension instead of pipeline's
            self.expected_dim = model_expected_dim
            self.use_model_dimension = True
            logger.info(f"Using model's expected dimension: {self.expected_dim}")
        else:
            self.use_model_dimension = False
            if model_expected_dim:
                logger.info(f"✓ Dimension match: Model and pipeline both expect {model_expected_dim}")
            else:
                logger.warning("Could not detect model's expected dimension. Using pipeline dimension.")
        
        # Detect class order
        self.class_order = self._detect_class_order()
        logger.info(f"Model {model_type} class order: {self.class_order}")

    def _load_model(self):
        """Load trained model"""
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
        logger.info(f"Loaded {self.model_type} model from {self.model_path}")
        return loader
    
    def _detect_class_order(self) -> Dict[int, str]:
        """
        Detect class order từ model.
        CRITICAL: Đảm bảo mapping đúng giữa class index và label name.
        Standard: classes_ = [0, 1] means index 0 = benign (0), index 1 = obfuscated (1)
        """
        try:
            # Sklearn models có attribute classes_
            if hasattr(self.model.model, 'classes_'):
                classes = self.model.model.classes_
                logger.info(f"Detected classes_ from model: {classes}")
                
                # CRITICAL: Map probability index to actual class label
                # classes_ array tells us: classes_[0] = first class, classes_[1] = second class
                # For binary: classes_ = [0, 1] means prob[0] = P(class=0=benign), prob[1] = P(class=1=obfuscated)
                class_map = {}
                for idx, cls in enumerate(classes):
                    if cls == 0:
                        class_map[idx] = 'benign'
                    elif cls == 1:
                        class_map[idx] = 'obfuscated'
                    else:
                        logger.warning(f"Unknown class value: {cls}")
                        class_map[idx] = 'unknown'
                
                return class_map
            else:
                # Default: assume standard order
                logger.warning("Model doesn't have classes_ attribute, using default order")
                return {0: 'benign', 1: 'obfuscated'}
        except Exception as e:
            logger.error(f"Error detecting class order: {e}")
            # Default fallback
            return {0: 'benign', 1: 'obfuscated'}
    
    def _get_model_expected_dim(self) -> Optional[int]:
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
                try:
                    first_layer = list(self.model.model.modules())[1]
                    if hasattr(first_layer, 'in_features'):
                        return int(first_layer.in_features)
                except:
                    pass
        except Exception as exc:
            logger.warning(f"Could not get expected dimension from model: {exc}")
        
        return None

    def _init_explainer(self):
        """Initialize SHAP explainer if available"""
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
        """
        Build feature vector from file.
        CRITICAL: Sử dụng exact schema từ training, hoặc align với model's dimension.
        """
        # Extract features using pipeline (deterministic)
        vector = self.feature_pipeline.build_feature_vector(file_path)
        pipeline_dim = len(vector)
        
        logger.debug(f"Extracted {pipeline_dim} features from pipeline, model expects {self.expected_dim}")
        
        # CRITICAL: Align to model's expected dimension BEFORE preprocessing
        # This ensures scaler and model both get the correct dimension
        if pipeline_dim != self.expected_dim:
            logger.info(
                f"Aligning features: pipeline={pipeline_dim}, model={self.expected_dim}. "
                f"Will {'pad' if pipeline_dim < self.expected_dim else 'truncate'}."
            )
            
            # Pad or truncate to match model's expected dimension
            if pipeline_dim < self.expected_dim:
                # Pad with zeros
                padding = np.zeros(self.expected_dim - pipeline_dim, dtype=vector.dtype)
                vector = np.concatenate([vector, padding])
                logger.debug(f"Padded features from {pipeline_dim} to {self.expected_dim}")
            else:
                # Truncate
                vector = vector[:self.expected_dim]
                logger.debug(f"Truncated features from {pipeline_dim} to {self.expected_dim}")
        
        # Verify dimension after alignment
        if len(vector) != self.expected_dim:
            logger.error(
                f"CRITICAL: Dimension still mismatch after alignment! "
                f"Got {len(vector)}, expected {self.expected_dim}"
            )
            # Force align
            if len(vector) < self.expected_dim:
                vector = np.pad(vector, (0, self.expected_dim - len(vector)), mode='constant')
            else:
                vector = vector[:self.expected_dim]
        
        # Apply preprocessing if scaler is available
        if self.scaler is not None:
            # Check if scaler expects different dimension
            scaler_dim = None
            if hasattr(self.scaler, 'n_features_in_'):
                scaler_dim = self.scaler.n_features_in_
            elif hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                scaler_dim = len(self.scaler.mean_)
            
            if scaler_dim is not None and scaler_dim != len(vector):
                logger.warning(
                    f"Scaler expects {scaler_dim} features, but have {len(vector)}. "
                    f"Will align to scaler dimension first, then to model dimension."
                )
                # Align to scaler dimension
                if len(vector) < scaler_dim:
                    vector = np.pad(vector, (0, scaler_dim - len(vector)), mode='constant')
                else:
                    vector = vector[:scaler_dim]
            
            try:
                vector = self.scaler.transform(vector.reshape(1, -1))[0]
                logger.debug(f"Applied preprocessing scaler (output dim: {len(vector)})")
                
                # After scaling, ensure dimension matches model
                if len(vector) != self.expected_dim:
                    logger.warning(
                        f"After scaling, dimension is {len(vector)}, but model expects {self.expected_dim}. "
                        f"Will align to model dimension."
                    )
                    if len(vector) < self.expected_dim:
                        vector = np.pad(vector, (0, self.expected_dim - len(vector)), mode='constant')
                    else:
                        vector = vector[:self.expected_dim]
            except Exception as e:
                logger.warning(f"Failed to apply scaler: {e}. Using unscaled features.")
                # Ensure dimension is correct even without scaler
                if len(vector) != self.expected_dim:
                    if len(vector) < self.expected_dim:
                        vector = np.pad(vector, (0, self.expected_dim - len(vector)), mode='constant')
                    else:
                        vector = vector[:self.expected_dim]
        
        # Final verification - CRITICAL: Must match model's expected dimension
        final_dim = len(vector)
        if final_dim != self.expected_dim:
            logger.error(
                f"CRITICAL: Final dimension mismatch! Got {final_dim}, expected {self.expected_dim}. "
                f"Forcing alignment..."
            )
            if final_dim < self.expected_dim:
                vector = np.pad(vector, (0, self.expected_dim - final_dim), mode='constant')
                logger.info(f"Force padded from {final_dim} to {self.expected_dim}")
            else:
                vector = vector[:self.expected_dim]
                logger.info(f"Force truncated from {final_dim} to {self.expected_dim}")
        
        # Double-check
        if len(vector) != self.expected_dim:
            logger.error(
                f"CRITICAL ERROR: Vector dimension still wrong after alignment! "
                f"Got {len(vector)}, expected {self.expected_dim}. "
                f"This should not happen!"
            )
            # Last resort: create zero vector with correct dimension
            vector = np.zeros(self.expected_dim, dtype=np.float32)
            logger.error("Created zero vector as fallback - predictions may be inaccurate!")
        
        logger.info(f"✓ Final feature vector dimension: {len(vector)} (expected: {self.expected_dim})")
        
        feature_names = self.feature_pipeline.combiner.get_feature_names()
        # Truncate feature names if needed
        if len(feature_names) > self.expected_dim:
            feature_names = feature_names[:self.expected_dim]
        elif len(feature_names) < self.expected_dim:
            # Add padding names
            feature_names = feature_names + [f"_pad_{i}" for i in range(self.expected_dim - len(feature_names))]
        
        return {
            'vector': vector.reshape(1, -1),
            'feature_names': feature_names,
            'raw_dim': pipeline_dim,
            'aligned_dim': len(vector)
        }

    def _format_probabilities(self, probs: np.ndarray) -> Dict[str, float]:
        """
        Format probabilities dựa trên class order thực tế.
        CRITICAL: Đảm bảo mapping đúng giữa probability index và label.
        
        Args:
            probs: Probability array from model.predict_proba()
                  Shape: (n_samples, n_classes) or (n_classes,)
        
        Returns:
            Dictionary với keys: 'benign', 'obfuscated'
        """
        if probs.ndim == 2:
            probs = probs[0]
        
        # Handle single probability (binary classification with 1 output)
        if probs.size == 1:
            prob_class1 = float(probs[0])
            prob_class0 = 1.0 - prob_class1
            # probs = np.array([prob_class0, prob_class1])
        else:
            prob_class0 = float(probs[0])
            prob_class1 = float(probs[1])
        
        # CRITICAL: Map probability indices to actual class labels
        # class_order maps: prob_index -> label_name
        # Example: {0: 'benign', 1: 'obfuscated'} means prob[0] = P(benign), prob[1] = P(obfuscated)
        prob_map = {}
        
        prob_map[self.class_order.get(0, 'benign')] = prob_class0
        prob_map[self.class_order.get(1, 'obfuscated')] = prob_class1

        
        # Log for debugging
        logger.debug(
            f"Probability mapping: class0={prob_class0:.4f}->'{self.class_order.get(0)}', "
            f"class1={prob_class1:.4f}->'{self.class_order.get(1)}'"

        )
        
        return prob_map

    def _explain(self, feature_vector: np.ndarray, feature_names: list) -> Optional[list]:
        """Generate SHAP explanations"""
        if not self.enable_explainability:
            return None
        
        explainer = self._init_explainer()
        if explainer is None:
            return None
        
        try:
            shap_values = explainer.shap_values(feature_vector)
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
        """
        Predict on a single file.
        CRITICAL: Deterministic prediction, consistency checks.
        
        Args:
            file_path: Path to binary file
            
        Returns:
            Dictionary with prediction results
        """
        # Build feature vector
        features = self._build_feature_vector(file_path)
        vector = features['vector']
        
        # Final verification before prediction - CRITICAL CHECK
        vector_dim = vector.shape[1]
        if vector_dim != self.expected_dim:
            logger.error(
                f"CRITICAL: Vector dimension mismatch before prediction! "
                f"Got {vector_dim}, expected {self.expected_dim}. "
                f"Attempting to fix..."
            )
            if vector_dim < self.expected_dim:
                padding = np.zeros((1, self.expected_dim - vector_dim), dtype=vector.dtype)
                vector = np.hstack([vector, padding])
                logger.warning(f"Fixed vector dimension from {vector_dim} to {vector.shape[1]} (padded)")
            else:
                vector = vector[:, :self.expected_dim]
                logger.warning(f"Fixed vector dimension from {vector_dim} to {vector.shape[1]} (truncated)")
        
        # Verify one more time
        if vector.shape[1] != self.expected_dim:
            logger.error(
                f"CRITICAL ERROR: Cannot fix dimension mismatch! "
                f"Vector shape: {vector.shape}, expected: (1, {self.expected_dim})"
            )
            raise ValueError(
                f"Feature dimension mismatch: got {vector.shape[1]}, expected {self.expected_dim}. "
                f"Please retrain models with new dataset."
            )

        # Get raw prediction and probabilities
        try:
            logger.debug(f"Predicting with vector shape: {vector.shape}, expected dim: {self.expected_dim}")
            raw_prediction = self.model.predict(vector)[0]
            probabilities = self.model.predict_proba(vector)
        except ValueError as e:
            # If still fails, log detailed info
            logger.error(
                f"Prediction failed! Vector shape: {vector.shape}, "
                f"Model expects: {self.expected_dim}, Error: {e}"
            )
            # Try to get actual model dimension
            actual_model_dim = self._get_model_expected_dim()
            logger.error(
                f"Model actual expected dimension: {actual_model_dim}, "
                f"Pipeline expected dimension: {self.expected_dim}, "
                f"Vector dimension: {vector.shape[1]}"
            )
            raise ValueError(
                f"Feature shape mismatch: expected {actual_model_dim}, got {vector.shape[1]}. "
                f"Please retrain models with: python src/dataset/generate_dataset.py && python src/models/train.py"
            ) from e
        
        # Format probabilities with correct class mapping
        prob_map = self._format_probabilities(probabilities)
        
        # CRITICAL: Always use probabilities to determine prediction
        # This is more reliable than raw label
        prob_benign = prob_map.get('benign', 0.0)
        prob_obfuscated = prob_map.get('obfuscated', 0.0)
        
        # Determine prediction based on highest probability
        if prob_obfuscated > prob_benign:
            final_prediction = 'Obfuscated'
            final_label = 1
            confidence = prob_obfuscated
        else:
            final_prediction = 'Benign'
            final_label = 0
            confidence = prob_benign
        
        
        # Log if there's a mismatch (for debugging)
        label_prediction = 'Obfuscated' if raw_prediction == 1 else 'Benign'
        if label_prediction != final_prediction:
            logger.warning(
                f"⚠️ MISMATCH DETECTED! Raw label={raw_prediction} ({label_prediction}), "
                f"but probabilities say: {final_prediction} "
                f"(benign={prob_benign:.4f}, obfuscated={prob_obfuscated:.4f}). "
                f"Using probability-based decision: {final_prediction}"
            )
        else:
            logger.debug(
                f"✓ Prediction consistent: {final_prediction} "
                f"(benign={prob_benign:.4f}, obfuscated={prob_obfuscated:.4f})"
            )

        result = {
            'prediction': final_prediction,
            'label': int(final_label),
            'confidence': float(confidence),
            'probabilities': {
                'benign': float(prob_benign),
                'obfuscated': float(prob_obfuscated)
            },
            'feature_count': features['raw_dim'],
            'model_type': self.model_type,
            # Debug info
            'raw_label': int(raw_prediction),
            'class_order': self.class_order,
        }

        # Add explanations if enabled
        if self.enable_explainability:
            explanations = self._explain(vector, features['feature_names'])
            if explanations:
                result['top_contributors'] = explanations

        return result


class EnsembleInferencePipeline:
    """
    Ensemble inference pipeline với consistency checks.
    REFACTORED: Kiểm tra mâu thuẫn giữa các models.
    """
    
    def __init__(self, model_configs: List[Dict], feature_metadata: str, scaler_path: Optional[str] = None):
        """
        Initialize ensemble pipeline.
        
        Args:
            model_configs: List of model configs, each with 'model_path', 'model_type', 'scaler_path'
            feature_metadata: Path to feature metadata file
            scaler_path: Global scaler path (if all models use same scaler)
        """
        self.pipelines = []
        for config in model_configs:
            pipeline = InferencePipeline(
                model_path=config['model_path'],
                model_type=config['model_type'],
                feature_metadata=feature_metadata,
                scaler_path=config.get('scaler_path', scaler_path)
            )
            self.pipelines.append(pipeline)
        
        logger.info(f"Initialized ensemble with {len(self.pipelines)} models")
    
    def predict_file(self, file_path: str, conflict_threshold: float = 0.3) -> Dict:
        """
        Predict using ensemble với consistency checks.
        
        Args:
            file_path: Path to binary file
            conflict_threshold: Minimum probability difference to consider conflict
            
        Returns:
            Dictionary with ensemble prediction and consistency info
        """
        predictions = []
        for pipeline in self.pipelines:
            pred = pipeline.predict_file(file_path)
            predictions.append(pred)
        
        # Aggregate predictions
        benign_votes = sum(1 for p in predictions if p['prediction'] == 'Benign')
        obfuscated_votes = sum(1 for p in predictions if p['prediction'] == 'Obfuscated')
        
        # Average probabilities
        avg_benign = np.mean([p['probabilities']['benign'] for p in predictions])
        avg_obfuscated = np.mean([p['probabilities']['obfuscated'] for p in predictions])
        
        # Determine final prediction
        if avg_obfuscated > avg_benign:
            final_prediction = 'Obfuscated'
            final_label = 1
            final_confidence = avg_obfuscated
        else:
            final_prediction = 'Benign'
            final_label = 0
            final_confidence = avg_benign
        
        # Check for conflicts
        conflicts = []
        for i, p1 in enumerate(predictions):
            for j, p2 in enumerate(predictions[i+1:], start=i+1):
                if p1['prediction'] != p2['prediction']:
                    prob_diff = abs(p1['probabilities']['obfuscated'] - p2['probabilities']['obfuscated'])
                    if prob_diff > conflict_threshold:
                        conflicts.append({
                            'model1': p1['model_type'],
                            'model2': p2['model_type'],
                            'pred1': p1['prediction'],
                            'pred2': p2['prediction'],
                            'prob_diff': prob_diff
                        })
        
        result = {
            'prediction': final_prediction,
            'label': int(final_label),
            'confidence': float(final_confidence),
            'probabilities': {
                'benign': float(avg_benign),
                'obfuscated': float(avg_obfuscated)
            },
            'votes': {
                'benign': benign_votes,
                'obfuscated': obfuscated_votes,
                'total': len(predictions)
            },
            'individual_predictions': predictions,
            'conflicts': conflicts,
            'has_conflict': len(conflicts) > 0
        }
        
        if conflicts:
            logger.warning(
                f"⚠️ CONFLICT DETECTED between models! "
                f"{len(conflicts)} conflicting pairs found."
            )
            for conflict in conflicts:
                logger.warning(
                    f"  {conflict['model1']} ({conflict['pred1']}) vs "
                    f"{conflict['model2']} ({conflict['pred2']}), "
                    f"prob_diff={conflict['prob_diff']:.4f}"
                )
        else:
            logger.info(f"✓ All models agree: {final_prediction} (confidence={final_confidence:.4f})")
        
        return result
