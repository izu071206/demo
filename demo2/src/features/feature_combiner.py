"""
Feature Combiner
Kết hợp các features từ nhiều nguồn khác nhau
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureCombiner:
    """Kết hợp các features từ static và dynamic analysis"""
    
    def __init__(self):
        """Initialize feature combiner"""
        self.feature_names = []
        self.feature_indices = {}
    
    def combine(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Kết hợp các features thành một vector
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Combined feature vector
        """
        combined = []
        feature_names = []
        
        for feature_name, feature_array in features.items():
            if isinstance(feature_array, np.ndarray):
                if feature_array.ndim == 1:
                    # Flatten và thêm vào
                    combined.append(feature_array)
                    # Tạo tên cho từng feature
                    if len(feature_array) == 1:
                        feature_names.append(feature_name)
                    else:
                        for i in range(len(feature_array)):
                            feature_names.append(f"{feature_name}_{i}")
                elif feature_array.ndim == 0:
                    # Scalar value
                    combined.append(np.array([feature_array]))
                    feature_names.append(feature_name)
            elif isinstance(feature_array, (int, float)):
                combined.append(np.array([float(feature_array)]))
                feature_names.append(feature_name)
        
        if combined:
            result = np.concatenate(combined)
            self.feature_names = feature_names
            return result
        else:
            logger.warning("No features to combine")
            return np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Trả về danh sách tên features"""
        return self.feature_names
    
    def normalize_features(self, features: np.ndarray, method: str = 'l2') -> np.ndarray:
        """
        Normalize feature vector
        
        Args:
            features: Feature vector
            method: Normalization method ('l2', 'minmax', 'standard')
            
        Returns:
            Normalized feature vector
        """
        if len(features) == 0:
            return features
        
        if method == 'l2':
            norm = np.linalg.norm(features)
            if norm > 0:
                return features / norm
            return features
        
        elif method == 'minmax':
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val > min_val:
                return (features - min_val) / (max_val - min_val)
            return features
        
        elif method == 'standard':
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                return (features - mean) / std
            return features
        
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return features

