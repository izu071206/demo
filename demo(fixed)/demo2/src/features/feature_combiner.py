"""
Feature Combiner - REFACTORED VERSION
Kết hợp các features với fixed schema và deterministic ordering
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSchema:
    """
    Fixed feature schema định nghĩa offsets và dimensions cho từng nhóm feature.
    Đảm bảo tính nhất quán giữa training và inference.
    """
    # Opcode n-gram features
    opcode_2gram_dim: int = 1000
    opcode_3gram_dim: int = 1000
    opcode_4gram_dim: int = 1000
    
    # API features
    api_calls_dim: int = 500
    
    # CFG features (scalar metrics)
    cfg_metrics: List[str] = None
    
    # API metadata (scalar)
    api_metadata: List[str] = None
    
    # Dynamic loader features (scalar)
    dynamic_loader_features: List[str] = None
    
    def __post_init__(self):
        """Initialize default feature lists"""
        if self.cfg_metrics is None:
            self.cfg_metrics = [
                'num_nodes', 'num_edges', 'avg_degree', 'cyclomatic_complexity',
                'num_loops', 'max_depth', 'avg_path_length', 'clustering_coefficient'
            ]
        if self.api_metadata is None:
            self.api_metadata = ['num_imports', 'num_dlls', 'api_entropy']
        if self.dynamic_loader_features is None:
            self.dynamic_loader_features = [
                'dynamic_loader_score', 'dynamic_loader_unique', 'uses_dynamic_loading',
                'call_loadlibrarya', 'call_loadlibraryw', 'call_loadlibraryexa',
                'call_loadlibraryexw', 'call_getprocaddress', 'call_getmodulehandlea',
                'call_getmodulehandlew', 'call_ldrgetprocedureaddress'
            ]
    
    def get_total_dim(self) -> int:
        """Tính tổng số dimensions của feature vector"""
        total = (
            self.opcode_2gram_dim +
            self.opcode_3gram_dim +
            self.opcode_4gram_dim +
            self.api_calls_dim +
            len(self.cfg_metrics) +
            len(self.api_metadata) +
            len(self.dynamic_loader_features)
        )
        return total
    
    def get_offsets(self) -> Dict[str, Tuple[int, int]]:
        """
        Trả về offsets (start, end) cho từng nhóm feature.
        CRITICAL: Thứ tự này phải cố định và nhất quán.
        """
        offsets = {}
        current = 0
        
        # Opcode features (theo thứ tự: 2-gram, 3-gram, 4-gram)
        offsets['opcode_2gram'] = (current, current + self.opcode_2gram_dim)
        current += self.opcode_2gram_dim
        
        offsets['opcode_3gram'] = (current, current + self.opcode_3gram_dim)
        current += self.opcode_3gram_dim
        
        offsets['opcode_4gram'] = (current, current + self.opcode_4gram_dim)
        current += self.opcode_4gram_dim
        
        # API calls
        offsets['api_calls'] = (current, current + self.api_calls_dim)
        current += self.api_calls_dim
        
        # CFG metrics
        offsets['cfg_metrics'] = (current, current + len(self.cfg_metrics))
        current += len(self.cfg_metrics)
        
        # API metadata
        offsets['api_metadata'] = (current, current + len(self.api_metadata))
        current += len(self.api_metadata)
        
        # Dynamic loader features
        offsets['dynamic_loader'] = (current, current + len(self.dynamic_loader_features))
        current += len(self.dynamic_loader_features)
        
        return offsets
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureSchema':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_metadata(cls, metadata_path: str) -> 'FeatureSchema':
        """Load schema from metadata file"""
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema metadata not found: {metadata_path}")
        
        with path.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract schema from metadata
        schema_data = metadata.get('schema', {})
        if not schema_data:
            # Fallback: try to infer from feature_dim
            feature_dim = metadata.get('feature_dim', 0)
            logger.warning(f"No schema in metadata, inferring from feature_dim={feature_dim}")
            return cls._infer_from_dim(feature_dim)
        
        return cls.from_dict(schema_data)
    
    @classmethod
    def _infer_from_dim(cls, total_dim: int) -> 'FeatureSchema':
        """Infer schema from total dimension (fallback)"""
        # Default dimensions
        default = cls()
        return default
    
    def save(self, output_path: Path):
        """Save schema to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Feature schema saved to {output_path}")


class FeatureCombiner:
    """
    Kết hợp các features với fixed schema và deterministic ordering.
    REFACTORED: Đảm bảo cùng file → cùng vector, không phụ thuộc thứ tự.
    """
    
    def __init__(self, schema: Optional[FeatureSchema] = None):
        """
        Initialize feature combiner với fixed schema.
        
        Args:
            schema: Feature schema định nghĩa offsets và dimensions.
                    Nếu None, sẽ tạo schema mặc định.
        """
        if schema is None:
            schema = FeatureSchema()
        self.schema = schema
        self.offsets = schema.get_offsets()
        self.total_dim = schema.get_total_dim()
        
        # Generate feature names for debugging/explainability
        self.feature_names = self._generate_feature_names()
        
        logger.info(f"FeatureCombiner initialized with total_dim={self.total_dim}")
        logger.debug(f"Feature offsets: {self.offsets}")
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names theo schema"""
        names = []
        
        # Opcode features
        for n in [2, 3, 4]:
            for i in range(getattr(self.schema, f'opcode_{n}gram_dim')):
                names.append(f'opcode_{n}gram_{i}')
        
        # API calls
        for i in range(self.schema.api_calls_dim):
            names.append(f'api_call_{i}')
        
        # CFG metrics
        names.extend([f'cfg_{m}' for m in self.schema.cfg_metrics])
        
        # API metadata
        names.extend([f'api_{m}' for m in self.schema.api_metadata])
        
        # Dynamic loader
        names.extend([f'dyn_{f}' for f in self.schema.dynamic_loader_features])
        
        return names
    
    def combine(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Kết hợp features thành vector với fixed schema.
        
        CRITICAL: Features được đặt vào đúng vị trí theo schema,
        không phụ thuộc vào thứ tự trong dict.
        
        Args:
            features: Dictionary of feature arrays với keys:
                     - 'opcode_2gram', 'opcode_3gram', 'opcode_4gram': opcode n-gram vectors
                     - 'api_calls': API call frequency vector
                     - CFG metrics: scalar values với keys từ cfg_metrics list
                     - API metadata: scalar values với keys từ api_metadata list
                     - Dynamic loader: scalar values với keys từ dynamic_loader_features list
        
        Returns:
            Combined feature vector với fixed dimension
        """
        # Initialize output vector với zeros
        result = np.zeros(self.total_dim, dtype=np.float32)
        
        # Place opcode features
        for n in [2, 3, 4]:
            key = f'opcode_{n}gram'
            if key in features:
                start, end = self.offsets[key]
                vector = features[key]
                # Pad or truncate to fit
                if len(vector) > (end - start):
                    vector = vector[:end - start]
                elif len(vector) < (end - start):
                    # Pad with zeros
                    padded = np.zeros(end - start, dtype=np.float32)
                    padded[:len(vector)] = vector
                    vector = padded
                result[start:start + len(vector)] = vector
            else:
                logger.debug(f"Missing {key} features, leaving zeros")
        
        # Place API calls
        if 'api_calls' in features:
            start, end = self.offsets['api_calls']
            vector = features['api_calls']
            if len(vector) > (end - start):
                vector = vector[:end - start]
            elif len(vector) < (end - start):
                padded = np.zeros(end - start, dtype=np.float32)
                padded[:len(vector)] = vector
                vector = padded
            result[start:start + len(vector)] = vector
        else:
            logger.debug("Missing api_calls features, leaving zeros")
        
        # Place CFG metrics
        start, end = self.offsets['cfg_metrics']
        for i, metric_name in enumerate(self.schema.cfg_metrics):
            if metric_name in features:
                value = features[metric_name]
                # Convert to scalar if needed
                if isinstance(value, np.ndarray):
                    value = float(value.item() if value.size == 1 else value[0])
                result[start + i] = float(value)
            else:
                logger.debug(f"Missing CFG metric {metric_name}, leaving zero")
        
        # Place API metadata
        start, end = self.offsets['api_metadata']
        for i, meta_name in enumerate(self.schema.api_metadata):
            if meta_name in features:
                value = features[meta_name]
                if isinstance(value, np.ndarray):
                    value = float(value.item() if value.size == 1 else value[0])
                result[start + i] = float(value)
            else:
                logger.debug(f"Missing API metadata {meta_name}, leaving zero")
        
        # Place dynamic loader features
        start, end = self.offsets['dynamic_loader']
        for i, feat_name in enumerate(self.schema.dynamic_loader_features):
            # Map feature name (e.g., 'call_loadlibrarya' -> 'call_LoadLibraryA')
            # Try both lowercase and original case
            keys_to_try = [
                feat_name,
                feat_name.replace('call_', 'call_').title(),
                f'call_{feat_name.replace("call_", "").lower()}'
            ]
            
            found = False
            for key in keys_to_try:
                if key in features:
                    value = features[key]
                    if isinstance(value, np.ndarray):
                        value = float(value.item() if value.size == 1 else value[0])
                    result[start + i] = float(value)
                    found = True
                    break
            
            if not found:
                logger.debug(f"Missing dynamic loader feature {feat_name}, leaving zero")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Trả về danh sách tên features theo thứ tự trong vector"""
        return self.feature_names
    
    def get_schema(self) -> FeatureSchema:
        """Trả về feature schema"""
        return self.schema
    
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
