"""
Feature Pipeline - REFACTORED VERSION
Deterministic feature extraction với fixed schema
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .feature_combiner import FeatureCombiner, FeatureSchema
from .static import APIExtractor, CFGExtractor, OpcodeExtractor

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    """Lightweight container cho cấu hình trích xuất feature."""

    opcode_max_features: int = 1000
    opcode_ngrams: Optional[list] = None
    api_max_features: int = 500
    api_list_path: Optional[str] = None
    enable_cfg: bool = True

    def __post_init__(self):
        """Set default n-grams if not provided"""
        if self.opcode_ngrams is None:
            self.opcode_ngrams = [2, 3, 4]

    @classmethod
    def from_dataset_config(cls, config_path: str) -> "FeaturePipelineConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")

        with path.open("r", encoding="utf-8") as fh:
            import yaml
            raw_cfg = yaml.safe_load(fh)

        dataset_cfg = raw_cfg.get("dataset", raw_cfg)
        feature_cfg = dataset_cfg.get("features", {})
        opcode_cfg = feature_cfg.get("opcode_ngrams", {})
        api_cfg = feature_cfg.get("api_calls", {})
        cfg_cfg = feature_cfg.get("cfg", {})

        return cls(
            opcode_max_features=opcode_cfg.get("max_features", 1000),
            opcode_ngrams=opcode_cfg.get("n", [2, 3, 4]),
            api_max_features=api_cfg.get("max_features", 500),
            api_list_path=api_cfg.get("api_list_path"),
            enable_cfg=cfg_cfg.get("extract_metrics", True),
        )

    @classmethod
    def from_metadata(cls, metadata_path: str) -> Tuple["FeaturePipelineConfig", int, FeatureSchema]:
        """
        Load config và schema từ metadata file.
        
        Returns:
            Tuple of (config, feature_dim, schema)
        """
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Feature metadata not found: {metadata_path}")

        with path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        config = cls(
            opcode_max_features=metadata.get("opcode_max_features", 1000),
            opcode_ngrams=metadata.get("opcode_ngrams", [2, 3, 4]),
            api_max_features=metadata.get("api_max_features", 500),
            api_list_path=metadata.get("api_list_path"),
            enable_cfg=metadata.get("enable_cfg", True),
        )
        
        feature_dim = int(metadata.get("feature_dim", 0))
        
        # Load schema
        schema = FeatureSchema.from_metadata(metadata_path)
        
        return config, feature_dim, schema


class FeaturePipeline:
    """
    Pipeline thống nhất để trích xuất và chuẩn hoá features.
    REFACTORED: Deterministic, fixed schema, không phụ thuộc thứ tự.
    """

    def __init__(self, config: FeaturePipelineConfig, schema: Optional[FeatureSchema] = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Feature extraction configuration
            schema: Fixed feature schema. Nếu None, sẽ tạo schema mới từ config.
        """
        self.config = config
        
        # Create schema if not provided
        if schema is None:
            schema = FeatureSchema(
                opcode_2gram_dim=config.opcode_max_features,
                opcode_3gram_dim=config.opcode_max_features,
                opcode_4gram_dim=config.opcode_max_features,
                api_calls_dim=config.api_max_features,
            )
        
        self.schema = schema
        self.combiner = FeatureCombiner(schema)
        self.expected_dim = schema.get_total_dim()
        
        # Initialize extractors
        self.opcode_extractor = OpcodeExtractor(n_grams=config.opcode_ngrams)
        self.cfg_extractor = CFGExtractor() if config.enable_cfg else None
        self.api_extractor = APIExtractor(api_list_path=config.api_list_path)
        
        logger.info(f"FeaturePipeline initialized with expected_dim={self.expected_dim}")

    @classmethod
    def from_metadata(cls, metadata_path: str) -> "FeaturePipeline":
        """
        Load pipeline từ metadata (dùng cho inference).
        CRITICAL: Đảm bảo schema giống hệt training.
        """
        config, feature_dim, schema = FeaturePipelineConfig.from_metadata(metadata_path)
        pipeline = cls(config, schema)
        pipeline.expected_dim = feature_dim
        
        # Verify dimension matches
        if pipeline.expected_dim != schema.get_total_dim():
            logger.warning(
                f"Dimension mismatch: metadata says {pipeline.expected_dim}, "
                f"schema says {schema.get_total_dim()}. Using schema dimension."
            )
            pipeline.expected_dim = schema.get_total_dim()
        
        logger.info(f"Pipeline loaded from metadata: expected_dim={pipeline.expected_dim}")
        return pipeline

    def extract_feature_dict(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Trích xuất features thành dictionary.
        CRITICAL: Thứ tự extraction không ảnh hưởng đến kết quả cuối cùng
        vì combiner sử dụng fixed schema.
        
        Args:
            file_path: Path to binary file
            
        Returns:
            Dictionary of features với keys theo schema
        """
        aggregated: Dict[str, np.ndarray] = {}
        
        # Extract opcode features (theo thứ tự: 2, 3, 4-gram)
        try:
            opcode_features = self.opcode_extractor.extract_from_file(
                file_path, max_features=self.config.opcode_max_features
            )
            
            # Map to schema keys
            for n in [2, 3, 4]:
                key = f'opcode_{n}gram'
                if key in opcode_features:
                    aggregated[key] = opcode_features[key]
                else:
                    # Create empty vector if missing
                    dim = getattr(self.schema, f'opcode_{n}gram_dim')
                    aggregated[key] = np.zeros(dim, dtype=np.float32)
                    logger.debug(f"No {n}-gram features found, using zeros")
            
        except Exception as exc:
            logger.warning(f"Opcode extraction failed for {file_path}: {exc}")
            # Fill with zeros
            for n in [2, 3, 4]:
                dim = getattr(self.schema, f'opcode_{n}gram_dim')
                aggregated[f'opcode_{n}gram'] = np.zeros(dim, dtype=np.float32)

        # Extract CFG features (optional) - với timeout để tránh chậm
        if self.cfg_extractor is not None:
            try:
                cfg_features = self.cfg_extractor.extract_features(file_path)
                # Map CFG metrics to schema
                for metric_name in self.schema.cfg_metrics:
                    if metric_name in cfg_features:
                        aggregated[metric_name] = np.array([float(cfg_features[metric_name])])
                    else:
                        aggregated[metric_name] = np.array([0.0])
                        logger.debug(f"Missing CFG metric {metric_name}, using zero")
            except Exception as exc:
                logger.warning(f"CFG extraction failed for {file_path}: {exc}")
                # Fill with zeros
                for metric_name in self.schema.cfg_metrics:
                    aggregated[metric_name] = np.array([0.0])
        else:
            # Fill with zeros if CFG disabled
            for metric_name in self.schema.cfg_metrics:
                aggregated[metric_name] = np.array([0.0])

        # Extract API call features
        try:
            api_features = self.api_extractor.extract_api_features(
                file_path, max_features=self.config.api_max_features
            )
            
            # Map API calls
            if 'api_calls' in api_features:
                aggregated['api_calls'] = api_features['api_calls']
            else:
                aggregated['api_calls'] = np.zeros(self.schema.api_calls_dim, dtype=np.float32)
            
            # Map API metadata
            for meta_name in self.schema.api_metadata:
                if meta_name in api_features:
                    value = api_features[meta_name]
                    if isinstance(value, (int, float)):
                        aggregated[meta_name] = np.array([float(value)])
                    elif isinstance(value, np.ndarray):
                        aggregated[meta_name] = value
                    else:
                        aggregated[meta_name] = np.array([0.0])
                else:
                    aggregated[meta_name] = np.array([0.0])
            
            # Map dynamic loader features
            for feat_name in self.schema.dynamic_loader_features:
                # Try to find matching key in api_features
                # Handle both 'call_LoadLibraryA' and 'call_loadlibrarya' formats
                found = False
                for key in api_features.keys():
                    if key.lower() == feat_name.lower() or \
                       key.lower().replace('_', '') == feat_name.lower().replace('_', ''):
                        value = api_features[key]
                        if isinstance(value, (int, float)):
                            aggregated[feat_name] = np.array([float(value)])
                        elif isinstance(value, np.ndarray):
                            aggregated[feat_name] = value
                        else:
                            aggregated[feat_name] = np.array([0.0])
                        found = True
                        break
                
                if not found:
                    aggregated[feat_name] = np.array([0.0])
            
        except Exception as exc:
            logger.warning(f"API extraction failed for {file_path}: {exc}")
            # Fill with zeros
            aggregated['api_calls'] = np.zeros(self.schema.api_calls_dim, dtype=np.float32)
            for meta_name in self.schema.api_metadata:
                aggregated[meta_name] = np.array([0.0])
            for feat_name in self.schema.dynamic_loader_features:
                aggregated[feat_name] = np.array([0.0])
        
        return aggregated

    def build_feature_vector(self, file_path: str) -> np.ndarray:
        """
        Combine tất cả features thành vector duy nhất với fixed dimension.
        CRITICAL: Cùng file → cùng vector (deterministic).
        
        Args:
            file_path: Path to binary file
            
        Returns:
            Feature vector với fixed dimension theo schema
        """
        feature_dict = self.extract_feature_dict(file_path)
        combined = self.combiner.combine(feature_dict)
        
        # Verify dimension
        if len(combined) != self.expected_dim:
            logger.warning(
                f"Feature dimension mismatch: got {len(combined)}, expected {self.expected_dim}. "
                f"Padding/truncating to expected dimension."
            )
            combined = self.pad_vector(combined, self.expected_dim)
        
        return combined

    def pad_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Chuẩn hoá vector về cùng kích thước bằng cách padding hoặc truncate.
        
        Args:
            vector: Input vector
            target_dim: Target dimension
            
        Returns:
            Vector với dimension = target_dim
        """
        if target_dim <= 0:
            return vector

        current = len(vector)
        if current == target_dim:
            return vector

        if current < target_dim:
            logger.debug(f"Padding vector from {current} to {target_dim}")
            return np.pad(vector, (0, target_dim - current), mode="constant")

        logger.debug(f"Truncating vector from {current} to {target_dim}")
        return vector[:target_dim]

    def save_metadata(self, output_path: Path, feature_dim: int) -> None:
        """
        Lưu metadata dùng lại cho suy luận.
        CRITICAL: Phải lưu cả config và schema.
        """
        metadata = {
            "feature_dim": int(feature_dim),
            "opcode_max_features": self.config.opcode_max_features,
            "opcode_ngrams": self.config.opcode_ngrams,
            "api_max_features": self.config.api_max_features,
            "api_list_path": str(self.config.api_list_path) if self.config.api_list_path else None,
            "enable_cfg": self.config.enable_cfg,
            "schema": self.schema.to_dict(),  # CRITICAL: Save schema
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        
        logger.info(f"Feature metadata saved to {output_path} (dim={feature_dim}, schema included)")
