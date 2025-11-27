"""
Feature Pipeline
Tái sử dụng logic trích xuất feature giữa bước huấn luyện và suy luận.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .feature_combiner import FeatureCombiner
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
    def from_metadata(cls, metadata_path: str) -> Tuple["FeaturePipelineConfig", int]:
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
        return config, feature_dim


class FeaturePipeline:
    """
    Pipeline thống nhất để trích xuất và chuẩn hoá features.
    Được dùng bởi cả dataset generator và inference pipeline.
    """

    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.expected_dim: Optional[int] = None
        self.opcode_extractor = OpcodeExtractor(n_grams=config.opcode_ngrams or [2, 3, 4])
        self.cfg_extractor = CFGExtractor() if config.enable_cfg else None
        self.api_extractor = APIExtractor(api_list_path=config.api_list_path)
        self.combiner = FeatureCombiner()

    @classmethod
    def from_metadata(cls, metadata_path: str) -> "FeaturePipeline":
        config, feature_dim = FeaturePipelineConfig.from_metadata(metadata_path)
        pipeline = cls(config)
        pipeline.expected_dim = feature_dim
        return pipeline

    def extract_feature_dict(self, file_path: str) -> Dict[str, np.ndarray]:
        """Trả về dict các feature thành phần trước khi combine."""
        aggregated: Dict[str, np.ndarray] = {}

        # Opcode features
        try:
            opcode_features = self.opcode_extractor.extract_from_file(
                file_path, max_features=self.config.opcode_max_features
            )
            aggregated.update(opcode_features)
        except Exception as exc:
            logger.warning("Opcode extraction failed for %s: %s", file_path, exc)

        # CFG features (optional)
        if self.cfg_extractor is not None:
            try:
                cfg_features = self.cfg_extractor.extract_features(file_path)
                aggregated.update(cfg_features)
            except Exception as exc:
                logger.warning("CFG extraction failed for %s: %s", file_path, exc)

        # API call features
        try:
            api_features = self.api_extractor.extract_api_features(
                file_path, max_features=self.config.api_max_features
            )
            aggregated.update(api_features)
        except Exception as exc:
            logger.warning("API extraction failed for %s: %s", file_path, exc)

        return aggregated

    def build_feature_vector(self, file_path: str) -> np.ndarray:
        """Combine tất cả feature thành vector duy nhất."""
        feature_dict = self.extract_feature_dict(file_path)
        combined = self.combiner.combine(feature_dict)
        return combined

    def pad_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Chuẩn hoá vector về cùng kích thước bằng cách padding hoặc truncate.
        """
        if target_dim <= 0:
            return vector

        current = len(vector)
        if current == target_dim:
            return vector

        if current < target_dim:
            return np.pad(vector, (0, target_dim - current), mode="constant")

        return vector[:target_dim]

    def save_metadata(self, output_path: Path, feature_dim: int) -> None:
        """
        Lưu metadata dùng lại cho suy luận.
        """
        metadata = {
            "feature_dim": int(feature_dim),
            "opcode_max_features": self.config.opcode_max_features,
            "opcode_ngrams": self.config.opcode_ngrams or [2, 3, 4],
            "api_max_features": self.config.api_max_features,
            "api_list_path": str(self.config.api_list_path) if self.config.api_list_path else None,
            "enable_cfg": self.config.enable_cfg,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Feature metadata saved to %s", output_path)



