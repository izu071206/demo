"""
Dataset Generation - REFACTORED VERSION
Strict family-based splitting, fixed feature dimensions, deterministic processing
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pefile
import yaml
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from src.features.feature_combiner import FeatureSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Generate dataset từ binary files.
    REFACTORED: Strict family-based split, fixed feature dimensions, deterministic.
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to dataset config YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']
        
        features_cfg = self.config.get('features', {})
        opcode_cfg = features_cfg.get('opcode_ngrams', {})
        api_cfg = features_cfg.get('api_calls', {})
        cfg_cfg = features_cfg.get('cfg', {})
        
        pipeline_cfg = FeaturePipelineConfig(
            opcode_max_features=opcode_cfg.get('max_features', 1000),
            opcode_ngrams=opcode_cfg.get('n', [2, 3, 4]),
            api_max_features=api_cfg.get('max_features', 500),
            api_list_path=api_cfg.get('api_list_path'),
            enable_cfg=cfg_cfg.get('extract_metrics', True)
        )
        
        # Create fixed schema
        schema = FeatureSchema(
            opcode_2gram_dim=opcode_cfg.get('max_features', 1000),
            opcode_3gram_dim=opcode_cfg.get('max_features', 1000),
            opcode_4gram_dim=opcode_cfg.get('max_features', 1000),
            api_calls_dim=api_cfg.get('max_features', 500),
        )
        
        self.feature_pipeline = FeaturePipeline(pipeline_cfg, schema)
        self.metadata_path = Path(self.config['processed_features_dir']) / 'feature_metadata.json'
        self.expected_dim = schema.get_total_dim()
        
        logger.info(f"DatasetGenerator initialized with expected_dim={self.expected_dim}")
    
    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """
        Trích xuất features từ một file.
        CRITICAL: Luôn trả về vector với fixed dimension.
        
        Args:
            file_path: Path to binary file
            
        Returns:
            Feature vector với fixed dimension
        """
        return self.feature_pipeline.build_feature_vector(file_path)
    
    def is_valid_binary_file(self, file_path: Path) -> bool:
        """
        Kiểm tra file có phải binary hợp lệ không
        
        Args:
            file_path: Path to file
            
        Returns:
            True nếu là binary hợp lệ
        """
        # Bỏ qua các file không phải binary
        skip_extensions = {'.gitkeep', '.txt', '.md', '.py', '.yaml', '.yml', 
                          '.json', '.csv', '.pkl', '.pt', '.log', '.png', '.jpg'}
        
        if file_path.suffix.lower() in skip_extensions:
            return False
        
        # Bỏ qua hidden files
        if file_path.name.startswith('.'):
            return False
        
        # Kiểm tra file size (ít nhất 100 bytes)
        try:
            if file_path.stat().st_size < 100:
                return False
        except:
            return False
        
        # Kiểm tra file có phải binary (có null bytes hoặc không phải text)
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(512)
                if len(chunk) == 0:
                    return False
                # Nếu có nhiều null bytes, có thể là binary
                null_count = chunk.count(b'\x00')
                if null_count > 10:  # Nhiều null bytes = binary
                    return True
                # Kiểm tra có phải text không
                try:
                    chunk.decode('utf-8')
                    # Nếu decode được và ít null bytes, có thể là text
                    if null_count == 0:
                        return False
                except:
                    # Không decode được = binary
                    return True
        except:
            return False
        
        return True
    
    def validate_pe(self, file_path: Path) -> bool:
        """Đảm bảo file là PE hợp lệ để tránh dữ liệu hỏng."""
        try:
            pe = pefile.PE(str(file_path), fast_load=True)
            is_valid = pe.DOS_HEADER.e_magic == 0x5A4D
            pe.close()
            return is_valid
        except Exception:
            return False
    
    def get_family_name(self, root_dir: Path, file_path: Path, label: int) -> str:
        """
        Sử dụng tên thư mục để suy ra malware family hoặc ứng dụng.
        CRITICAL: Family name phải nhất quán để đảm bảo group split đúng.
        """
        try:
            relative = file_path.relative_to(root_dir)
            parts = relative.parts
            if len(parts) > 1:
                # Use first directory as family name
                family = parts[0]
                # Normalize: lowercase, remove special chars
                family = family.lower().strip()
                return family
        except Exception:
            pass
        
        # Fallback: use parent directory name
        parent_name = file_path.parent.name.lower().strip()
        if parent_name and parent_name != root_dir.name:
            return parent_name
        
        # Final fallback
        return "benign" if label == 0 else "unknown_malware"
    
    def process_directory(self, directory: str, label: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Process tất cả files trong directory.
        CRITICAL: Tất cả features phải có cùng dimension ngay từ đầu.
        
        Args:
            directory: Directory path
            label: Label (0: benign, 1: obfuscated)
            
        Returns:
            Tuple of (features array, labels array, metadata list)
        """
        features_list: List[np.ndarray] = []
        labels_list: List[int] = []
        metadata_list: List[Dict] = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return np.array([]), np.array([]), []
        
        # Lấy tất cả files và filter
        all_files = list(Path(directory).rglob('*'))
        files = [f for f in all_files if f.is_file() and self.is_valid_binary_file(f)]
        
        if len(files) == 0:
            logger.warning(f"No valid binary files found in {directory}")
            return np.array([]), np.array([]), []
        
        logger.info(f"Processing {len(files)} valid binary files from {directory}")
        
        root_dir = Path(directory)
        for file_path in tqdm(files, desc=f"Processing {directory}"):
            try:
                if not self.validate_pe(file_path):
                    logger.debug("Skipping invalid PE file: %s", file_path.name)
                    continue
                
                # Extract features (always returns fixed dimension)
                features = self.extract_features_from_file(str(file_path))
                
                # Verify dimension
                if len(features) != self.expected_dim:
                    logger.warning(
                        f"Feature dimension mismatch for {file_path.name}: "
                        f"got {len(features)}, expected {self.expected_dim}. Skipping."
                    )
                    continue
                
                features_list.append(features)
                labels_list.append(label)
                
                # Get family name for group splitting
                family = self.get_family_name(root_dir, file_path, label)
                
                metadata_list.append({
                    'file_path': str(file_path),
                    'label': label,
                    'family': family,
                    'source_dir': directory,
                    'size_bytes': file_path.stat().st_size,
                })
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        if features_list:
            # CRITICAL: All features should already have the same dimension
            # Verify and convert to array
            features_array = np.array(features_list)
            
            # Double-check dimensions
            if features_array.ndim == 2:
                actual_dim = features_array.shape[1]
                if actual_dim != self.expected_dim:
                    logger.error(
                        f"CRITICAL: Feature dimension mismatch! "
                        f"Expected {self.expected_dim}, got {actual_dim}. "
                        f"This should not happen with fixed schema."
                    )
                    # Pad/truncate as emergency fix
                    if actual_dim < self.expected_dim:
                        padding = np.zeros((features_array.shape[0], self.expected_dim - actual_dim))
                        features_array = np.hstack([features_array, padding])
                    else:
                        features_array = features_array[:, :self.expected_dim]
            
            return features_array, np.array(labels_list), metadata_list
        else:
            return np.array([]), np.array([]), []
    
    def generate_dataset(self):
        """
        Generate complete dataset với strict family-based splitting.
        CRITICAL: 
        - Tất cả features có cùng dimension
        - Không có family nào xuất hiện ở cả train và test
        - Deterministic processing
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting dataset generation...")
        logger.info("=" * 60)
        
        # Process benign samples
        logger.info("Processing benign samples...")
        benign_start = time.time()
        benign_dir = self.config['benign_source_dir']
        benign_features, benign_labels, benign_metadata = self.process_directory(benign_dir, label=0)
        benign_time = time.time() - benign_start
        logger.info(f"✓ Benign samples processed in {benign_time:.2f} seconds ({benign_time/60:.2f} minutes)")
        
        # Process obfuscated samples
        logger.info("Processing obfuscated samples...")
        obfuscated_start = time.time()
        obfuscated_dir = self.config['obfuscated_output_dir']
        obfuscated_features, obfuscated_labels, obfuscated_metadata = self.process_directory(obfuscated_dir, label=1)
        obfuscated_time = time.time() - obfuscated_start
        logger.info(f"✓ Obfuscated samples processed in {obfuscated_time:.2f} seconds ({obfuscated_time/60:.2f} minutes)")
        
        # Combine
        if benign_features.size == 0 and obfuscated_features.size == 0:
            logger.error("No features extracted from any directory!")
            return
        
        # Combine features and metadata
        if benign_features.size > 0 and obfuscated_features.size > 0:
            all_features = np.vstack([benign_features, obfuscated_features])
            all_labels = np.hstack([benign_labels, obfuscated_labels])
            all_metadata = benign_metadata + obfuscated_metadata
        elif benign_features.size > 0:
            all_features = benign_features
            all_labels = benign_labels
            all_metadata = benign_metadata
        else:
            all_features = obfuscated_features
            all_labels = obfuscated_labels
            all_metadata = obfuscated_metadata
        
        # Verify all features have same dimension
        if all_features.ndim == 2:
            actual_dim = all_features.shape[1]
            if actual_dim != self.expected_dim:
                logger.error(
                    f"CRITICAL: Feature dimension mismatch after combining! "
                    f"Expected {self.expected_dim}, got {actual_dim}"
                )
                return
        else:
            logger.error(f"CRITICAL: Features array has wrong shape: {all_features.shape}")
            return
        
        # CRITICAL: Strict family-based splitting
        # Extract family groups
        groups = np.array([meta.get('family', 'unknown') for meta in all_metadata])
        unique_families = np.unique(groups)
        logger.info(f"Found {len(unique_families)} unique families for group splitting")
        
        # Log family distribution
        family_counts = {}
        for family in unique_families:
            count = np.sum(groups == family)
            family_counts[family] = count
            logger.debug(f"  Family '{family}': {count} samples")
        
        # Split ratios
        rng = self.config.get('random_state', 42)
        test_ratio = self.config.get('test_ratio', 0.15)
        val_ratio = self.config.get('val_ratio', 0.15)
        train_ratio = self.config.get('train_ratio', 0.7)
        
        # Verify ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            logger.warning(f"Split ratios sum to {total_ratio}, normalizing to 1.0")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        def strict_group_split(features, labels, groups, test_size, random_state):
            """
            Strict group split: đảm bảo không có family nào xuất hiện ở cả train và test.
            """
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            try:
                train_idx, test_idx = next(splitter.split(features, labels, groups=groups))
                
                # Verify no family leakage
                train_families = set(groups[train_idx])
                test_families = set(groups[test_idx])
                overlap = train_families & test_families
                
                if overlap:
                    logger.error(
                        f"CRITICAL: Family leakage detected! "
                        f"Families in both train and test: {overlap}"
                    )
                    # Try to fix by moving overlapping families to test only
                    for family in overlap:
                        family_indices = np.where(groups == family)[0]
                        # Move all to test
                        test_idx = np.union1d(test_idx, family_indices)
                        train_idx = np.setdiff1d(train_idx, family_indices)
                    
                    logger.warning(f"Fixed leakage by moving {len(overlap)} families to test only")
                
                return train_idx, test_idx
                
            except ValueError as exc:
                logger.error(f"Group split failed: {exc}")
                raise
        
        # First split: train+val vs test
        trainval_idx, test_idx = strict_group_split(
            all_features, all_labels, groups, test_ratio, rng
        )
        
        # Second split: train vs val
        trainval_features = all_features[trainval_idx]
        trainval_labels = all_labels[trainval_idx]
        trainval_groups = groups[trainval_idx]
        
        val_relative = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0.0
        
        train_idx_rel, val_idx_rel = strict_group_split(
            trainval_features, trainval_labels, trainval_groups, val_relative, rng + 1
        )
        
        # Map back to original indices
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]
        
        # Final splits
        X_train, y_train = all_features[train_idx], all_labels[train_idx]
        X_val, y_val = all_features[val_idx], all_labels[val_idx]
        X_test, y_test = all_features[test_idx], all_labels[test_idx]
        
        # Create split map for metadata
        split_map = ['train'] * len(all_features)
        for idx in val_idx:
            split_map[idx] = 'val'
        for idx in test_idx:
            split_map[idx] = 'test'
        
        # Verify no family leakage in final splits
        train_families = set(groups[train_idx])
        val_families = set(groups[val_idx])
        test_families = set(groups[test_idx])
        
        train_val_overlap = train_families & val_families
        train_test_overlap = train_families & test_families
        val_test_overlap = val_families & test_families
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.warning(
                f"Family overlap detected: "
                f"train-val: {train_val_overlap}, "
                f"train-test: {train_test_overlap}, "
                f"val-test: {val_test_overlap}"
            )
        else:
            logger.info("✓ No family leakage detected in final splits")
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df['split'] = split_map
        metadata_df['feature_dim'] = self.expected_dim
        
        # Save splits
        output_dir = self.config['processed_features_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        self.save_split(output_dir, 'train', X_train, y_train)
        self.save_split(output_dir, 'val', X_val, y_val)
        self.save_split(output_dir, 'test', X_test, y_test)
        
        # Save metadata
        metadata_path = Path(output_dir) / 'sample_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        
        # Save feature metadata (CRITICAL: includes schema)
        self.feature_pipeline.save_metadata(self.metadata_path, self.expected_dim)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Log summary
        logger.info("=" * 60)
        logger.info("Dataset generation completed:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        logger.info(f"  Feature dimension: {self.expected_dim}")
        logger.info(f"  Train families: {len(train_families)}")
        logger.info(f"  Val families: {len(val_families)}")
        logger.info(f"  Test families: {len(test_families)}")
        logger.info("")
        logger.info("⏱️  Timing Summary:")
        logger.info(f"  Benign processing: {benign_time:.2f}s ({benign_time/60:.2f} min)")
        logger.info(f"  Obfuscated processing: {obfuscated_time:.2f}s ({obfuscated_time/60:.2f} min)")
        logger.info(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        if len(benign_features) > 0:
            logger.info(f"  Avg time per benign file: {benign_time/len(benign_features):.2f}s")
        if len(obfuscated_features) > 0:
            logger.info(f"  Avg time per obfuscated file: {obfuscated_time/len(obfuscated_features):.2f}s")
        logger.info("=" * 60)
    
    def save_split(self, output_dir: str, name: str, features: np.ndarray, labels: np.ndarray):
        """Save train/val/test split"""
        output_path = Path(output_dir) / f'{name}_features.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump((features, labels), f)
        logger.info(f"Saved {name} split: {len(features)} samples to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml",
                       help="Path to dataset config file")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config)
    generator.generate_dataset()
