"""
Dataset Generation
Tạo dataset từ mã nguồn hợp pháp và obfuscated samples
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pefile
import yaml
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate dataset từ binary files"""
    
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
        self.feature_pipeline = FeaturePipeline(pipeline_cfg)
        self.metadata_path = Path(self.config['processed_features_dir']) / 'feature_metadata.json'
    
    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """
        Trích xuất tất cả features từ một file
        
        Args:
            file_path: Path to binary file
            
        Returns:
            Combined feature vector
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
                # Hoặc nếu có bytes không phải printable ASCII
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
        """Sử dụng tên thư mục để suy ra malware family hoặc ứng dụng."""
        try:
            relative = file_path.relative_to(root_dir)
            parts = relative.parts
            if len(parts) > 1:
                return parts[0]
        except Exception:
            pass
        return file_path.parent.name or ("benign" if label == 0 else "malware")
    
    def process_directory(self, directory: str, label: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Process tất cả files trong directory
        
        Args:
            directory: Directory path
            label: Label (0: benign, 1: obfuscated)
            
        Returns:
            Tuple of (features array, labels array)
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
            logger.info(f"  Total files found: {len(all_files)}")
            logger.info(f"  Valid binary files: {len(files)}")
            logger.info(f"  Please add binary samples (.exe, .dll, .bin, etc.) to {directory}")
            return np.array([]), np.array([]), []
        
        logger.info(f"Processing {len(files)} valid binary files from {directory}")
        
        root_dir = Path(directory)
        for file_path in tqdm(files, desc=f"Processing {directory}"):
            try:
                if not self.validate_pe(file_path):
                    logger.debug("Skipping invalid PE file: %s", file_path.name)
                    continue
                features = self.extract_features_from_file(str(file_path))
                if len(features) > 0:
                    features_list.append(features)
                    labels_list.append(label)
                    metadata_list.append({
                        'file_path': str(file_path),
                        'label': label,
                        'family': self.get_family_name(root_dir, file_path, label),
                        'source_dir': directory,
                        'size_bytes': file_path.stat().st_size,
                    })
                else:
                    logger.debug(f"No features extracted from {file_path.name}")
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        if features_list:
            max_len = max(len(f) for f in features_list)
            padded_features = [
                self.feature_pipeline.pad_vector(f, max_len)
                for f in features_list
            ]
            return np.array(padded_features), np.array(labels_list), metadata_list
        else:
            return np.array([]), np.array([]), []

    def pad_matrix(self, matrix: np.ndarray, target_dim: int) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        current_dim = matrix.shape[1]
        if current_dim == target_dim:
            return matrix
        if current_dim < target_dim:
            padding = np.zeros((matrix.shape[0], target_dim - current_dim))
            return np.hstack([matrix, padding])
        return matrix[:, :target_dim]
    
    def save_split(self, output_dir: str, name: str, features: np.ndarray, labels: np.ndarray):
        with open(os.path.join(output_dir, f'{name}_features.pkl'), 'wb') as f:
            pickle.dump((features, labels), f)
    
    def generate_dataset(self):
        """Generate complete dataset"""
        logger.info("Starting dataset generation...")
        
        # Process benign samples
        benign_dir = self.config['benign_source_dir']
        benign_features, benign_labels, benign_metadata = self.process_directory(benign_dir, label=0)
        
        # Process obfuscated samples
        obfuscated_dir = self.config['obfuscated_output_dir']
        obfuscated_features, obfuscated_labels, obfuscated_metadata = self.process_directory(obfuscated_dir, label=1)
        
        # Combine
        available_sets = [arr for arr in [benign_features, obfuscated_features] if arr.size > 0]
        if not available_sets:
            logger.error("No features extracted!")
            return

        target_dim = max(arr.shape[1] for arr in available_sets)
        benign_features = self.pad_matrix(benign_features, target_dim)
        obfuscated_features = self.pad_matrix(obfuscated_features, target_dim)

        if len(benign_features) > 0 and len(obfuscated_features) > 0:
            all_features = np.vstack([benign_features, obfuscated_features])
            all_labels = np.hstack([benign_labels, obfuscated_labels])
            all_metadata = benign_metadata + obfuscated_metadata
        elif len(benign_features) > 0:
            all_features = benign_features
            all_labels = benign_labels
            all_metadata = benign_metadata
        elif len(obfuscated_features) > 0:
            all_features = obfuscated_features
            all_labels = obfuscated_labels
            all_metadata = obfuscated_metadata
        else:
            logger.error("No features extracted!")
            return

        groups = np.array([meta.get('family', 'unknown') for meta in all_metadata])
        rng = self.config.get('random_state', 42)
        test_ratio = self.config.get('test_ratio', 0.15)
        val_ratio = self.config.get('val_ratio', 0.15)
        train_ratio = self.config.get('train_ratio', 0.7)

        def group_split(features, labels, groups, test_size, random_state):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            try:
                train_idx, test_idx = next(splitter.split(features, labels, groups=groups))
                return train_idx, test_idx
            except ValueError as exc:
                logger.warning("Group split failed (%s). Falling back to random split.", exc)
                indices = np.random.permutation(len(features))
                cutoff = int(len(features) * (1 - test_size))
                return indices[:cutoff], indices[cutoff:]

        trainval_idx, test_idx = group_split(all_features, all_labels, groups, test_ratio, rng)
        remaining_ratio = train_ratio + val_ratio
        val_relative = val_ratio / remaining_ratio if remaining_ratio > 0 else 0.0

        trainval_features = all_features[trainval_idx]
        trainval_labels = all_labels[trainval_idx]
        trainval_groups = groups[trainval_idx]

        train_idx_rel, val_idx_rel = group_split(trainval_features, trainval_labels, trainval_groups, val_relative, rng + 1)
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        X_train, y_train = all_features[train_idx], all_labels[train_idx]
        X_val, y_val = all_features[val_idx], all_labels[val_idx]
        X_test, y_test = all_features[test_idx], all_labels[test_idx]

        split_map = ['train'] * len(all_features)
        for idx in val_idx:
            split_map[idx] = 'val'
        for idx in test_idx:
            split_map[idx] = 'test'

        metadata_df = pd.DataFrame(all_metadata)
        metadata_df['split'] = split_map

        output_dir = self.config['processed_features_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        self.save_split(output_dir, 'train', X_train, y_train)
        self.save_split(output_dir, 'val', X_val, y_val)
        self.save_split(output_dir, 'test', X_test, y_test)

        metadata_path = Path(output_dir) / 'sample_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)

        feature_dim = None
        for split in (X_train, X_val, X_test):
            if len(split) > 0:
                feature_dim = split.shape[1]
                break
        
        if feature_dim is None:
            feature_dim = all_features.shape[1] if all_features.ndim > 1 else len(all_features)
        
        self.feature_pipeline.save_metadata(self.metadata_path, int(feature_dim))
        
        feature_dim_display = X_train.shape[1] if len(X_train) > 0 else target_dim
        logger.info("Dataset generated:")
        logger.info("  Train: %d samples", len(X_train))
        logger.info("  Val: %d samples", len(X_val))
        logger.info("  Test: %d samples", len(X_test))
        logger.info("  Feature dimension: %d", feature_dim_display)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml",
                       help="Path to dataset config file")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config)
    generator.generate_dataset()

