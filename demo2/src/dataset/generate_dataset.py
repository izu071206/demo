"""
Dataset Generation
Tạo dataset từ mã nguồn hợp pháp và obfuscated samples
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
import pickle
from tqdm import tqdm

from src.features.static import OpcodeExtractor, CFGExtractor, APIExtractor
from src.features.feature_combiner import FeatureCombiner

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
        
        # Initialize extractors
        self.opcode_extractor = OpcodeExtractor(
            n_grams=self.config['features']['opcode_ngrams']['n']
        )
        self.cfg_extractor = CFGExtractor()
        self.api_extractor = APIExtractor()
        self.feature_combiner = FeatureCombiner()
    
    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """
        Trích xuất tất cả features từ một file
        
        Args:
            file_path: Path to binary file
            
        Returns:
            Combined feature vector
        """
        all_features = {}
        
        # Opcode n-grams
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            opcode_features = self.opcode_extractor.extract_features(
                binary_data,
                max_features=self.config['features']['opcode_ngrams']['max_features']
            )
            all_features.update(opcode_features)
        except Exception as e:
            logger.warning(f"Error extracting opcodes from {file_path}: {e}")
        
        # CFG features
        try:
            cfg_features = self.cfg_extractor.extract_features(file_path)
            all_features.update(cfg_features)
        except Exception as e:
            logger.warning(f"Error extracting CFG from {file_path}: {e}")
        
        # API calls
        try:
            api_features = self.api_extractor.extract_api_features(
                file_path,
                max_features=self.config['features']['api_calls']['max_features']
            )
            all_features.update(api_features)
        except Exception as e:
            logger.warning(f"Error extracting APIs from {file_path}: {e}")
        
        # Combine features
        combined = self.feature_combiner.combine(all_features)
        
        return combined
    
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
    
    def process_directory(self, directory: str, label: int) -> tuple:
        """
        Process tất cả files trong directory
        
        Args:
            directory: Directory path
            label: Label (0: benign, 1: obfuscated)
            
        Returns:
            Tuple of (features array, labels array)
        """
        features_list = []
        labels_list = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return np.array([]), np.array([])
        
        # Lấy tất cả files và filter
        all_files = list(Path(directory).rglob('*'))
        files = [f for f in all_files if f.is_file() and self.is_valid_binary_file(f)]
        
        if len(files) == 0:
            logger.warning(f"No valid binary files found in {directory}")
            logger.info(f"  Total files found: {len(all_files)}")
            logger.info(f"  Valid binary files: {len(files)}")
            logger.info(f"  Please add binary samples (.exe, .dll, .bin, etc.) to {directory}")
            return np.array([]), np.array([])
        
        logger.info(f"Processing {len(files)} valid binary files from {directory}")
        
        for file_path in tqdm(files, desc=f"Processing {directory}"):
            try:
                features = self.extract_features_from_file(str(file_path))
                if len(features) > 0:
                    features_list.append(features)
                    labels_list.append(label)
                else:
                    logger.debug(f"No features extracted from {file_path.name}")
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        if features_list:
            # Pad features to same length
            max_len = max(len(f) for f in features_list)
            padded_features = []
            for f in features_list:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), 'constant')
                else:
                    padded = f[:max_len]
                padded_features.append(padded)
            
            return np.array(padded_features), np.array(labels_list)
        else:
            return np.array([]), np.array([])
    
    def generate_dataset(self):
        """Generate complete dataset"""
        logger.info("Starting dataset generation...")
        
        # Process benign samples
        benign_dir = self.config['benign_source_dir']
        benign_features, benign_labels = self.process_directory(benign_dir, label=0)
        
        # Process obfuscated samples
        obfuscated_dir = self.config['obfuscated_output_dir']
        obfuscated_features, obfuscated_labels = self.process_directory(obfuscated_dir, label=1)
        
        # Combine
        if len(benign_features) > 0 and len(obfuscated_features) > 0:
            all_features = np.vstack([benign_features, obfuscated_features])
            all_labels = np.hstack([benign_labels, obfuscated_labels])
        elif len(benign_features) > 0:
            all_features = benign_features
            all_labels = benign_labels
        elif len(obfuscated_features) > 0:
            all_features = obfuscated_features
            all_labels = obfuscated_labels
        else:
            logger.error("No features extracted!")
            return
        
        # Shuffle
        indices = np.random.permutation(len(all_features))
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        
        # Split
        n = len(all_features)
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        X_train = all_features[:train_end]
        y_train = all_labels[:train_end]
        X_val = all_features[train_end:val_end]
        y_val = all_labels[train_end:val_end]
        X_test = all_features[val_end:]
        y_test = all_labels[val_end:]
        
        # Save
        output_dir = self.config['processed_features_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'train_features.pkl'), 'wb') as f:
            pickle.dump((X_train, y_train), f)
        
        with open(os.path.join(output_dir, 'val_features.pkl'), 'wb') as f:
            pickle.dump((X_val, y_val), f)
        
        with open(os.path.join(output_dir, 'test_features.pkl'), 'wb') as f:
            pickle.dump((X_test, y_test), f)
        
        logger.info(f"Dataset generated:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        logger.info(f"  Feature dimension: {X_train.shape[1]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml",
                       help="Path to dataset config file")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config)
    generator.generate_dataset()

