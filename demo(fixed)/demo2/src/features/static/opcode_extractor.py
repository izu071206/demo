"""
Opcode N-gram Extractor - FIXED VERSION with better error handling and logging
"""

import capstone
import numpy as np
from collections import Counter
from typing import List, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class OpcodeExtractor:
    """Trích xuất opcode n-grams từ binary - FIXED VERSION"""
    
    def __init__(self, arch='x86', mode=64, n_grams: List[int] = [2, 3, 4]):
        """
        Args:
            arch: Architecture (x86, arm, mips, etc.)
            mode: 32 or 64 bit
            n_grams: List of n-gram sizes to extract
        """
        self.arch = arch
        self.mode = mode
        self.n_grams = n_grams
        
        # Initialize Capstone disassembler
        arch_map = {
            'x86': capstone.CS_ARCH_X86,
            'arm': capstone.CS_ARCH_ARM,
            'mips': capstone.CS_ARCH_MIPS
        }
        mode_map = {
            32: capstone.CS_MODE_32,
            64: capstone.CS_MODE_64
        }
        
        self.md = capstone.Cs(
            arch_map.get(arch, capstone.CS_ARCH_X86),
            mode_map.get(mode, capstone.CS_MODE_64)
        )
        self.md.detail = True
        
        logger.debug(f"OpcodeExtractor initialized: arch={arch}, mode={mode}, n_grams={n_grams}")
    
    def disassemble(self, binary_data: bytes) -> List[str]:
        """
        Disassemble binary và trả về danh sách opcodes - WITH DETAILED LOGGING
        
        Args:
            binary_data: Raw binary data
            
        Returns:
            List of opcode mnemonics
        """
        opcodes = []
        
        if not binary_data or len(binary_data) == 0:
            logger.warning("Empty binary data provided to disassemble()")
            return opcodes
        
        logger.debug(f"Disassembling {len(binary_data):,} bytes of binary data...")
        
        try:
            instruction_count = 0
            for instruction in self.md.disasm(binary_data, 0x1000):
                opcodes.append(instruction.mnemonic)
                instruction_count += 1
            
            logger.info(f"Disassembled {instruction_count} instructions")
            
            if instruction_count == 0:
                logger.warning("No instructions disassembled! Binary may be invalid or encrypted.")
            
        except Exception as e:
            logger.error(f"Error during disassembly: {e}")
            import traceback
            traceback.print_exc()
        
        return opcodes
    
    def extract_ngrams(self, opcodes: List[str], n: int) -> Counter:
        """
        Trích xuất n-grams từ opcodes
        
        Args:
            opcodes: List of opcodes
            n: n-gram size
            
        Returns:
            Counter of n-grams
        """
        if len(opcodes) < n:
            logger.debug(f"Not enough opcodes ({len(opcodes)}) for {n}-grams")
            return Counter()
        
        ngrams = []
        for i in range(len(opcodes) - n + 1):
            ngram = ' '.join(opcodes[i:i+n])
            ngrams.append(ngram)
        
        counter = Counter(ngrams)
        logger.debug(f"Extracted {len(counter)} unique {n}-grams from {len(ngrams)} total")
        
        return counter
    
    def extract_features(self, binary_data: bytes, max_features: int = 1000) -> Dict[str, np.ndarray]:
        """
        Trích xuất tất cả opcode n-gram features - WITH VALIDATION
        
        Args:
            binary_data: Raw binary data
            max_features: Maximum number of features per n-gram type
            
        Returns:
            Dictionary với keys là n-gram sizes và values là feature vectors
        """
        logger.info(f"Extracting opcode features (max {max_features} per n-gram type)...")
        
        if not binary_data or len(binary_data) == 0:
            logger.error("Cannot extract features from empty binary data!")
            return {}
        
        # Disassemble
        opcodes = self.disassemble(binary_data)
        
        if not opcodes:
            logger.error("No opcodes extracted! Feature extraction failed.")
            return {}
        
        logger.info(f"Working with {len(opcodes)} opcodes")
        
        features = {}
        
        for n in self.n_grams:
            logger.debug(f"Extracting {n}-grams...")
            
            ngrams = self.extract_ngrams(opcodes, n)
            
            if not ngrams:
                logger.warning(f"No {n}-grams extracted")
                continue
            
            # Lấy top n-grams
            top_ngrams = dict(ngrams.most_common(max_features))
            
            logger.debug(f"Selected top {len(top_ngrams)} {n}-grams out of {len(ngrams)} unique")
            
            # Tạo feature vector (frequency-based)
            feature_vector = np.array(list(top_ngrams.values()), dtype=np.float32)
            
            # Validate vector
            if len(feature_vector) == 0:
                logger.warning(f"Empty feature vector for {n}-grams")
                continue
            
            # Normalize
            vector_sum = np.sum(feature_vector)
            if vector_sum > 0:
                feature_vector = feature_vector / vector_sum
                logger.debug(f"{n}-gram vector: {len(feature_vector)} features, normalized sum={np.sum(feature_vector):.4f}")
            else:
                logger.warning(f"{n}-gram vector has zero sum!")
            
            features[f'opcode_{n}gram'] = feature_vector
            
            # Log some example n-grams
            top_5 = list(top_ngrams.items())[:5]
            logger.debug(f"  Top 5 {n}-grams: {[f'{ng}({cnt})' for ng, cnt in top_5]}")
        
        if not features:
            logger.error("No features extracted from any n-gram size!")
        else:
            logger.info(f"Successfully extracted {len(features)} n-gram feature types")
        
        return features
    
    def extract_from_file(self, file_path: str, max_features: int = 1000) -> Dict[str, np.ndarray]:
        """
        Trích xuất features từ file - WITH FILE VALIDATION
        
        Args:
            file_path: Path to binary file
            max_features: Maximum number of features per n-gram type
            
        Returns:
            Dictionary of features
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return {}
        
        file_size = file_path_obj.stat().st_size
        logger.info(f"Reading binary file: {file_path_obj.name} ({file_size:,} bytes)")
        
        if file_size == 0:
            logger.error(f"File is empty: {file_path}")
            return {}
        
        if file_size < 100:
            logger.warning(f"File is very small ({file_size} bytes), may not be valid binary")
        
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            
            logger.debug(f"Successfully read {len(binary_data):,} bytes")
            
            # Basic binary validation
            if len(binary_data) < 64:
                logger.warning("Binary data is too short for meaningful analysis")
            
            # Check for PE header (Windows executables)
            if binary_data[:2] == b'MZ':
                logger.debug("Detected PE (Windows) executable")
            # Check for ELF header (Linux executables)
            elif binary_data[:4] == b'\x7fELF':
                logger.debug("Detected ELF (Linux) executable")
            else:
                logger.warning("Unknown binary format (not PE or ELF)")
            
            return self.extract_features(binary_data, max_features)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}