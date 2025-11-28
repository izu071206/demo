"""
Opcode N-gram Extractor
Trích xuất opcode n-grams từ binary files
"""

import capstone
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class OpcodeExtractor:
    """Trích xuất opcode n-grams từ binary"""
    
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
    
    def disassemble(self, binary_data: bytes) -> List[str]:
        """
        Disassemble binary và trả về danh sách opcodes
        
        Args:
            binary_data: Raw binary data
            
        Returns:
            List of opcode mnemonics
        """
        opcodes = []
        try:
            for instruction in self.md.disasm(binary_data, 0x1000):
                opcodes.append(instruction.mnemonic)
        except Exception as e:
            logger.warning(f"Error disassembling: {e}")
        
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
            return Counter()
        
        ngrams = []
        for i in range(len(opcodes) - n + 1):
            ngram = ' '.join(opcodes[i:i+n])
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def extract_features(self, binary_data: bytes, max_features: int = 1000) -> Dict[str, np.ndarray]:
        """
        Trích xuất tất cả opcode n-gram features
        
        Args:
            binary_data: Raw binary data
            max_features: Maximum number of features per n-gram type
            
        Returns:
            Dictionary với keys là n-gram sizes và values là feature vectors
        """
        opcodes = self.disassemble(binary_data)
        
        if not opcodes:
            logger.warning("No opcodes extracted")
            return {}
        
        features = {}
        
        for n in self.n_grams:
            ngrams = self.extract_ngrams(opcodes, n)
            
            # Lấy top n-grams
            top_ngrams = dict(ngrams.most_common(max_features))
            
            # Tạo feature vector (frequency-based)
            feature_vector = np.array(list(top_ngrams.values()), dtype=np.float32)
            
            # Normalize
            if len(feature_vector) > 0:
                feature_vector = feature_vector / (np.sum(feature_vector) + 1e-10)
            
            features[f'opcode_{n}gram'] = feature_vector
        
        return features
    
    def extract_from_file(self, file_path: str, max_features: int = 1000) -> Dict[str, np.ndarray]:
        """
        Trích xuất features từ file
        
        Args:
            file_path: Path to binary file
            max_features: Maximum number of features per n-gram type
            
        Returns:
            Dictionary of features
        """
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            return self.extract_features(binary_data, max_features)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}

