"""
Test Feature Extraction
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.static import OpcodeExtractor, CFGExtractor, APIExtractor
from src.features.feature_combiner import FeatureCombiner


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction modules"""
    
    def test_opcode_extractor(self):
        """Test opcode extraction"""
        # Create dummy binary data (x86-64 instructions)
        # mov eax, 1; add eax, 2; ret
        binary_data = b'\xb8\x01\x00\x00\x00\x83\xc0\x02\xc3'
        
        extractor = OpcodeExtractor(arch='x86', mode=64, n_grams=[2, 3])
        features = extractor.extract_features(binary_data, max_features=100)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    def test_feature_combiner(self):
        """Test feature combiner"""
        combiner = FeatureCombiner()
        
        features = {
            'opcode_2gram': np.array([0.1, 0.2, 0.3]),
            'num_nodes': 100.0,
            'api_calls': np.array([0.5, 0.3, 0.2])
        }
        
        combined = combiner.combine(features)
        
        self.assertIsInstance(combined, np.ndarray)
        self.assertGreater(len(combined), 0)


if __name__ == '__main__':
    unittest.main()

