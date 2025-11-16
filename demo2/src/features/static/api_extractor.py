"""
API Calls Extractor
Trích xuất API calls từ binary (static analysis)
"""

import pefile
import yara
import numpy as np
from collections import Counter
from typing import List, Dict, Set
import logging
import os

logger = logging.getLogger(__name__)


class APIExtractor:
    """Trích xuất API calls từ PE files"""
    
    def __init__(self):
        """Initialize API extractor"""
        self.common_apis = self._load_common_apis()
    
    def _load_common_apis(self) -> Set[str]:
        """Load danh sách các API phổ biến"""
        # Windows API categories
        common_apis = {
            # File operations
            'CreateFile', 'ReadFile', 'WriteFile', 'DeleteFile', 'CopyFile',
            'MoveFile', 'FindFirstFile', 'FindNextFile',
            
            # Registry
            'RegOpenKey', 'RegSetValue', 'RegGetValue', 'RegDeleteKey',
            
            # Network
            'socket', 'connect', 'send', 'recv', 'WSAStartup',
            'InternetOpen', 'InternetConnect', 'HttpOpenRequest',
            
            # Process/Thread
            'CreateProcess', 'CreateThread', 'TerminateProcess',
            'OpenProcess', 'WriteProcessMemory', 'ReadProcessMemory',
            
            # Memory
            'VirtualAlloc', 'VirtualFree', 'HeapAlloc', 'HeapFree',
            
            # Crypto
            'CryptEncrypt', 'CryptDecrypt', 'CryptCreateHash',
            
            # System
            'GetSystemTime', 'GetTickCount', 'Sleep', 'ExitProcess'
        }
        return common_apis
    
    def extract_imports(self, binary_path: str) -> Dict[str, List[str]]:
        """
        Trích xuất imports từ PE file
        
        Args:
            binary_path: Path to PE file
            
        Returns:
            Dictionary với keys là DLL names và values là list of functions
        """
        imports = {}
        
        try:
            pe = pefile.PE(binary_path)
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8', errors='ignore')
                    functions = []
                    
                    for imp in entry.imports:
                        if imp.name:
                            func_name = imp.name.decode('utf-8', errors='ignore')
                            functions.append(func_name)
                    
                    if functions:
                        imports[dll_name] = functions
            
            pe.close()
        
        except Exception as e:
            logger.warning(f"Error extracting imports from {binary_path}: {e}")
        
        return imports
    
    def extract_strings(self, binary_path: str, min_length: int = 4) -> List[str]:
        """
        Trích xuất strings từ binary
        
        Args:
            binary_path: Path to binary file
            min_length: Minimum string length
            
        Returns:
            List of strings
        """
        strings = []
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            current_string = ""
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) >= min_length:
                        strings.append(current_string)
                    current_string = ""
            
            if len(current_string) >= min_length:
                strings.append(current_string)
        
        except Exception as e:
            logger.warning(f"Error extracting strings from {binary_path}: {e}")
        
        return strings
    
    def extract_api_features(self, binary_path: str, max_features: int = 500) -> Dict[str, np.ndarray]:
        """
        Trích xuất API call features
        
        Args:
            binary_path: Path to binary file
            max_features: Maximum number of features
            
        Returns:
            Dictionary of API features
        """
        features = {}
        
        # Extract imports
        imports = self.extract_imports(binary_path)
        all_apis = []
        for dll, functions in imports.items():
            all_apis.extend(functions)
        
        # Extract strings (có thể chứa API names)
        strings = self.extract_strings(binary_path)
        
        # Tìm API calls trong strings
        for string in strings:
            for api in self.common_apis:
                if api.lower() in string.lower():
                    all_apis.append(api)
        
        # Count API frequencies
        api_counter = Counter(all_apis)
        
        # Tạo feature vector
        top_apis = dict(api_counter.most_common(max_features))
        
        if top_apis:
            feature_vector = np.array(list(top_apis.values()), dtype=np.float32)
            # Normalize
            feature_vector = feature_vector / (np.sum(feature_vector) + 1e-10)
            features['api_calls'] = feature_vector
        else:
            features['api_calls'] = np.array([], dtype=np.float32)
        
        # Metadata
        features['num_imports'] = len(all_apis)
        features['num_dlls'] = len(imports)
        
        return features

