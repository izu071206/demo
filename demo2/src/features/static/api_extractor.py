"""
API Calls Extractor
Trích xuất API calls từ binary (static analysis)
"""

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pefile

logger = logging.getLogger(__name__)

DEFAULT_API_LIST = Path("config/api_list.yaml")


class APIExtractor:
    """Trích xuất API calls từ PE files"""
    
    def __init__(self, api_list_path: Optional[str] = None):
        """Initialize API extractor"""
        self.api_list_path = Path(api_list_path) if api_list_path else DEFAULT_API_LIST
        self.common_apis = self._load_api_catalog()
        self.dynamic_indicators = {
            "LoadLibraryA",
            "LoadLibraryW",
            "LoadLibraryExA",
            "LoadLibraryExW",
            "GetProcAddress",
            "GetModuleHandleA",
            "GetModuleHandleW",
            "LdrGetProcedureAddress",
        }
    
    def _load_api_catalog(self) -> Set[str]:
        """Load danh sách các API phổ biến"""
        if self.api_list_path and self.api_list_path.exists():
            try:
                import yaml

                with self.api_list_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                categories = data.get("categories", {})
                flattened = set()
                for items in categories.values():
                    flattened.update(items)
                logger.info("Loaded %d API entries from %s", len(flattened), self.api_list_path)
                return flattened
            except Exception as exc:
                logger.warning("Failed to load API list from %s: %s", self.api_list_path, exc)
        
        logger.info("Falling back to built-in API catalog.")
        return {
            'CreateFileA', 'CreateFileW', 'ReadFile', 'WriteFile', 'DeleteFileA', 'DeleteFileW',
            'CopyFileA', 'CopyFileW', 'MoveFileA', 'MoveFileW', 'FindFirstFileA', 'FindNextFileA',
            'RegOpenKeyExA', 'RegSetValueExA', 'RegGetValueA', 'RegDeleteKeyA',
            'socket', 'connect', 'send', 'recv', 'WSAStartup', 'InternetOpenA', 'InternetConnectA',
            'HttpOpenRequestA', 'CreateProcessA', 'CreateThread', 'TerminateProcess', 'OpenProcess',
            'WriteProcessMemory', 'ReadProcessMemory', 'VirtualAlloc', 'VirtualFree', 'HeapAlloc',
            'HeapFree', 'CryptEncrypt', 'CryptDecrypt', 'CryptCreateHash', 'GetSystemTime',
            'GetTickCount', 'Sleep', 'ExitProcess',
        }
    
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
    
    def extract_strings(self, binary_path: str, min_length: int = 4, chunk_size: int = 1024 * 512) -> List[str]:
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
            current_chars: List[str] = []
            with open(binary_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    for byte in chunk:
                        if 32 <= byte <= 126:  # Printable ASCII
                            current_chars.append(chr(byte))
                        else:
                            if len(current_chars) >= min_length:
                                strings.append(''.join(current_chars))
                            current_chars = []
            if len(current_chars) >= min_length:
                strings.append(''.join(current_chars))
        except Exception as e:
            logger.warning(f"Error extracting strings from {binary_path}: {e}")
        
        return strings

    def _detect_dynamic_imports(self, imports: Dict[str, List[str]], strings: List[str]) -> Dict[str, float]:
        counts = defaultdict(int)
        for dll_functions in imports.values():
            for func in dll_functions:
                if func in self.dynamic_indicators:
                    counts[func] += 1

        lowered_strings = [s.lower() for s in strings]
        for indicator in self.dynamic_indicators:
            indicator_lower = indicator.lower()
            for s in lowered_strings:
                if indicator_lower in s:
                    counts[indicator] += 1

        features = {
            'dynamic_loader_score': float(sum(counts.values())),
            'dynamic_loader_unique': float(len(counts)),
            'uses_dynamic_loading': 1.0 if counts else 0.0,
        }
        for indicator in self.dynamic_indicators:
            features[f'call_{indicator.lower()}'] = float(counts.get(indicator, 0))
        return features
    
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
        features['api_entropy'] = float(
            -np.sum(feature_vector * np.log2(feature_vector + 1e-12))
        ) if len(feature_vector) else 0.0

        # Dynamic loader heuristics
        dynamic_features = self._detect_dynamic_imports(imports, strings)
        features.update(dynamic_features)
        
        return features

