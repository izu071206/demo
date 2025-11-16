"""
Script để tạo cấu trúc thư mục mẫu cho dataset
"""

import os
from pathlib import Path


def create_directory_structure():
    """Tạo cấu trúc thư mục cần thiết"""
    
    directories = [
        'data/raw',
        'data/benign',
        'data/obfuscated',
        'data/processed',
        'data/upload',
        'models',
        'results',
        'config',
        'tests',
        'scripts',
        'src/features/static',
        'src/features/dynamic',
        'src/models',
        'src/dataset',
        'src/evaluation',
        'src/dashboard/templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Tạo file .gitkeep để giữ thư mục trong git
        (Path(directory) / '.gitkeep').touch()
        print(f"Created: {directory}")
    
    print("\n[OK] Cau truc thu muc da duoc tao!")
    print("\n[NOTE] Luu y:")
    print("  - Dat cac file binary hop phap vao data/benign/")
    print("  - Dat cac file da obfuscate vao data/obfuscated/")
    print("  - Cac model da train se duoc luu vao models/")
    print("  - Ket qua danh gia se duoc luu vao results/")


if __name__ == "__main__":
    create_directory_structure()

