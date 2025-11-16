"""
Script kiểm tra môi trường và dependencies
"""

import sys
import importlib

def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Can Python 3.8 tro len!")
        return False
    else:
        print("[OK] Python version OK")
        return True

def check_dependency(module_name, package_name=None):
    """Kiểm tra một dependency"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"[OK] {package_name} da cai")
        return True
    except ImportError:
        print(f"[ERROR] {package_name} chua cai - Chay: pip install {package_name}")
        return False

def main():
    print("="*50)
    print("KIEM TRA MOI TRUONG")
    print("="*50)
    
    # Kiểm tra Python
    python_ok = check_python_version()
    print()
    
    # Kiểm tra dependencies
    print("Kiem tra dependencies:")
    print("-" * 50)
    
    dependencies = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('torch', 'torch'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
        ('capstone', 'capstone'),
        ('pefile', 'pefile'),
        ('flask', 'flask'),
    ]
    
    results = []
    for module, package in dependencies:
        results.append(check_dependency(module, package))
    
    # Optional dependencies
    print("\nOptional dependencies:")
    print("-" * 50)
    optional = [
        ('angr', 'angr'),
    ]
    
    for module, package in optional:
        if check_dependency(module, package):
            print("  (angr duoc khuyen nghi cho CFG extraction)")
        else:
            print("  (angr khong bat buoc, co the bo qua CFG extraction)")
    
    print()
    print("="*50)
    
    # Tổng kết
    required_ok = all(results)
    
    if python_ok and required_ok:
        print("[OK] Moi truong da san sang!")
        print("\nBuoc tiep theo:")
        print("1. python scripts/create_sample_structure.py")
        print("2. Them binary samples vao data/benign/ va data/obfuscated/")
        print("3. python main.py generate-dataset")
        print("4. python main.py train")
        return 0
    else:
        print("[ERROR] Can cai dat them dependencies!")
        print("\nChay: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

