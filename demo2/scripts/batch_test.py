"""
Batch Test Script
Test nhiều malware samples cùng lúc
"""

import argparse
import json
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_malware import extract_features_from_sample, predict_with_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_test(malware_dir: str, model_path: str, model_type: str, 
               output_file: str = None, recursive: bool = False):
    """
    Test nhiều malware samples
    
    Args:
        malware_dir: Directory chứa malware samples
        model_path: Path to trained model
        model_type: Type of model
        output_file: Output file for results (JSON)
        recursive: Search recursively in subdirectories
    """
    malware_path = Path(malware_dir)
    
    if not malware_path.exists():
        logger.error(f"Directory not found: {malware_dir}")
        return
    
    # Tìm tất cả files
    if recursive:
        files = list(malware_path.rglob('*'))
    else:
        files = list(malware_path.glob('*'))
    
    # Filter chỉ lấy files (không phải directories)
    files = [f for f in files if f.is_file()]
    
    # Filter bỏ các file không phải binary
    binary_extensions = {'.exe', '.dll', '.bin', '.so', '.elf', ''}  # '' = no extension
    files = [f for f in files if f.suffix.lower() in binary_extensions or f.suffix == '']
    
    logger.info(f"Found {len(files)} files to test")
    
    if len(files) == 0:
        logger.warning("No binary files found!")
        return
    
    results = []
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] Testing: {file_path.name}")
        
        try:
            # Extract features
            features = extract_features_from_sample(str(file_path))
            
            if len(features['features']) == 0:
                logger.warning(f"  No features extracted from {file_path.name}")
                results.append({
                    'file': file_path.name,
                    'path': str(file_path),
                    'status': 'failed',
                    'error': 'No features extracted'
                })
                failed += 1
                continue
            
            # Predict
            prediction = predict_with_model(model_path, model_type, features)
            
            result = {
                'file': file_path.name,
                'path': str(file_path),
                'status': 'success',
                'prediction': prediction['prediction'],
                'confidence': float(prediction['confidence']),
                'probabilities': prediction['probabilities'],
                'model_type': prediction['model_type'],
                'feature_count': len(features['features'])
            }
            
            results.append(result)
            successful += 1
            
            logger.info(f"  Result: {prediction['prediction']}")
            logger.info(f"  Confidence: {prediction['confidence']*100:.2f}%")
        
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results.append({
                'file': file_path.name,
                'path': str(file_path),
                'status': 'failed',
                'error': str(e)
            })
            failed += 1
    
    # Summary
    summary = {
        'total_files': len(files),
        'successful': successful,
        'failed': failed,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'model_type': model_type
    }
    
    # Save results
    output_data = {
        'summary': summary,
        'results': results
    }
    
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path("results") / f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH TEST SUMMARY")
    print("="*50)
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_path}")
    print("="*50)
    
    # Statistics
    if successful > 0:
        obfuscated_count = sum(1 for r in results if r.get('prediction') == 'Obfuscated')
        benign_count = successful - obfuscated_count
        
        print(f"\nPredictions:")
        print(f"  Obfuscated: {obfuscated_count} ({obfuscated_count/successful*100:.1f}%)")
        print(f"  Benign: {benign_count} ({benign_count/successful*100:.1f}%)")
        
        avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('status') == 'success') / successful
        print(f"\nAverage confidence: {avg_confidence*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Batch test malware samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  CẢNH BÁO BẢO MẬT:
- Chỉ chạy trong môi trường cách ly (VM/sandbox)
- Không chạy trên hệ thống production
- Tuân thủ các quy định pháp lý về malware research

Ví dụ:
  python scripts/batch_test.py C:/malware_test/ \\
      --model models/random_forest_model.pkl \\
      --model-type random_forest \\
      --output results/batch_results.json
        """
    )
    
    parser.add_argument('malware_dir', type=str, 
                       help='Directory chứa malware samples')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['random_forest', 'xgboost', 'neural_network'],
                       help='Type of model')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON). Default: results/batch_test_TIMESTAMP.json')
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively in subdirectories')
    
    args = parser.parse_args()
    
    batch_test(
        args.malware_dir,
        args.model,
        args.model_type,
        args.output,
        args.recursive
    )


if __name__ == "__main__":
    main()

