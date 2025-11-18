"""
Script để test với tất cả models cùng lúc
CẢNH BÁO: Chỉ chạy trong môi trường cách ly (sandbox/VM)
"""

import argparse
import sys
import json
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_malware import extract_features_from_sample, predict_with_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_all_models(sample_path: str, output_file: str = None):
    """
    Test một sample với tất cả models
    
    Args:
        sample_path: Path to sample file
        output_file: Output file for results (JSON)
    """
    sample_path = Path(sample_path)
    
    if not sample_path.exists():
        logger.error(f"Sample file not found: {sample_path}")
        return
    
    logger.info(f"Testing {sample_path.name} with all models...")
    
    # Extract features once
    logger.info("Extracting features...")
    features = extract_features_from_sample(str(sample_path))
    
    if len(features['features']) == 0:
        logger.error("No features extracted!")
        return
    
    logger.info(f"Extracted {len(features['features'])} features\n")
    
    # Define all models
    models = [
        {
            'name': 'Random Forest',
            'path': 'models/random_forest_model.pkl',
            'type': 'random_forest'
        },
        {
            'name': 'XGBoost',
            'path': 'models/xgboost_model.json',
            'type': 'xgboost'
        },
        {
            'name': 'Neural Network',
            'path': 'models/neural_network_model.pt',
            'type': 'neural_network'
        }
    ]
    
    results = {}
    all_predictions = []
    
    # Test with each model
    for model_info in models:
        model_path = Path(model_info['path'])
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}, skipping...")
            continue
        
        logger.info(f"Testing with {model_info['name']}...")
        
        try:
            prediction = predict_with_model(
                str(model_path),
                model_info['type'],
                features
            )
            
            results[model_info['type']] = {
                'model_name': model_info['name'],
                'prediction': prediction['prediction'],
                'confidence': float(prediction['confidence']),
                'probabilities': prediction['probabilities']
            }
            
            all_predictions.append(prediction['prediction'])
            
            logger.info(f"  Result: {prediction['prediction']} ({prediction['confidence']*100:.2f}%)")
        
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results[model_info['type']] = {
                'model_name': model_info['name'],
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*60)
    print("KẾT QUẢ TEST VỚI TẤT CẢ MODELS")
    print("="*60)
    print(f"File: {sample_path}")
    print(f"Features extracted: {len(features['features'])}")
    print("\n" + "-"*60)
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{result['model_name']}: ERROR - {result['error']}")
        else:
            print(f"{result['model_name']}:")
            print(f"  Kết quả: {result['prediction']}")
            print(f"  Độ tin cậy: {result['confidence']*100:.2f}%")
            print(f"  - Benign: {result['probabilities']['benign']*100:.2f}%")
            print(f"  - Obfuscated: {result['probabilities']['obfuscated']*100:.2f}%")
        print()
    
    # Consensus
    if len(all_predictions) > 0:
        obfuscated_count = sum(1 for p in all_predictions if p == 'Obfuscated')
        benign_count = len(all_predictions) - obfuscated_count
        
        print("-"*60)
        print("TỔNG KẾT:")
        print(f"  Obfuscated: {obfuscated_count}/{len(all_predictions)} models")
        print(f"  Benign: {benign_count}/{len(all_predictions)} models")
        
        if obfuscated_count > benign_count:
            consensus = "Obfuscated"
        elif benign_count > obfuscated_count:
            consensus = "Benign"
        else:
            consensus = "Không thống nhất"
        
        print(f"  Consensus: {consensus}")
        print("="*60)
    
    # Save results
    output_data = {
        'sample': str(sample_path),
        'timestamp': datetime.now().isoformat(),
        'feature_count': len(features['features']),
        'results': results,
        'consensus': {
            'obfuscated_votes': obfuscated_count if len(all_predictions) > 0 else 0,
            'benign_votes': benign_count if len(all_predictions) > 0 else 0,
            'total_models': len(all_predictions),
            'consensus': consensus if len(all_predictions) > 0 else "N/A"
        }
    }
    
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path("results") / f"all_models_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test malware sample với tất cả models cùng lúc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  CẢNH BÁO BẢO MẬT:
- Chỉ chạy trong môi trường cách ly (sandbox/VM)
- Không chạy trên hệ thống production
- Tuân thủ các quy định pháp lý về phân tích malware

Ví dụ:
  python scripts/test_all_models.py C:/malware/sample.exe
  python scripts/test_all_models.py C:/malware/sample.exe --output results/all_models_result.json
        """
    )
    
    parser.add_argument('sample', type=str, help='Path to sample file to analyze')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    test_all_models(args.sample, args.output)


if __name__ == "__main__":
    main()

