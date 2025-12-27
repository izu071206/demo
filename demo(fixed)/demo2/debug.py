"""
Debug Pipeline Tool
Helps diagnose issues with feature extraction and model predictions
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pickle

# Setup path
current_dir = Path(__file__).parent
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from src.pipeline.inference_pipeline import InferencePipeline

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def check_dataset():
    """Check dataset files and their labels."""
    print("\n" + "="*80)
    print("CHECKING DATASET")
    print("="*80)
    
    processed_dir = project_root / "data" / "processed"
    
    for split in ['train', 'val', 'test']:
        split_file = processed_dir / f"{split}_features.pkl"
        
        if not split_file.exists():
            print(f"❌ {split}_features.pkl NOT FOUND")
            continue
            
        with open(split_file, 'rb') as f:
            X, y = pickle.load(f)
        
        print(f"\n{split.upper()} SET:")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Labels: {np.unique(y, return_counts=True)}")
        print(f"  Label 0 (Benign): {np.sum(y == 0)} samples ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  Label 1 (Obfuscated): {np.sum(y == 1)} samples ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"  Feature stats:")
        print(f"    Non-zero features per sample: {np.count_nonzero(X, axis=1).mean():.1f} ± {np.count_nonzero(X, axis=1).std():.1f}")
        print(f"    Feature sum per sample: {np.sum(X, axis=1).mean():.4f} ± {np.sum(X, axis=1).std():.4f}")
        
        # Check if all samples are identical (red flag!)
        if len(X) > 1:
            all_same = np.all(X[0] == X[1:])
            if all_same:
                print(f"  ⚠️  WARNING: ALL SAMPLES APPEAR IDENTICAL!")
            else:
                # Check diversity
                unique_rows = len(np.unique(X, axis=0))
                print(f"  Unique feature vectors: {unique_rows}/{len(X)} ({unique_rows/len(X)*100:.1f}%)")


def check_models():
    """Check trained models."""
    print("\n" + "="*80)
    print("CHECKING MODELS")
    print("="*80)
    
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.json")) + list(models_dir.glob("*.pt"))
    
    if not model_files:
        print("❌ No model files found")
        return
    
    print(f"\nFound {len(model_files)} model files:")
    for model_file in model_files:
        print(f"  - {model_file.name} ({model_file.stat().st_size / 1024:.1f} KB)")


def test_feature_extraction(file_path: str):
    """Test feature extraction on a single file."""
    print("\n" + "="*80)
    print(f"TESTING FEATURE EXTRACTION: {Path(file_path).name}")
    print("="*80)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Load config
    config_file = project_root / "config" / "dataset_config.yaml"
    if not config_file.exists():
        print(f"❌ Config not found: {config_file}")
        return
    
    config = FeaturePipelineConfig.from_dataset_config(str(config_file))
    pipeline = FeaturePipeline(config)
    
    # Extract features
    print("\nExtracting features...")
    feature_dict = pipeline.extract_feature_dict(file_path)
    
    print(f"\nFeature components extracted: {len(feature_dict)}")
    for key, val in feature_dict.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, non-zero={np.count_nonzero(val)}, sum={np.sum(val):.4f}")
        else:
            print(f"  {key}: {val}")
    
    # Combine features
    combined = pipeline.build_feature_vector(file_path)
    print(f"\nCombined feature vector:")
    print(f"  Shape: {combined.shape}")
    print(f"  Non-zero: {np.count_nonzero(combined)}")
    print(f"  Sum: {np.sum(combined):.4f}")
    print(f"  Mean: {np.mean(combined):.6f}")
    print(f"  Std: {np.std(combined):.6f}")
    print(f"  Range: [{np.min(combined):.6f}, {np.max(combined):.6f}]")


def test_model_prediction(file_path: str, model_type: str = "random_forest"):
    """Test model prediction on a file."""
    print("\n" + "="*80)
    print(f"TESTING MODEL PREDICTION: {model_type.upper()}")
    print("="*80)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Find model
    models_dir = project_root / "models"
    model_patterns = {
        'random_forest': '*.pkl',
        'xgboost': '*.json',
        'neural_network': '*.pt'
    }
    
    model_files = list(models_dir.glob(model_patterns.get(model_type, '*.pkl')))
    model_files = [f for f in model_files if model_type in f.name]
    
    if not model_files:
        print(f"❌ No {model_type} model found")
        return
    
    model_path = model_files[0]
    print(f"Using model: {model_path.name}")
    
    # Load feature metadata
    metadata_path = project_root / "data" / "processed" / "feature_metadata.json"
    if not metadata_path.exists():
        print(f"❌ Feature metadata not found: {metadata_path}")
        return
    
    # Check for scaler
    scaler_path = project_root / "models" / "scaler.pkl"
    scaler_str = str(scaler_path) if scaler_path.exists() else None
    
    # Initialize pipeline
    print("\nInitializing inference pipeline...")
    if scaler_str:
        print(f"Using scaler: {scaler_str}")
    pipeline = InferencePipeline(
        model_path=str(model_path),
        model_type=model_type,
        feature_metadata=str(metadata_path),
        scaler_path=scaler_str,  # CRITICAL: Pass scaler if available
        enable_explainability=False
    )
    
    # Predict
    print("\nRunning prediction...")
    result = pipeline.predict_file(file_path)
    
    print("\nRESULT:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Probabilities:")
    print(f"    Benign: {result['probabilities']['benign']:.4f}")
    print(f"    Obfuscated: {result['probabilities']['obfuscated']:.4f}")
    print(f"  Feature count: {result['feature_count']}")
    
    if 'debug_info' in result:
        print(f"\nDEBUG INFO:")
        for key, val in result['debug_info'].items():
            print(f"  {key}: {val}")


def compare_multiple_files(file_paths: list, model_type: str = "random_forest"):
    """Compare predictions on multiple files to check if model gives different results."""
    print("\n" + "="*80)
    print(f"COMPARING MULTIPLE FILES")
    print("="*80)
    
    # Setup pipeline
    models_dir = project_root / "models"
    model_patterns = {
        'random_forest': '*.pkl',
        'xgboost': '*.json',
        'neural_network': '*.pt'
    }
    
    model_files = list(models_dir.glob(model_patterns.get(model_type, '*.pkl')))
    model_files = [f for f in model_files if model_type in f.name]
    
    if not model_files:
        print(f"❌ No {model_type} model found")
        return
    
    model_path = model_files[0]
    metadata_path = project_root / "data" / "processed" / "feature_metadata.json"
    scaler_path = project_root / "models" / "scaler.pkl"
    scaler_str = str(scaler_path) if scaler_path.exists() else None
    
    pipeline = InferencePipeline(
        model_path=str(model_path),
        model_type=model_type,
        feature_metadata=str(metadata_path),
        scaler_path=scaler_str,  # CRITICAL: Pass scaler if available
        enable_explainability=False
    )
    
    results = []
    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"⚠️  Skipping {file_path} - not found")
            continue
        
        print(f"\nPredicting: {Path(file_path).name}")
        result = pipeline.predict_file(file_path)
        results.append({
            'file': Path(file_path).name,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'prob_benign': result['probabilities']['benign'],
            'prob_obf': result['probabilities']['obfuscated'],
            'feature_sum': result['debug_info']['feature_stats']['sum']
        })
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'File':<30} {'Prediction':<15} {'Confidence':<12} {'Feature Sum':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['file']:<30} {r['prediction']:<15} {r['confidence']:<12.4f} {r['feature_sum']:<12.4f}")
    
    # Check if all predictions are identical
    if len(results) > 1:
        all_same_pred = all(r['prediction'] == results[0]['prediction'] for r in results)
        all_same_conf = all(abs(r['confidence'] - results[0]['confidence']) < 0.001 for r in results)
        
        if all_same_pred and all_same_conf:
            print("\n⚠️  WARNING: ALL PREDICTIONS ARE IDENTICAL!")
            print("This indicates the model is not learning from features properly.")
        else:
            print("\n✅ Predictions vary across files (expected behavior)")


def main():
    """Main debug function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug pipeline tool")
    parser.add_argument('--check-dataset', action='store_true', help='Check dataset files')
    parser.add_argument('--check-models', action='store_true', help='Check model files')
    parser.add_argument('--test-features', type=str, help='Test feature extraction on file')
    parser.add_argument('--test-predict', type=str, help='Test prediction on file')
    parser.add_argument('--compare-files', nargs='+', help='Compare predictions on multiple files')
    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'xgboost', 'neural_network'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    if args.check_dataset:
        check_dataset()
    
    if args.check_models:
        check_models()
    
    if args.test_features:
        test_feature_extraction(args.test_features)
    
    if args.test_predict:
        test_model_prediction(args.test_predict, args.model_type)
    
    if args.compare_files:
        compare_multiple_files(args.compare_files, args.model_type)
    
    if not any([args.check_dataset, args.check_models, args.test_features, 
                args.test_predict, args.compare_files]):
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()