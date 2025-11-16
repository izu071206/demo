"""
Main Entry Point
Script chính để chạy các chức năng của hệ thống
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Obfuscation Detection System - Main Entry Point"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dataset generation
    dataset_parser = subparsers.add_parser('generate-dataset', help='Generate dataset')
    dataset_parser.add_argument('--config', type=str, default='config/dataset_config.yaml')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', type=str, default='config/train_config.yaml')
    
    # Evaluation
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--model-type', type=str, required=True,
                            choices=['random_forest', 'xgboost', 'neural_network'])
    eval_parser.add_argument('--test-data', type=str, default='data/processed/test_features.pkl')
    eval_parser.add_argument('--output-dir', type=str, default='results/')
    
    # Dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Start dashboard')
    dashboard_parser.add_argument('--port', type=int, default=5000)
    dashboard_parser.add_argument('--host', type=str, default='0.0.0.0')
    
    args = parser.parse_args()
    
    if args.command == 'generate-dataset':
        from src.dataset.generate_dataset import DatasetGenerator
        generator = DatasetGenerator(args.config)
        generator.generate_dataset()
    
    elif args.command == 'train':
        from src.models.train import main as train_main
        import sys
        sys.argv = ['train.py', '--config', args.config]
        train_main()
    
    elif args.command == 'evaluate':
        from src.evaluation.evaluate import main as eval_main
        import sys
        sys.argv = ['evaluate.py', '--model', args.model, '--model-type', args.model_type,
                   '--test-data', args.test_data, '--output-dir', args.output_dir]
        eval_main()
    
    elif args.command == 'dashboard':
        from src.dashboard.app import app
        app.run(debug=True, host=args.host, port=args.port)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

