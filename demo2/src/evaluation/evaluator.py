"""
Model Evaluator
Đánh giá models và tạo báo cáo false positives/negatives
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Đánh giá ML models"""
    
    def __init__(self, model, model_name: str):
        """
        Args:
            model: Trained model (must have predict and predict_proba methods)
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                  output_dir: str = "results/") -> Dict:
        """
        Đánh giá model trên test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save results
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {self.model_name}...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # False positives and negatives
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'confusion_matrix': cm.tolist()
        }
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{output_dir}/{self.model_name}_metrics.csv", index=False)
        
        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_dir}/{self.model_name}_classification_report.csv")
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(cm, f"{output_dir}/{self.model_name}_confusion_matrix.png")
        
        # Save ROC curve
        self._plot_roc_curve(y_test, y_proba[:, 1], f"{output_dir}/{self.model_name}_roc_curve.png")
        
        # Create detailed report
        self._create_detailed_report(metrics, report, f"{output_dir}/{self.model_name}_report.txt")
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"False Positives: {fp}, False Negatives: {fn}")
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, filepath: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Obfuscated'],
                   yticklabels=['Benign', 'Obfuscated'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, filepath: str):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _create_detailed_report(self, metrics: Dict, report: Dict, filepath: str):
        """Create detailed text report"""
        with open(filepath, 'w') as f:
            f.write(f"Evaluation Report - {self.model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives:  {metrics['true_positives']}\n")
            f.write(f"  True Negatives:  {metrics['true_negatives']}\n")
            f.write(f"  False Positives: {metrics['false_positives']}\n")
            f.write(f"  False Negatives: {metrics['false_negatives']}\n\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report.__str__())

