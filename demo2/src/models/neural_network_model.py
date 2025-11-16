"""
Neural Network Model using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """Dataset class for PyTorch"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ObfuscationNN(nn.Module):
    """Neural Network for obfuscation detection"""
    
    def __init__(self, input_size: int, hidden_layers: List[int] = [128, 64, 32],
                 dropout: float = 0.3, num_classes: int = 2):
        """
        Args:
            input_size: Size of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            num_classes: Number of classes (2: benign, obfuscated)
        """
        super(ObfuscationNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkModel(BaseModel):
    """Neural Network classifier using PyTorch"""
    
    def __init__(self, hidden_layers: List[int] = [128, 64, 32],
                 dropout: float = 0.3, learning_rate: float = 0.001,
                 batch_size: int = 32, epochs: int = 50,
                 early_stopping_patience: int = 10, device: str = 'cpu'):
        """
        Args:
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            device: 'cpu' or 'cuda'
        """
        super().__init__("NeuralNetwork")
        
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.input_size = None
    
    def _create_model(self, input_size: int, num_classes: int = 2):
        """Create neural network model"""
        self.input_size = input_size
        self.model = ObfuscationNN(
            input_size=input_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            num_classes=num_classes
        ).to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train Neural Network model
        """
        logger.info(f"Training {self.model_name}...")
        
        # Create model if not exists
        if self.model is None:
            num_classes = len(np.unique(y_train))
            self._create_model(X_train.shape[1], num_classes)
        
        # Create datasets
        train_dataset = FeatureDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = FeatureDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_acc = val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Acc: {train_acc:.4f}")
        
        self.is_trained = True
        logger.info(f"Training completed. Final train accuracy: {train_acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_size = checkpoint['input_size']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout = checkpoint['dropout']
        
        num_classes = 2  # Default
        self._create_model(self.input_size, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")

