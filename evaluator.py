import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall
from typing import Dict, List, Tuple
import torch.nn as nn


class ModelEvaluator:
    
    def __init__(self, model: nn.Module, num_classes: int):
        self.model = model
        self.num_classes = num_classes
        
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, List[float], List[float], List[int]]:
        accuracy_metric = Accuracy(task='multiclass', num_classes=self.num_classes)
        precision_metric = Precision(task='multiclass', num_classes=self.num_classes, average=None)
        recall_metric = Recall(task='multiclass', num_classes=self.num_classes, average=None)
        
        self.model.eval()
        predicted = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                output = self.model(inputs)
                cat = torch.argmax(output, dim=-1)
                predicted.extend(cat.tolist())
                
                accuracy_metric(cat, labels)
                precision_metric(cat, labels)
                recall_metric(cat, labels)
        
        accuracy = accuracy_metric.compute().item()
        precision = precision_metric.compute().tolist()
        recall = recall_metric.compute().tolist()
        
        return accuracy, precision, recall, predicted
    
    def print_results(self, accuracy: float, precision: List[float], recall: List[float]) -> None:
        """Print evaluation results."""
        print('Accuracy:', accuracy)
        print('Precision (per class):', precision)
        print('Recall (per class):', recall)
