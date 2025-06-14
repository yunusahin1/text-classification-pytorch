import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class ModelTrainer:
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.05):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train(self, train_loader: DataLoader, epochs: int = 3) -> None:
        self.model.train()
        
        for epoch in range(epochs):
            running_loss, num_processed = 0, 0
            
            for inputs, labels in train_loader:
                self.model.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                num_processed += len(inputs)
            
            avg_loss = running_loss / num_processed
            print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.6f}")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        self.model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")
    

