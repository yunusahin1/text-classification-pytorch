import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List


class TextDataLoader:
    
    def __init__(self, words_file: str = "words.json", 
                 text_file: str = "text.json", 
                 labels_file: str = "labels.npy"):
        self.words_file = words_file
        self.text_file = text_file
        self.labels_file = labels_file
        self.word2idx = {}
        self.idx2word = {}
        
    def load_data(self) -> Tuple[List, np.ndarray, Dict, Dict]:
        with open(self.words_file, 'r') as f1:
            words = json.load(f1)
        with open(self.text_file, 'r') as f2:
            text = json.load(f2)
        labels = np.load(self.labels_file)
        
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for idx, word in enumerate(words)}
        
        return text, labels, self.word2idx, self.idx2word
    
    def preprocess_text(self, text: List[List[str]], word2idx: Dict, seq_len: int = 50) -> np.ndarray:
        for i, sentence in enumerate(text):
            text[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
        
        return self.pad_input(text, seq_len)
    
    @staticmethod
    def pad_input(sentences: List[List[int]], seq_len: int) -> np.ndarray:
        features = np.zeros((len(sentences), seq_len), dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features
    
    def create_data_loaders(self, text: np.ndarray, labels: np.ndarray, 
                           batch_size: int = 400, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        train_text, test_text, train_label, test_label = train_test_split(
            text, labels, test_size=test_size, random_state=random_state
        )
        
        train_data = TensorDataset(torch.from_numpy(train_text), torch.from_numpy(train_label).long())
        test_data = TensorDataset(torch.from_numpy(test_text), torch.from_numpy(test_label).long())
        
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        
        return train_loader, test_loader
