import numpy as np
import yaml
import sys
from typing import Dict, Any

from data_loader import TextDataLoader
from model import TicketClassifier
from trainer import ModelTrainer
from evaluator import ModelEvaluator


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default values.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

def get_default_config() -> Dict[str, Any]:
    return {
        'data': {
            'words_file': 'words.json',
            'text_file': 'text.json',
            'labels_file': 'labels.npy',
            'sequence_length': 50
        },
        'model': {
            'embedding_dim': 64
        },
        'training': {
            'batch_size': 400,
            'learning_rate': 0.05,
            'epochs': 3,
            'test_size': 0.2,
            'random_state': 42
        },
        'paths': {
            'model_save_path': 'ticket_classifier_model.pth'
        }
    }


def main():
    config = load_config()
    
    data_loader = TextDataLoader(
        words_file=config['data']['words_file'],
        text_file=config['data']['text_file'],
        labels_file=config['data']['labels_file']
    )
    
    print("Loading data...")
    text, labels, word2idx, idx2word = data_loader.load_data()
    text = data_loader.preprocess_text(text, word2idx, config['data']['sequence_length'])
    
    train_loader, test_loader = data_loader.create_data_loaders(
        text, labels, 
        batch_size=config['training']['batch_size'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    
    vocab_size = len(word2idx) + 1
    target_size = len(np.unique(labels))
    
    model = TicketClassifier(vocab_size, config['model']['embedding_dim'], target_size)
    print(f"Model initialized with vocab_size={vocab_size}, target_size={target_size}")
    
    trainer = ModelTrainer(model, learning_rate=config['training']['learning_rate'])
    print("Starting training...")
    trainer.train(train_loader, epochs=config['training']['epochs'])
    
    trainer.save_model(config['paths']['model_save_path'])
    
    evaluator = ModelEvaluator(model, target_size)
    print("Evaluating model...")
    accuracy, precision, recall, predictions = evaluator.evaluate(test_loader)
    evaluator.print_results(accuracy, precision, recall)


if __name__ == "__main__":
    main()


