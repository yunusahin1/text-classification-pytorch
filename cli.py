import argparse
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
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

def train_command(args):
    config = load_config(args.config if hasattr(args, 'config') else "config.yaml")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
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
    
    trainer = ModelTrainer(model, learning_rate=config['training']['learning_rate'])
    trainer.train(train_loader, epochs=config['training']['epochs'])
    
    evaluator = ModelEvaluator(model, target_size)
    accuracy, precision, recall, predictions = evaluator.evaluate(test_loader)
    
    save_path = args.output or config['paths']['model_save_path']
    trainer.save_model(save_path)
    
    print(f"Model trained and saved to {save_path}")
    print(f"Final accuracy: {accuracy:.4f}")

def predict_command(args):
    config = load_config(args.config if hasattr(args, 'config') else "config.yaml")
    
    data_loader = TextDataLoader(
        words_file=config['data']['words_file'],
        text_file=config['data']['text_file'],
        labels_file=config['data']['labels_file']
    )
    _, labels, word2idx, idx2word = data_loader.load_data()
    
    vocab_size = len(word2idx) + 1
    target_size = len(np.unique(labels))
    model = TicketClassifier(vocab_size, config['model']['embedding_dim'], target_size)
    
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    if args.text:
        text_samples = [args.text]
    elif args.file:
        with open(args.file, 'r') as f:
            text_samples = [line.strip() for line in f if line.strip()]
    else:
        print("Please provide either --text or --file argument")
        return
    
    for i, text in enumerate(text_samples):
        tokens = text.lower().split()
        indices = [word2idx.get(word, 0) for word in tokens]
        
        padded = data_loader.pad_input([indices], config['data']['sequence_length'])
        input_tensor = torch.from_numpy(padded)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=-1).item()
        
        print(f"Text {i+1}: {text}")
        print(f"Prediction: {prediction}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Text Classification CLI")
    parser.add_argument('--config', type=str, default="config.yaml", 
                       help='Path to configuration YAML file')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--output', type=str, help='Output path for saved model')
    train_parser.add_argument('--config', type=str, default="config.yaml", 
                             help='Path to configuration YAML file')
    
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--text', type=str, help='Text to classify')
    predict_parser.add_argument('--file', type=str, help='File containing texts to classify')
    predict_parser.add_argument('--config', type=str, default="config.yaml", 
                               help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
