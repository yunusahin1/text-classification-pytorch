# Text Classification with PyTorch

A production-ready text classification system using PyTorch with CNN architecture.

## Features

- CNN-based text classification
- Configurable hyperparameters
- Model saving/loading
- Command-line interface
- Comprehensive logging
- Error handling and validation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python cli.py train --epochs 5 --batch-size 256 --learning-rate 0.01
```

### Making Predictions

```bash
# Single text prediction
python cli.py predict --model text_classifier_model.pth --text "Your text here"

# Batch prediction from file
python cli.py predict --model text_classifier_model.pth --file input_texts.txt
```

### Programmatic Usage

```python
from text_classifier import TextClassifier, Config

# Initialize
config = Config()
classifier = TextClassifier(config)

# Train
text, labels = classifier.load_data()
train_loader, test_loader = classifier.prepare_data_loaders(text, labels)
classifier.initialize_model()
classifier.train(train_loader)

# Evaluate
metrics = classifier.evaluate(test_loader)

# Save model
classifier.save_model("my_model.pth")

# Load and predict
classifier.load_model("my_model.pth")
predictions = classifier.predict(["Sample text to classify"])
```

## Configuration

Modify the `Config` class in `text_classifier.py` to adjust:
- Batch size
- Learning rate
- Number of epochs
- Embedding dimensions
- Sequence length
- File paths

## Model Architecture

The model uses:
- Embedding layer for word representations
- 1D Convolutional layer with ReLU activation
- Global average pooling
- Fully connected output layer
- Dropout for regularization

## Requirements

- Python 3.7+
- PyTorch 1.12+
- See `requirements.txt` for complete list
