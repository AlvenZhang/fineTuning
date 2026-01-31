# Fine-tune Qwen3-0.6B for Emotion Classification

## Overview
Create a single Python file to fine-tune the Qwen/Qwen3-0.6B model for emotion classification task using the Emotion dataset from Hugging Face.

## Dataset Information

### Emotion Dataset (Hugging Face)
- **Source**: Hugging Face Datasets
- **Emotion Categories**: 6 basic emotions (anger, fear, joy, love, sadness, surprise)
- **Dataset Size**: 16,000+ examples
- **Split**: Train (16000), Validation (2000), Test (2000)
- **Language**: English
- **Use Case**: Well-structured for benchmarking emotion classification models
- **Access**: Available through `datasets.load_dataset("emotion")`

## Implementation Steps

### 1. Create Fine-tuning Script
- Create a new file `fine_tune_qwen_emotion.py` in the project root
- Import necessary libraries: transformers, datasets, torch, peft, accelerate

### 2. Model and Tokenizer Setup
- Load Qwen3-0.6B model and tokenizer from Hugging Face
- Configure tokenizer for classification task
- Set up PEFT (Parameter-Efficient Fine-Tuning) for efficient training
- Use LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning

### 3. Dataset Preparation
- Load Emotion dataset from Hugging Face Datasets
- Preprocess dataset for Qwen model input format
- Tokenize text data with appropriate padding and truncation
- Map emotion labels to model-compatible format

### 4. Training Configuration
- Define training arguments:
  - Learning rate: 2e-4
  - Batch size: 8 (adjust based on available memory)
  - Epochs: 3
  - Warmup steps: 500
  - Weight decay: 0.01
- Set up optimizer (AdamW) and learning rate scheduler
- Configure evaluation metrics: accuracy, precision, recall, F1-score

### 5. Training Process
- Initialize Trainer with model, dataset, and training arguments
- Run fine-tuning process
- Monitor training progress and evaluation metrics
- Handle checkpointing and early stopping if needed

### 6. Model Evaluation
- Evaluate fine-tuned model on test set
- Calculate comprehensive evaluation metrics
- Generate classification report with per-class performance
- Analyze model predictions and error patterns

### 7. Model Saving
- Save fine-tuned model and tokenizer
- Provide instructions for model usage
- Include example inference code

## Key Features
- Uses PEFT for memory-efficient fine-tuning
- Supports both GPU and CPU training
- Includes comprehensive evaluation metrics
- Provides example usage of the fine-tuned model
- Handles all preprocessing and postprocessing steps automatically
- Includes detailed logging and progress tracking

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- Datasets 2.0+
- PEFT 0.10+
- Accelerate 0.20+
- Scikit-learn (for evaluation metrics)

## Expected Output
- Fine-tuned Qwen3-0.6B model for emotion classification
- Evaluation metrics report
- Example inference script
- Model usage documentation

## Usage Instructions
1. Install required dependencies
2. Run the fine-tuning script
3. Use the saved model for emotion classification tasks
4. Evaluate model performance on custom datasets

## Example Inference
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-qwen-emotion")
model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-qwen-emotion")

def classify_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    return emotion_labels[predictions.item()]

# Example usage
text = "I'm so happy today!"
emotion = classify_emotion(text)
print(f"Emotion: {emotion}")
```
