#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference example for fine-tuned Qwen3-0.6B emotion classification model.
This script reads test data from the dataset and calculates accuracy.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm

def classify_batch(texts, tokenizer, model, batch_size=8):
    """Classify emotions in multiple texts using batch processing."""
    predictions = []
    
    # Calculate total batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            batch_predictions = outputs.logits.argmax(dim=-1)
        
        predictions.extend(batch_predictions.tolist())
    
    return predictions

def load_test_dataset(dataset_name="emotion", cache_dir="./dataset_cache"):
    """Load test dataset."""
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset["test"]

def main():
    """Main function to run inference on test dataset and calculate accuracy."""
    # Model and tokenizer paths
    model_path = "/Users/xifeng/python_file/fineTuning/models/Qwen3-0.6B/"
    
    # Check for MPS availability
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Move model to device
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset()
    
    # Get label mapping
    label_mapping = test_dataset.features["label"].int2str
    num_labels = test_dataset.features["label"].num_classes
    emotion_labels = [label_mapping(i) for i in range(num_labels)]
    
    print(f"\nTest dataset loaded successfully")
    print(f"Number of test examples: {len(test_dataset)}")
    print(f"Emotion labels: {emotion_labels}")
    
    # Run inference on test dataset
    print("\nRunning inference on test dataset...")
    
    # Process first 100 examples for demonstration
    # max_examples = min(100, len(test_dataset))
    max_examples = len(test_dataset)
    test_examples = test_dataset.select(range(max_examples))
    
    # Extract texts and true labels
    texts = test_examples["text"]
    true_labels = test_examples["label"]
    
    # Run batch inference
    batch_size = 64
    print(f"Processing {max_examples} examples with batch size {batch_size}...")
    predictions = classify_batch(texts, tokenizer, model, batch_size)
    
    print(f"Processed all {max_examples} examples")
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n=== Inference Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tested on {max_examples} examples")
    
    # Display some sample predictions
    print(f"\n=== Sample Predictions ===")
    sample_size = min(5, max_examples)
    
    for i in range(sample_size):
        example = test_dataset[i]
        text = example["text"]
        true_label = label_mapping(true_labels[i])
        pred_label = label_mapping(predictions[i])
        
        print(f"\nText: {text}")
        print(f"True emotion: {true_label}")
        print(f"Predicted emotion: {pred_label}")
        print(f"Correct: {true_label == pred_label}")

if __name__ == "__main__":
    main()
