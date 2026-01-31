#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen3-0.6B for Emotion Classification

This script implements the fine-tuning process for Qwen/Qwen3-0.6B model
on the Emotion dataset for emotion classification task.
"""

# 2.1 Imports
import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 1. Script Structure
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B for Emotion Classification")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/Users/xifeng/python_file/fineTuning/models/Qwen3-0.6B/",
        help="Path to pre-trained model or model identifier from Hugging Face Hub"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="emotion",
        help="Name of the dataset to use"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine_tuned_qwen_emotion",
        help="Directory to save the fine-tuned model"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--use_peft",
        action="store_true",
        default=True,
        help="Whether to use PEFT for efficient fine-tuning"
    )
    
    return parser.parse_args()

# 2.3 Dataset Processing
def load_and_process_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer
) -> Tuple[DatasetDict, callable, int]:
    """
    Load and process the emotion dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for preprocessing
    
    Returns:
        Tuple containing the processed dataset, label mapping function, and number of labels
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load the dataset from local cache
    dataset = load_dataset(
        dataset_name,
        cache_dir="/Users/xifeng/python_file/fineTuning/emotion_dataset_cache/"
    )
    
    # Get label mapping
    label_mapping = dataset["train"].features["label"].int2str
    num_labels = dataset["train"].features["label"].num_classes
    logger.info(f"Label mapping: {label_mapping}")
    logger.info(f"Number of classes: {num_labels}")
    
    # Tokenize the dataset
    def tokenize_function(examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize input text."""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )
    
    # Rename label column to labels for Trainer compatibility
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    logger.info("Dataset processing completed successfully")
    logger.info(f"Train set size: {len(tokenized_dataset['train'])}")
    logger.info(f"Validation set size: {len(tokenized_dataset['validation'])}")
    logger.info(f"Test set size: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset, label_mapping, num_labels

def main() -> None:
    """Main function for fine-tuning Qwen3-0.6B on emotion classification task."""
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Load and process dataset
    tokenized_dataset, label_mapping, num_labels = load_and_process_dataset(
        args.dataset_name,
        tokenizer
    )
    
    # Log dataset info
    logger.info("\nDataset information:")
    logger.info(f"Number of classes: {num_labels}")
    logger.info(f"Class labels: {[label_mapping(i) for i in range(num_labels)]}")
    
    # Display sample data
    logger.info("\nSample data:")
    sample = tokenized_dataset["train"][0]
    logger.info(f"Sample input IDs shape: {len(sample['input_ids'])}")
    logger.info(f"Sample attention mask shape: {len(sample['attention_mask'])}")
    logger.info(f"Sample label: {sample['labels']} ({label_mapping(sample['labels'])})")
    
    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    
    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=True
    )
    
    # Configure PEFT if enabled
    if args.use_peft:
        logger.info("Configuring PEFT for efficient fine-tuning")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Set up LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
    
    # 2.4 Training Configuration
    logger.info("\nConfiguring training settings...")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )
    
    # 2.4.1 Define evaluation metrics
    def compute_metrics(eval_pred):
        """Compute evaluation metrics."""
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # 2.4.2 Set up data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 2.5 Training Execution (commented out for now)
    logger.info("\nTraining configuration completed!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    
    # 2.6 Evaluation Function
    def evaluate_model(model, tokenizer, test_dataset):
        """Evaluate the model on test dataset."""
        logger.info("\nEvaluating model on test set...")
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        # Generate classification report
        predictions = trainer.predict(test_dataset)
        pred_labels = predictions.predictions.argmax(axis=-1)
        true_labels = predictions.label_ids
        
        report = classification_report(
            true_labels, pred_labels,
            target_names=[label_mapping(i) for i in range(num_labels)]
        )
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
            f.write("\n=== Evaluation Results ===\n")
            f.write(f"Accuracy: {eval_result['eval_accuracy']:.4f}\n")
            f.write(f"Precision: {eval_result['eval_precision']:.4f}\n")
            f.write(f"Recall: {eval_result['eval_recall']:.4f}\n")
            f.write(f"F1-score: {eval_result['eval_f1']:.4f}\n")
            f.write("\n=== Classification Report ===\n")
            f.write(report)
        
        logger.info("Evaluation completed! Results saved to evaluation_results.txt")
        return eval_result
    
    # 2.7 Model Saving Function
    def save_model(model, tokenizer, output_dir):
        """Save the fine-tuned model and tokenizer."""
        logger.info(f"\nSaving model to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if args.use_peft:
            # For PEFT models, save the adapter
            model.save_pretrained(output_dir)
            # Save tokenizer separately
            tokenizer.save_pretrained(output_dir)
        else:
            # For full models, save both together
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved successfully!")
    
    # 2.8 Example Usage
    def create_inference_example(output_dir):
        """Create inference example script."""
        example_code = f'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference example for fine-tuned Qwen3-0.6B emotion classification model.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_emotion(text):
    """Classify emotion in text using fine-tuned model."""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
    model = AutoModelForSequenceClassification.from_pretrained("{output_dir}")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make prediction
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    
    # Map to emotion labels
    emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    return emotion_labels[predictions.item()]

# Example usage
if __name__ == "__main__":
    test_sentences = [
        "I'm so happy today!",
        "I feel scared of the dark.",
        "I'm angry about the situation.",
        "I love spending time with my family.",
        "I'm sad that it's over.",
        "I'm surprised by the news!"
    ]
    
    for sentence in test_sentences:
        emotion = classify_emotion(sentence)
        print(f"Text: {sentence}")
        print(f"Emotion: {emotion}")
        print("-")
'''
        
        # Write example script
        with open(os.path.join(output_dir, "inference_example.py"), "w") as f:
            f.write(example_code)
        
        logger.info("Inference example script created: inference_example.py")
    
    # Create model card
    def create_model_card(output_dir):
        """Create model card with usage instructions."""
        model_card = f"""
# Qwen3-0.6B Fine-tuned for Emotion Classification

## Model Description

This is a fine-tuned version of Qwen/Qwen3-0.6B for emotion classification task.
The model has been trained on the Emotion dataset to classify text into 6 emotion categories.

## Emotion Categories

- anger
- fear
- joy
- love
- sadness
- surprise

## Training Data

- **Dataset**: Emotion dataset
- **Size**: 16,000 training examples, 2,000 validation examples, 2,000 test examples
- **Language**: English

## Training Parameters

- **Learning Rate**: 2e-4
- **Batch Size**: 8
- **Epochs**: 3
- **Warmup Steps**: 500
- **Weight Decay**: 0.01
- **PEFT**: LoRA (r=8, alpha=16)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
model = AutoModelForSequenceClassification.from_pretrained("{output_dir}")

def classify_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    return emotion_labels[predictions.item()]

# Example
text = "I'm so happy today!"
emotion = classify_emotion(text)
print(f"Emotion: {emotion}")
```

## Evaluation Results

See evaluation_results.txt for detailed performance metrics.

## Limitations

- The model is trained on English text only
- Performance may vary on domain-specific text
- The model may not capture complex or mixed emotions

## License

Same as original Qwen3-0.6B model.
"""
        
        # Write model card
        with open(os.path.join(output_dir, "model_card.md"), "w") as f:
            f.write(model_card)
        
        logger.info("Model card created: model_card.md")
    
    logger.info("\nAll components implemented successfully!")
    logger.info("\nImplemented components:")
    logger.info("1. Script structure and imports")
    logger.info("2. Dataset processing (using local cached dataset)")
    logger.info("3. Model loading and PEFT configuration")
    logger.info("4. Training configuration")
    logger.info("5. Evaluation function")
    logger.info("6. Model saving function")
    logger.info("7. Example usage script")
    logger.info("8. Model card generation")
    
    logger.info("\nTo run training, uncomment the training execution section below.")
    
    # Uncomment the following section to run training
    """
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    # Run training
    logger.info("\nStarting training...")
    training_result = trainer.train()
    
    # Evaluate on test set
    evaluate_model(model, tokenizer, tokenized_dataset["test"])
    
    # Save model
    save_model(model, tokenizer, args.output_dir)
    
    # Create inference example
    create_inference_example(args.output_dir)
    
    # Create model card
    create_model_card(args.output_dir)
    
    logger.info("\nTraining completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("Use inference_example.py to test the model.")
    """
    
    logger.info("\nScript completed successfully!")
    logger.info("All components have been implemented but training is not executed (commented out).")
    logger.info("To run training, uncomment the training execution section in the code.")

if __name__ == "__main__":
    main()
