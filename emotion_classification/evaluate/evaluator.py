#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module for Qwen3-0.6B emotion classification.
"""

from typing import Dict, Any, Callable
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report
import os
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    test_dataset: Dataset,
    training_args: TrainingArguments,
    label_mapping: Callable[[int], str],
    output_dir: str
) -> Dict[str, float]:
    """
    Evaluate the model on test dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        training_args: Training arguments
        label_mapping: Label mapping function
        output_dir: Directory to save evaluation results
    
    Returns:
        Evaluation results dictionary
    """
    logger.info("Evaluating model on test set...")
    
    try:
        # Create trainer for evaluation
        from train.trainer import compute_metrics
        from transformers import DataCollatorWithPadding
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
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
        
        # Get class names
        num_labels = len(set(true_labels))
        class_names = [label_mapping(i) for i in range(num_labels)]
        
        report = classification_report(
            true_labels, pred_labels,
            target_names=class_names
        )
        
        # Save evaluation results
        save_evaluation_results(
            eval_result, report, output_dir, class_names
        )
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Test accuracy: {eval_result['eval_accuracy']:.4f}")
        logger.info(f"Test F1-score: {eval_result['eval_f1']:.4f}")
        
        return eval_result
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def save_evaluation_results(
    eval_result: Dict[str, float],
    report: str,
    output_dir: str,
    class_names: list
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        eval_result: Evaluation results dictionary
        report: Classification report
        output_dir: Directory to save results
        class_names: List of class names
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to file
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n=== Evaluation Results ===\n")
        f.write(f"Accuracy: {eval_result['eval_accuracy']:.4f}\n")
        f.write(f"Precision: {eval_result['eval_precision']:.4f}\n")
        f.write(f"Recall: {eval_result['eval_recall']:.4f}\n")
        f.write(f"F1-score: {eval_result['eval_f1']:.4f}\n")
        
        f.write("\n=== Classification Report ===\n")
        f.write(report)
        
        f.write("\n=== Class Names ===\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{i}: {class_name}\n")
    
    logger.info(f"Evaluation results saved to: {results_file}")


def create_model_card(
    output_dir: str,
    config: Any,
    eval_result: Dict[str, float],
    class_names: list
) -> None:
    """
    Create model card with usage instructions.
    
    Args:
        output_dir: Directory to save model card
        config: Configuration object
        eval_result: Evaluation results
        class_names: List of class names
    """
    logger.info("Creating model card...")
    
    model_card = f"""
# Qwen3-0.6B Fine-tuned for Emotion Classification

## Model Description

This is a fine-tuned version of Qwen/Qwen3-0.6B for emotion classification task.
The model has been trained on the Emotion dataset to classify text into {len(class_names)} emotion categories.

## Emotion Categories

{chr(10).join([f"- {class_name}" for class_name in class_names])}

## Training Data

- **Dataset**: Emotion dataset
- **Size**: 16,000 training examples, 2,000 validation examples, 2,000 test examples
- **Language**: English

## Training Parameters

- **Learning Rate**: {config.learning_rate}
- **Batch Size**: {config.batch_size}
- **Epochs**: {config.num_train_epochs}
- **Warmup Steps**: {config.warmup_steps}
- **Weight Decay**: {config.weight_decay}
- **PEFT**: {"LoRA (r=8, alpha=16)" if config.use_peft else "No"}

## Evaluation Results

- **Accuracy**: {eval_result.get('eval_accuracy', 'N/A'):.4f}
- **Precision**: {eval_result.get('eval_precision', 'N/A'):.4f}
- **Recall**: {eval_result.get('eval_recall', 'N/A'):.4f}
- **F1-score**: {eval_result.get('eval_f1', 'N/A'):.4f}

See evaluation_results.txt for detailed classification report.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_emotion(text):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
    model = AutoModelForSequenceClassification.from_pretrained("{output_dir}")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make prediction
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    
    # Map to emotion labels
    emotion_labels = {class_names}
    return emotion_labels[predictions.item()]

# Example usage
text = "I'm so happy today!"
emotion = classify_emotion(text)
print(f"Text: {text}")
print(f"Emotion: {emotion}")
```

## Limitations

- The model is trained on English text only
- Performance may vary on domain-specific text
- The model may not capture complex or mixed emotions

## License

Same as original Qwen3-0.6B model.
"""
    
    # Escape curly braces in class names
    class_names_str = str(class_names).replace("{", "{{").replace("}", "}}")
    model_card = model_card.format(
        output_dir=output_dir,
        class_names=class_names_str
    )
    
    # Save model card
    model_card_file = os.path.join(output_dir, "model_card.md")
    with open(model_card_file, "w", encoding="utf-8") as f:
        f.write(model_card)
    
    logger.info(f"Model card created: {model_card_file}")
