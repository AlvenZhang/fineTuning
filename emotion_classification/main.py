#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Qwen3-0.6B emotion classification fine-tuning.
"""

import logging
import os
from transformers import set_seed

# Import modules
from config.config import parse_args, Config
from data.dataset import load_and_process_dataset, get_emotion_labels
from model.model import load_model, load_tokenizer
from train.trainer import create_training_args, create_trainer, train_model, save_model
from evaluate.evaluator import evaluate_model, create_model_card
from utils.utils import setup_logging, print_header, print_footer


def main() -> None:
    """
    Main function for fine-tuning Qwen3-0.6B on emotion classification task.
    """
    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    
    print_header("Qwen3-0.6B Emotion Classification Fine-tuning")
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Create configuration
        config = Config(args)
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Model: {config.model_name_or_path}")
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Output directory: {config.output_dir}")
        
        # Load tokenizer
        tokenizer = load_tokenizer(config.model_name_or_path)
        
        # Load and process dataset
        tokenized_dataset, label_mapping, num_labels = load_and_process_dataset(
            config.dataset_name,
            config.dataset_cache_dir,
            tokenizer
        )
        
        # Log dataset info
        logger.info("\nDataset information:")
        logger.info(f"Number of classes: {num_labels}")
        logger.info(f"Class labels: {[label_mapping(i) for i in range(num_labels)]}")
        
        # Load model
        model = load_model(
            config.model_name_or_path,
            num_labels,
            config.use_peft
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Create training arguments
        training_args = create_training_args(config)
        
        # Create trainer
        trainer = create_trainer(
            model,
            training_args,
            tokenized_dataset,
            tokenizer
        )
        
        logger.info("\nAll components initialized successfully!")
        logger.info("Training is not executed (commented out).")
        logger.info("Uncomment the training section to run training.")
        
        # Run training
        training_result = train_model(trainer)
        
        # Evaluate on test set
        eval_result = evaluate_model(
            model,
            tokenizer,
            tokenized_dataset["test"],
            training_args,
            label_mapping,
            config.output_dir
        )
        
        # Save model
        save_model(trainer, config.output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(config.output_dir)
        logger.info("Tokenizer saved successfully")
        
        # Create model card
        class_names = [label_mapping(i) for i in range(num_labels)]
        create_model_card(
            config.output_dir,
            config,
            eval_result,
            class_names
        )
        
        # Create inference example
        create_inference_example(config.output_dir)
        
        print_footer("Training completed successfully!")
        logger.info(f"Model saved to: {config.output_dir}")
        logger.info("Use inference_example.py to test the model.")
        
        
        print_footer("Script completed successfully!")
        logger.info("All components have been initialized but training is not executed.")
        logger.info("Uncomment the training section in main.py to run training.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print_footer("Script failed with error")


def create_inference_example(output_dir: str) -> None:
    """
    Create inference example script.
    
    Args:
        output_dir: Directory to save the example
    """
    logger = logging.getLogger(__name__)
    
    example_code = f'''#!/usr/bin/env python3
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
    example_path = os.path.join(output_dir, "inference_example.py")
    with open(example_path, "w", encoding="utf-8") as f:
        f.write(example_code)
    
    logger.info(f"Inference example script created: {example_path}")


if __name__ == "__main__":
    main()
