#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for Qwen3-0.6B emotion classification.
"""

from typing import Dict, Any, Optional
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


def create_training_args(config: Any) -> TrainingArguments:
    """
    Create training arguments from configuration.
    
    Args:
        config: Configuration object with training parameters
    
    Returns:
        TrainingArguments object
    """
    logger.info("Creating training arguments...")
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=config.logging_dir,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        seed=config.seed
    )
    
    logger.info("Training arguments created successfully")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    
    return training_args


def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Tuple of (logits, labels)
    
    Returns:
        Dictionary of metrics
    """
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


def create_trainer(
    model: Any,
    training_args: TrainingArguments,
    tokenized_dataset: DatasetDict,
    tokenizer: Any
) -> Trainer:
    """
    Create Trainer instance for model training.
    
    Args:
        model: Model to train
        training_args: Training arguments
        tokenized_dataset: Tokenized dataset
        tokenizer: Tokenizer for data collation
    
    Returns:
        Trainer instance
    """
    logger.info("Creating Trainer instance...")
    
    # 验证并再次确保 tokenizer 的 pad_token 设置
    logger.info(f"Tokenizer pad_token in create_trainer: {tokenizer.pad_token}")
    logger.info(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    # 显式再次设置 pad_token（针对 Qwen3 模型的特殊处理）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Re-set pad_token to eos_token: {tokenizer.pad_token}")
        logger.info(f"Re-set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
    
    # 显式设置 DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=512
    )
    
    logger.info("DataCollatorWithPadding created successfully")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    logger.info("Trainer created successfully")
    return trainer


def train_model(trainer: Trainer) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        trainer: Trainer instance
    
    Returns:
        Training result dictionary
    """
    logger.info("Starting training...")
    
    try:
        training_result = trainer.train()
        
        logger.info("Training completed successfully")
        # logger.info(f"Training time: {training_result.training_time:.2f} seconds")
        logger.info(f"Training loss: {training_result.training_loss:.4f}")
        
        
        return training_result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def save_model(trainer: Trainer, output_dir: str) -> None:
    """
    Save the trained model.
    
    Args:
        trainer: Trainer instance
        output_dir: Directory to save the model
    """
    logger.info(f"Saving model to: {output_dir}")
    
    try:
        trainer.save_model(output_dir)
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Model saving failed: {e}")
        raise
