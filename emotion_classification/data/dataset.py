#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset processing module for Qwen3-0.6B emotion classification.
"""

from typing import Dict, Tuple, Callable
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


def load_and_process_dataset(
    dataset_name: str,
    dataset_cache_dir: str,
    tokenizer: AutoTokenizer
) -> Tuple[DatasetDict, Callable[[int], str], int]:
    """
    Load and process the emotion dataset from local cache.
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_cache_dir: Path to local dataset cache
        tokenizer: Tokenizer to use for preprocessing
    
    Returns:
        Tuple containing:
        - Processed dataset
        - Label mapping function
        - Number of labels
    """
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Using local cache: {dataset_cache_dir}")
    
    # Load the dataset from local cache
    dataset = load_dataset(
        dataset_name,
        cache_dir=dataset_cache_dir
    )
    
    # Get label mapping and number of classes
    label_mapping = dataset["train"].features["label"].int2str
    num_labels = dataset["train"].features["label"].num_classes
    
    logger.info(f"Label mapping: {label_mapping}")
    logger.info(f"Number of classes: {num_labels}")
    
    # Tokenize the dataset
    def tokenize_function(examples: Dict[str, list]) -> Dict[str, list]:
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
    
    # Log dataset statistics
    logger.info("Dataset processing completed successfully")
    logger.info(f"Train set size: {len(tokenized_dataset['train'])}")
    logger.info(f"Validation set size: {len(tokenized_dataset['validation'])}")
    logger.info(f"Test set size: {len(tokenized_dataset['test'])}")
    
    # Display sample data
    sample = tokenized_dataset["train"][0]
    logger.info("\nSample data:")
    logger.info(f"Sample input IDs shape: {len(sample['input_ids'])}")
    logger.info(f"Sample attention mask shape: {len(sample['attention_mask'])}")
    logger.info(f"Sample label: {sample['labels']} ({label_mapping(sample['labels'])})")
    
    return tokenized_dataset, label_mapping, num_labels


def get_emotion_labels() -> list:
    """
    Get list of emotion labels.
    
    Returns:
        List of emotion labels in order
    """
    return ["anger", "fear", "joy", "love", "sadness", "surprise"]
