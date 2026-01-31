#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for Qwen3-0.6B emotion classification fine-tuning.
"""

import argparse
import os


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
    
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default="/Users/xifeng/python_file/fineTuning/emotion_dataset_cache/",
        help="Path to local dataset cache"
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
        default=2,
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
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory for logging"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    return parser.parse_args()


class Config:
    """Configuration class with default values."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from arguments."""
        # Model configuration
        self.model_name_or_path = args.model_name_or_path
        
        # Dataset configuration
        self.dataset_name = args.dataset_name
        self.dataset_cache_dir = args.dataset_cache_dir
        
        # Training configuration
        self.output_dir = args.output_dir
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_train_epochs = args.num_train_epochs
        self.seed = args.seed
        self.use_peft = args.use_peft
        self.logging_dir = args.logging_dir
        self.warmup_steps = args.warmup_steps
        self.weight_decay = args.weight_decay
        
        # Paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.output_dir = os.path.join(self.project_root, self.output_dir)
        self.logging_dir = os.path.join(self.project_root, self.logging_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "dataset_name": self.dataset_name,
            "dataset_cache_dir": self.dataset_cache_dir,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "seed": self.seed,
            "use_peft": self.use_peft,
            "logging_dir": self.logging_dir,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay
        }
