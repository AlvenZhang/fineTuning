#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model setup module for Qwen3-0.6B emotion classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
import torch

logger = logging.getLogger(__name__)

# Get device
def get_device():
    """Get the appropriate device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found, using CPU")
    return device

device = get_device()


def load_model(
    model_name_or_path: str,
    num_labels: int,
    use_peft: bool = True
):
    """
    Load Qwen3-0.6B model and configure for emotion classification.
    
    Args:
        model_name_or_path: Path to pre-trained model
        num_labels: Number of emotion classes
        use_peft: Whether to use PEFT for efficient fine-tuning
    
    Returns:
        Loaded and configured model
    """
    logger.info(f"Loading model from: {model_name_or_path}")
    
    try:
        # Load model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        logger.info(f"Number of labels: {num_labels}")
        
        # Configure PEFT if enabled
        if use_peft:
            logger.info("Configuring PEFT for efficient fine-tuning")
            
            try:
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
                logger.info("Trainable parameters:")
                model.print_trainable_parameters()
                
            except Exception as e:
                logger.error(f"PEFT configuration failed: {e}")
                logger.warning("Continuing without PEFT")
        
        # Move model to device
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


def load_tokenizer(model_name_or_path: str):
    """
    Load tokenizer for Qwen3-0.6B model.
    
    Args:
        model_name_or_path: Path to pre-trained model
    
    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from: {model_name_or_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        logger.info(f"Tokenizer loaded successfully: {type(tokenizer).__name__}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {e}")
        raise
