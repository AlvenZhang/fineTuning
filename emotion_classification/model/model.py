#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model setup module for Qwen3-0.6B emotion classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)


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
                    r=4,
                    lora_alpha=16,
                    target_modules=["o_proj"],
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
