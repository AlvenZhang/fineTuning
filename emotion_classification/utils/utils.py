#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Qwen3-0.6B emotion classification.
"""

import os
import sys
import logging
import json
from typing import Any, Dict, Optional


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: File path to save data
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory:
        create_directory(directory)
    
    # Save data to JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved data to: {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: File path to load data from
    
    Returns:
        Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logging.info(f"Loaded data from: {file_path}")
    return data


def get_project_root() -> str:
    """
    Get project root directory.
    
    Returns:
        Project root directory path
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def validate_path(path: str, must_exist: bool = False) -> str:
    """
    Validate path and expand user directory.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
    
    Returns:
        Validated path
    """
    # Expand user directory
    path = os.path.expanduser(path)
    
    # Check if path exists if required
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return path


def print_header(message: str, char: str = "=") -> None:
    """
    Print header with message.
    
    Args:
        message: Header message
        char: Character to use for header
    """
    length = len(message)
    border = char * (length + 4)
    
    print(f"\n{border}")
    print(f"{char} {message} {char}")
    print(f"{border}\n")


def print_footer(message: str, char: str = "=") -> None:
    """
    Print footer with message.
    
    Args:
        message: Footer message
        char: Character to use for footer
    """
    length = len(message)
    border = char * (length + 4)
    
    print(f"\n{border}")
    print(f"{char} {message} {char}")
    print(f"{border}\n")
