#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to demonstrate emotion dataset structure and usage.
"""

from datasets import load_dataset


def main():
    """
    Main function to demonstrate emotion dataset usage.
    """
    # Load the emotion dataset
    print("Loading emotion dataset...")
    dataset = load_dataset("emotion")
    
    # Print dataset structure
    print("\nDataset structure:")
    print(dataset)
    
    # Print dataset splits
    print("\nDataset splits:")
    for split in dataset.keys():
        print(f"{split}: {len(dataset[split])} examples")
    
    # Print feature information
    print("\nFeature information:")
    print(dataset["train"].features)
    
    # Get label mapping
    label_mapping = dataset["train"].features["label"].int2str
    print("\nLabel mapping:")
    for i in range(dataset["train"].features["label"].num_classes):
        print(f"{i}: {label_mapping(i)}")
    
    # Read and display a single example
    print("\nSingle example from train set:")
    example = dataset["train"][0]
    print(f"Text: {example['text']}")
    print(f"Label (integer): {example['label']}")
    print(f"Label (string): {label_mapping(example['label'])}")
    
    # Read and display multiple examples
    print("\nFirst 5 examples from train set:")
    for i in range(5):
        example = dataset["train"][i]
        print(f"{i+1}. Text: {example['text']}")
        print(f"   Emotion: {label_mapping(example['label'])}")
        print()
    
    # Accessing validation and test sets
    print("\nExample from validation set:")
    val_example = dataset["validation"][0]
    print(f"Text: {val_example['text']}")
    print(f"Emotion: {label_mapping(val_example['label'])}")
    
    print("\nExample from test set:")
    test_example = dataset["test"][0]
    print(f"Text: {test_example['text']}")
    print(f"Emotion: {label_mapping(test_example['label'])}")


if __name__ == "__main__":
    main()
