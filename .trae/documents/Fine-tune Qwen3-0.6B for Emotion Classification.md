# Split Fine-tuning Script into Modular Components

## Overview
Split the large `fine_tune_qwen_emotion.py` file into multiple smaller, focused files within the existing `/Users/xifeng/python_file/fineTuning/emotion_classification/` directory for better organization and maintainability.

## Implementation Steps

### 1. Create Directory Structure
- **Base Directory**: `/Users/xifeng/python_file/fineTuning/emotion_classification/`
- **Subdirectories**:
  - `config/` - Configuration files
  - `data/` - Dataset processing
  - `model/` - Model setup and loading
  - `train/` - Training logic
  - `evaluate/` - Evaluation functions
  - `utils/` - Utility functions

### 2. Split Components

#### 2.1 Configuration Files
- **File**: `config/config.py`
- **Content**: Command-line arguments, training parameters, constants

#### 2.2 Dataset Processing
- **File**: `data/dataset.py`
- **Content**: Dataset loading, processing, and tokenization

#### 2.3 Model Setup
- **File**: `model/model.py`
- **Content**: Model loading, PEFT configuration, model preparation

#### 2.4 Training Logic
- **File**: `train/trainer.py`
- **Content**: Training arguments, Trainer setup, training execution

#### 2.5 Evaluation Functions
- **File**: `evaluate/evaluator.py`
- **Content**: Model evaluation, metrics calculation, results saving

#### 2.6 Utility Functions
- **File**: `utils/utils.py`
- **Content**: Helper functions, logging setup, common utilities

#### 2.7 Main Script
- **File**: `main.py`
- **Content**: Main execution flow, component integration

#### 2.8 Example Usage
- **File**: `inference_example.py`
- **Content**: Model inference example

### 3. Implementation Details

#### 3.1 Configuration File (`config/config.py`)
- Command-line argument parsing
- Default parameter values
- Path constants
- Training hyperparameters

#### 3.2 Dataset Processing (`data/dataset.py`)
- Local dataset loading from `/Users/xifeng/python_file/fineTuning/emotion_dataset_cache/`
- Tokenization function
- Dataset preprocessing
- Label mapping extraction

#### 3.3 Model Setup (`model/model.py`)
- Qwen3-0.6B model loading from local path
- PEFT configuration with LoRA
- Model preparation for classification
- Trainable parameter calculation

#### 3.4 Training Logic (`train/trainer.py`)
- TrainingArguments setup
- Trainer initialization
- Training execution
- Checkpoint management

#### 3.5 Evaluation Functions (`evaluate/evaluator.py`)
- Test set evaluation
- Metrics calculation
- Classification report generation
- Results saving to file

#### 3.6 Utility Functions (`utils/utils.py`)
- Logging setup
- Directory creation
- Common helper functions
- Error handling utilities

#### 3.7 Main Script (`main.py`)
- Component integration
- Execution flow control
- Training orchestration
- Post-training tasks

### 4. Key Features
- **Modular Design**: Each file has a single responsibility
- **Easy Maintenance**: Changes to one component don't affect others
- **Reusability**: Components can be reused across projects
- **Clear Organization**: Logical file structure
- **Comprehensive Documentation**: Each module is well-documented

### 5. Migration Strategy
1. **Create Directory Structure** within emotion_classification
2. **Extract Configuration** to `config/config.py`
3. **Extract Dataset Processing** to `data/dataset.py`
4. **Extract Model Setup** to `model/model.py`
5. **Extract Training Logic** to `train/trainer.py`
6. **Extract Evaluation Functions** to `evaluate/evaluator.py`
7. **Extract Utilities** to `utils/utils.py`
8. **Create Main Script** to integrate all components
9. **Test the New Structure** to ensure functionality

### 6. Expected Outcome
- Multiple small, focused files replacing the single large script
- Same functionality maintained
- Better code organization and readability
- Easier debugging and maintenance
- Clear separation of concerns

### 7. Usage Instructions
1. **Run Main Script**: `python main.py`
2. **Customize Configuration**: Modify `config.py` or use command-line arguments
3. **Extend Functionality**: Add new features to specific modules
4. **Debug Components**: Test individual modules independently