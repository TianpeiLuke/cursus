"""
Framework pattern detection for step type-aware validation.

This module provides pattern detection capabilities for different ML frameworks
used in SageMaker steps, enabling framework-specific validation.
"""

import re
from typing import Dict, List, Optional, Set
from pathlib import Path


def detect_training_patterns(script_content: str) -> Dict[str, List[str]]:
    """
    Detect training-specific patterns in script content.
    
    Args:
        script_content: Content of the training script
        
    Returns:
        Dictionary containing detected training patterns
    """
    patterns = {
        "training_loop_patterns": [],
        "model_saving_patterns": [],
        "hyperparameter_loading_patterns": [],
        "evaluation_patterns": []
    }
    
    # Training loop patterns
    training_patterns = [
        r'\.fit\(',
        r'\.train\(',
        r'xgb\.train\(',
        r'model\.train\(',
        r'for.*epoch',
        r'training_step\(',
        r'train_model\(',
        r'\.learn\('
    ]
    
    # Model saving patterns
    saving_patterns = [
        r'\.save_model\(',
        r'\.save\(',
        r'pickle\.dump\(',
        r'torch\.save\(',
        r'/opt/ml/model',
        r'model\.save_pretrained\(',
        r'joblib\.dump\('
    ]
    
    # Hyperparameter loading patterns
    hyperparam_patterns = [
        r'json\.load\(',
        r'hyperparameters\.json',
        r'/opt/ml/input/data/config',
        r'load.*config',
        r'argparse\.ArgumentParser\(',
        r'os\.environ\.get\('
    ]
    
    # Evaluation patterns
    evaluation_patterns = [
        r'\.evaluate\(',
        r'\.score\(',
        r'accuracy_score\(',
        r'classification_report\(',
        r'confusion_matrix\(',
        r'/opt/ml/output'
    ]
    
    # Search for patterns in script content
    patterns["training_loop_patterns"] = _find_patterns(script_content, training_patterns)
    patterns["model_saving_patterns"] = _find_patterns(script_content, saving_patterns)
    patterns["hyperparameter_loading_patterns"] = _find_patterns(script_content, hyperparam_patterns)
    patterns["evaluation_patterns"] = _find_patterns(script_content, evaluation_patterns)
    
    return patterns


def detect_xgboost_patterns(script_content: str) -> Dict[str, List[str]]:
    """
    Detect XGBoost-specific patterns in script content.
    
    Args:
        script_content: Content of the script
        
    Returns:
        Dictionary containing detected XGBoost patterns
    """
    patterns = {
        "xgboost_imports": [],
        "dmatrix_patterns": [],
        "xgboost_training": [],
        "xgboost_evaluation": [],
        "xgboost_model_saving": []
    }
    
    # XGBoost import patterns
    import_patterns = [
        r'import xgboost',
        r'from xgboost import',
        r'import xgb',
        r'from xgb import'
    ]
    
    # DMatrix patterns
    dmatrix_patterns = [
        r'xgb\.DMatrix',
        r'xgboost\.DMatrix',
        r'DMatrix\('
    ]
    
    # XGBoost training patterns
    training_patterns = [
        r'xgb\.train\(',
        r'xgboost\.train\(',
        r'xgb\.XGBClassifier\(',
        r'xgb\.XGBRegressor\(',
        r'XGBClassifier\(',
        r'XGBRegressor\('
    ]
    
    # XGBoost evaluation patterns
    evaluation_patterns = [
        r'\.predict\(',
        r'\.get_fscore\(',
        r'\.get_score\(',
        r'xgb\.cv\(',
        r'xgboost\.cv\('
    ]
    
    # XGBoost model saving patterns
    saving_patterns = [
        r'\.save_model\(',
        r'booster\.save_model\(',
        r'model\.save_model\('
    ]
    
    # Search for patterns
    patterns["xgboost_imports"] = _find_patterns(script_content, import_patterns)
    patterns["dmatrix_patterns"] = _find_patterns(script_content, dmatrix_patterns)
    patterns["xgboost_training"] = _find_patterns(script_content, training_patterns)
    patterns["xgboost_evaluation"] = _find_patterns(script_content, evaluation_patterns)
    patterns["xgboost_model_saving"] = _find_patterns(script_content, saving_patterns)
    
    return patterns


def detect_pytorch_patterns(script_content: str) -> Dict[str, List[str]]:
    """
    Detect PyTorch-specific patterns in script content.
    
    Args:
        script_content: Content of the script
        
    Returns:
        Dictionary containing detected PyTorch patterns
    """
    patterns = {
        "pytorch_imports": [],
        "model_definition": [],
        "training_loop": [],
        "loss_computation": [],
        "optimizer_usage": [],
        "model_saving": []
    }
    
    # PyTorch import patterns
    import_patterns = [
        r'import torch',
        r'from torch import',
        r'import torch\.nn',
        r'from torch\.nn import',
        r'import torch\.optim',
        r'from torch\.optim import'
    ]
    
    # Model definition patterns
    model_patterns = [
        r'class.*\(nn\.Module\)',
        r'nn\.Sequential\(',
        r'nn\.Linear\(',
        r'nn\.Conv2d\(',
        r'def forward\('
    ]
    
    # Training loop patterns
    training_patterns = [
        r'model\.train\(',
        r'for.*in.*dataloader',
        r'optimizer\.zero_grad\(',
        r'loss\.backward\(',
        r'optimizer\.step\('
    ]
    
    # Loss computation patterns
    loss_patterns = [
        r'nn\.CrossEntropyLoss\(',
        r'nn\.MSELoss\(',
        r'F\.cross_entropy\(',
        r'criterion\(',
        r'loss\s*='
    ]
    
    # Optimizer patterns
    optimizer_patterns = [
        r'torch\.optim\.Adam\(',
        r'torch\.optim\.SGD\(',
        r'optim\.Adam\(',
        r'optim\.SGD\('
    ]
    
    # Model saving patterns
    saving_patterns = [
        r'torch\.save\(',
        r'model\.state_dict\(',
        r'torch\.jit\.save\(',
        r'model\.save_pretrained\('
    ]
    
    # Search for patterns
    patterns["pytorch_imports"] = _find_patterns(script_content, import_patterns)
    patterns["model_definition"] = _find_patterns(script_content, model_patterns)
    patterns["training_loop"] = _find_patterns(script_content, training_patterns)
    patterns["loss_computation"] = _find_patterns(script_content, loss_patterns)
    patterns["optimizer_usage"] = _find_patterns(script_content, optimizer_patterns)
    patterns["model_saving"] = _find_patterns(script_content, saving_patterns)
    
    return patterns


def detect_processing_patterns(script_content: str) -> Dict[str, List[str]]:
    """
    Detect processing-specific patterns in script content.
    
    Args:
        script_content: Content of the processing script
        
    Returns:
        Dictionary containing detected processing patterns
    """
    patterns = {
        "data_loading_patterns": [],
        "data_transformation_patterns": [],
        "data_saving_patterns": [],
        "environment_variable_patterns": []
    }
    
    # Data loading patterns
    loading_patterns = [
        r'pd\.read_csv\(',
        r'pd\.read_parquet\(',
        r'np\.load\(',
        r'/opt/ml/processing/input',
        r'os\.listdir\(',
        r'glob\.glob\('
    ]
    
    # Data transformation patterns
    transformation_patterns = [
        r'\.transform\(',
        r'\.fit_transform\(',
        r'\.apply\(',
        r'\.map\(',
        r'\.groupby\(',
        r'\.merge\(',
        r'\.join\('
    ]
    
    # Data saving patterns
    saving_patterns = [
        r'\.to_csv\(',
        r'\.to_parquet\(',
        r'np\.save\(',
        r'/opt/ml/processing/output',
        r'pickle\.dump\(',
        r'joblib\.dump\('
    ]
    
    # Environment variable patterns
    env_patterns = [
        r'os\.environ\.get\(',
        r'os\.getenv\(',
        r'SM_CHANNEL_',
        r'SM_MODEL_DIR',
        r'SM_OUTPUT_DATA_DIR'
    ]
    
    # Search for patterns
    patterns["data_loading_patterns"] = _find_patterns(script_content, loading_patterns)
    patterns["data_transformation_patterns"] = _find_patterns(script_content, transformation_patterns)
    patterns["data_saving_patterns"] = _find_patterns(script_content, saving_patterns)
    patterns["environment_variable_patterns"] = _find_patterns(script_content, env_patterns)
    
    return patterns


def detect_create_model_patterns(script_content: str) -> Dict[str, List[str]]:
    """
    Detect CreateModel-specific patterns in script content.
    
    Args:
        script_content: Content of the model creation script
        
    Returns:
        Dictionary containing detected CreateModel patterns
    """
    patterns = {
        "model_loading_patterns": [],
        "inference_patterns": [],
        "serialization_patterns": [],
        "container_patterns": []
    }
    
    # Model loading patterns
    loading_patterns = [
        r'pickle\.load\(',
        r'joblib\.load\(',
        r'torch\.load\(',
        r'xgb\.Booster\(',
        r'model\.load\(',
        r'/opt/ml/model'
    ]
    
    # Inference patterns
    inference_patterns = [
        r'def model_fn\(',
        r'def input_fn\(',
        r'def predict_fn\(',
        r'def output_fn\(',
        r'\.predict\(',
        r'\.inference\('
    ]
    
    # Serialization patterns
    serialization_patterns = [
        r'json\.loads\(',
        r'json\.dumps\(',
        r'pickle\.loads\(',
        r'pickle\.dumps\(',
        r'np\.frombuffer\(',
        r'\.decode\('
    ]
    
    # Container patterns
    container_patterns = [
        r'sagemaker\.model\.Model\(',
        r'sagemaker\.pytorch\.PyTorchModel\(',
        r'sagemaker\.xgboost\.XGBoostModel\(',
        r'sagemaker\.sklearn\.SKLearnModel\('
    ]
    
    # Search for patterns
    patterns["model_loading_patterns"] = _find_patterns(script_content, loading_patterns)
    patterns["inference_patterns"] = _find_patterns(script_content, inference_patterns)
    patterns["serialization_patterns"] = _find_patterns(script_content, serialization_patterns)
    patterns["container_patterns"] = _find_patterns(script_content, container_patterns)
    
    return patterns


def detect_framework_from_script_content(script_content: str) -> Optional[str]:
    """
    Detect the primary framework used in script content.
    
    Args:
        script_content: Content of the script
        
    Returns:
        Detected framework name or None if no framework detected
    """
    framework_indicators = {
        'xgboost': [
            r'import xgboost',
            r'from xgboost',
            r'xgb\.train\(',
            r'xgb\.DMatrix'
        ],
        'pytorch': [
            r'import torch',
            r'from torch',
            r'nn\.Module',
            r'torch\.save\('
        ],
        'sklearn': [
            r'from sklearn',
            r'import sklearn',
            r'\.fit_transform\(',
            r'\.fit\('
        ],
        'tensorflow': [
            r'import tensorflow',
            r'from tensorflow',
            r'tf\.',
            r'keras\.'
        ]
    }
    
    framework_scores = {}
    
    for framework, patterns in framework_indicators.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, script_content, re.IGNORECASE)
            score += len(matches)
        framework_scores[framework] = score
    
    # Return framework with highest score, if any
    if framework_scores:
        best_framework = max(framework_scores, key=framework_scores.get)
        if framework_scores[best_framework] > 0:
            return best_framework
    
    return None


def _find_patterns(content: str, patterns: List[str]) -> List[str]:
    """
    Find all occurrences of patterns in content.
    
    Args:
        content: Text content to search
        patterns: List of regex patterns to search for
        
    Returns:
        List of matched pattern strings with line context
    """
    matches = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                context = f"Line {i}: {line.strip()}"
                if context not in matches:  # Avoid duplicates
                    matches.append(context)
    
    return matches


def get_step_type_specific_patterns(script_content: str, step_type: str) -> Dict[str, List[str]]:
    """
    Get step type-specific patterns for a script.
    
    Args:
        script_content: Content of the script
        step_type: SageMaker step type (Processing, Training, etc.)
        
    Returns:
        Dictionary containing step type-specific patterns
    """
    if step_type == "Training":
        return detect_training_patterns(script_content)
    elif step_type == "Processing":
        return detect_processing_patterns(script_content)
    elif step_type == "CreateModel":
        return detect_create_model_patterns(script_content)
    else:
        # For other step types, return basic patterns
        return {
            "general_patterns": _find_patterns(script_content, [
                r'import\s+\w+',
                r'def\s+\w+\(',
                r'class\s+\w+',
                r'/opt/ml/'
            ])
        }


def validate_framework_patterns(script_content: str, expected_framework: str) -> List[str]:
    """
    Validate that script contains expected framework patterns.
    
    Args:
        script_content: Content of the script
        expected_framework: Expected framework name
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    if expected_framework == "xgboost":
        xgb_patterns = detect_xgboost_patterns(script_content)
        if not xgb_patterns["xgboost_imports"]:
            issues.append("Missing XGBoost imports")
        if not xgb_patterns["xgboost_training"]:
            issues.append("Missing XGBoost training patterns")
            
    elif expected_framework == "pytorch":
        pytorch_patterns = detect_pytorch_patterns(script_content)
        if not pytorch_patterns["pytorch_imports"]:
            issues.append("Missing PyTorch imports")
        if not pytorch_patterns["model_definition"]:
            issues.append("Missing PyTorch model definition")
            
    elif expected_framework == "sklearn":
        # Basic sklearn validation
        if not re.search(r'from sklearn|import sklearn', script_content, re.IGNORECASE):
            issues.append("Missing scikit-learn imports")
    
    return issues
