"""
Step type detection for alignment validation.

Provides functions to detect SageMaker step types and ML frameworks
from scripts and their analysis results.
"""

from typing import List, Optional
from .script_analysis_models import ImportStatement


def detect_step_type_from_registry(script_name: str) -> str:
    """
    Use existing step registry to determine SageMaker step type.
    
    Args:
        script_name: Name of the script (without .py extension)
        
    Returns:
        SageMaker step type (Processing, Training, CreateModel, etc.)
        Defaults to "Processing" if detection fails
    """
    try:
        # Try the new hybrid registry system first
        from src.cursus.registry.step_names import get_sagemaker_step_type, get_canonical_name_from_file_name
        canonical_name = get_canonical_name_from_file_name(script_name)
        return get_sagemaker_step_type(canonical_name)
    except (ValueError, ImportError, AttributeError):
        try:
            # Fallback to old import path for backward compatibility
            from cursus.registry.step_names import get_sagemaker_step_type, get_canonical_name_from_file_name
            canonical_name = get_canonical_name_from_file_name(script_name)
            return get_sagemaker_step_type(canonical_name)
        except (ValueError, ImportError, AttributeError):
            # Default fallback to Processing for backward compatibility
            return "Processing"


def detect_framework_from_imports(imports: List) -> Optional[str]:
    """
    Detect framework from existing import analysis.
    
    Args:
        imports: List of ImportStatement objects or strings from script analysis
        
    Returns:
        Detected framework name or None if no framework detected
    """
    framework_patterns = {
        'xgboost': ['xgboost', 'xgb'],
        'pytorch': ['torch', 'pytorch'],
        'sklearn': ['sklearn', 'scikit-learn', 'scikit_learn'],
        'sagemaker': ['sagemaker'],
        'pandas': ['pandas', 'pd'],
        'numpy': ['numpy', 'np']
    }
    
    detected_frameworks = []
    
    for imp in imports:
        # Handle both ImportStatement objects and strings
        if hasattr(imp, 'module_name'):
            module_name_lower = imp.module_name.lower()
        else:
            # Assume it's a string
            module_name_lower = str(imp).lower()
            
        for framework, patterns in framework_patterns.items():
            if any(pattern in module_name_lower for pattern in patterns):
                detected_frameworks.append(framework)
    
    # Return primary framework (prioritize ML frameworks)
    priority_order = ['xgboost', 'pytorch', 'sklearn', 'sagemaker']
    for framework in priority_order:
        if framework in detected_frameworks:
            return framework
    
    # Return first detected framework if no priority match
    return detected_frameworks[0] if detected_frameworks else None


def detect_step_type_from_script_patterns(script_content: str) -> Optional[str]:
    """
    Detect step type from script content patterns.
    
    Args:
        script_content: Content of the script
        
    Returns:
        Detected step type or None if not determinable
    """
    # Training patterns
    training_indicators = [
        'xgb.train(',
        'model.fit(',
        'torch.save(',
        '/opt/ml/model',
        'hyperparameters.json',
        'model.save_model('
    ]
    
    # Processing patterns
    processing_indicators = [
        '/opt/ml/processing/input',
        '/opt/ml/processing/output',
        'pd.read_csv(',
        '.transform(',
        '.fit_transform('
    ]
    
    # CreateModel patterns
    create_model_indicators = [
        'def model_fn(',
        'def input_fn(',
        'def predict_fn(',
        'def output_fn(',
        'pickle.load(',
        'joblib.load('
    ]
    
    # Count pattern matches
    training_score = sum(1 for pattern in training_indicators if pattern in script_content)
    processing_score = sum(1 for pattern in processing_indicators if pattern in script_content)
    create_model_score = sum(1 for pattern in create_model_indicators if pattern in script_content)
    
    # Return type with highest score
    scores = {
        'Training': training_score,
        'Processing': processing_score,
        'CreateModel': create_model_score
    }
    
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores, key=scores.get)
    
    return None


def get_step_type_context(script_name: str, script_content: Optional[str] = None) -> dict:
    """
    Get comprehensive step type context for a script.
    
    Args:
        script_name: Name of the script
        script_content: Optional script content for pattern analysis
        
    Returns:
        Dictionary with step type context information
    """
    context = {
        'script_name': script_name,
        'registry_step_type': None,
        'pattern_step_type': None,
        'final_step_type': None,
        'confidence': 'low'
    }
    
    # Try registry-based detection
    try:
        registry_type = detect_step_type_from_registry(script_name)
        context['registry_step_type'] = registry_type
    except Exception:
        pass
    
    # Try pattern-based detection if script content available
    if script_content:
        try:
            pattern_type = detect_step_type_from_script_patterns(script_content)
            context['pattern_step_type'] = pattern_type
        except Exception:
            pass
    
    # Determine final step type and confidence
    registry_type = context['registry_step_type']
    pattern_type = context['pattern_step_type']
    
    if registry_type and pattern_type:
        if registry_type == pattern_type:
            context['final_step_type'] = registry_type
            context['confidence'] = 'high'
        else:
            # Registry takes precedence but lower confidence
            context['final_step_type'] = registry_type
            context['confidence'] = 'medium'
    elif registry_type:
        context['final_step_type'] = registry_type
        context['confidence'] = 'medium'
    elif pattern_type:
        context['final_step_type'] = pattern_type
        context['confidence'] = 'low'
    else:
        context['final_step_type'] = 'Processing'  # Default
        context['confidence'] = 'low'
    
    return context
