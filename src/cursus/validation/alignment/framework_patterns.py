"""
Framework Patterns Detection

Provides framework-specific pattern detection for validation enhancement.
Detects training patterns, XGBoost patterns, PyTorch patterns, and other framework-specific code patterns.
"""

from typing import Dict, Any, List, Optional


def detect_training_patterns(script_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect general training patterns in script analysis.
    
    Args:
        script_analysis: Script analysis results
        
    Returns:
        Dictionary containing detected training patterns
    """
    patterns = {
        'has_training_loop': False,
        'has_model_saving': False,
        'has_hyperparameter_loading': False,
        'has_data_loading': False,
        'has_evaluation': False
    }
    
    functions = script_analysis.get('functions', [])
    path_references = script_analysis.get('path_references', [])
    
    # Check for training loop patterns
    training_keywords = ['fit', 'train', 'epoch', 'batch', 'forward', 'backward']
    patterns['has_training_loop'] = any(
        any(keyword.lower() in str(func).lower() for keyword in training_keywords)
        for func in functions
    )
    
    # Check for model saving patterns
    saving_keywords = ['save', 'dump', 'pickle', 'joblib', 'torch.save']
    model_path_keywords = ['/opt/ml/model']
    patterns['has_model_saving'] = (
        any(any(keyword.lower() in str(func).lower() for keyword in saving_keywords) for func in functions) or
        any(any(keyword in str(path) for keyword in model_path_keywords) for path in path_references)
    )
    
    # Check for hyperparameter loading patterns
    hp_keywords = ['hyperparameters', 'config', 'params']
    hp_path_keywords = ['/opt/ml/input/data/config']
    patterns['has_hyperparameter_loading'] = (
        any(any(keyword.lower() in str(func).lower() for keyword in hp_keywords) for func in functions) or
        any(any(keyword in str(path) for keyword in hp_path_keywords) for path in path_references)
    )
    
    # Check for data loading patterns
    data_keywords = ['read_csv', 'load', 'data']
    data_path_keywords = ['/opt/ml/input/data/train']
    patterns['has_data_loading'] = (
        any(any(keyword.lower() in str(func).lower() for keyword in data_keywords) for func in functions) or
        any(any(keyword in str(path) for keyword in data_path_keywords) for path in path_references)
    )
    
    # Check for evaluation patterns
    eval_keywords = ['evaluate', 'score', 'metric', 'accuracy', 'loss', 'validation']
    patterns['has_evaluation'] = any(
        any(keyword.lower() in str(func).lower() for keyword in eval_keywords)
        for func in functions
    )
    
    return patterns


def detect_xgboost_patterns(script_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect XGBoost-specific patterns in script analysis.
    
    Args:
        script_analysis: Script analysis results
        
    Returns:
        Dictionary containing detected XGBoost patterns
    """
    patterns = {
        'has_xgboost_import': False,
        'has_dmatrix_usage': False,
        'has_xgb_train': False,
        'has_booster_usage': False,
        'has_model_loading': False
    }
    
    imports = script_analysis.get('imports', [])
    functions = script_analysis.get('functions', [])
    
    # Check for XGBoost imports
    xgb_import_keywords = ['xgboost', 'xgb']
    patterns['has_xgboost_import'] = any(
        any(keyword.lower() in str(imp).lower() for keyword in xgb_import_keywords)
        for imp in imports
    )
    
    # Check for DMatrix usage
    dmatrix_keywords = ['DMatrix', 'xgb.DMatrix']
    patterns['has_dmatrix_usage'] = any(
        any(keyword in str(func) for keyword in dmatrix_keywords)
        for func in functions
    )
    
    # Check for XGBoost training
    train_keywords = ['xgb.train', 'train']
    patterns['has_xgb_train'] = any(
        any(keyword in str(func) for keyword in train_keywords)
        for func in functions
    )
    
    # Check for Booster usage
    booster_keywords = ['Booster', 'xgb.Booster']
    patterns['has_booster_usage'] = any(
        any(keyword in str(func) for keyword in booster_keywords)
        for func in functions
    )
    
    # Check for model loading
    loading_keywords = ['load_model', 'pickle.load', 'joblib.load']
    patterns['has_model_loading'] = any(
        any(keyword in str(func) for keyword in loading_keywords)
        for func in functions
    )
    
    return patterns


def detect_pytorch_patterns(script_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect PyTorch-specific patterns in script analysis.
    
    Args:
        script_analysis: Script analysis results
        
    Returns:
        Dictionary containing detected PyTorch patterns
    """
    patterns = {
        'has_torch_import': False,
        'has_nn_module': False,
        'has_optimizer': False,
        'has_loss_function': False,
        'has_model_loading': False,
        'has_training_loop': False
    }
    
    imports = script_analysis.get('imports', [])
    functions = script_analysis.get('functions', [])
    
    # Check for PyTorch imports
    torch_import_keywords = ['torch', 'pytorch']
    patterns['has_torch_import'] = any(
        any(keyword.lower() in str(imp).lower() for keyword in torch_import_keywords)
        for imp in imports
    )
    
    # Check for nn.Module usage
    module_keywords = ['nn.Module', 'torch.nn']
    patterns['has_nn_module'] = any(
        any(keyword in str(func) for keyword in module_keywords)
        for func in functions
    )
    
    # Check for optimizer usage
    optimizer_keywords = ['optim', 'optimizer', 'Adam', 'SGD']
    patterns['has_optimizer'] = any(
        any(keyword in str(func) for keyword in optimizer_keywords)
        for func in functions
    )
    
    # Check for loss function usage
    loss_keywords = ['loss', 'criterion', 'CrossEntropyLoss', 'MSELoss']
    patterns['has_loss_function'] = any(
        any(keyword in str(func) for keyword in loss_keywords)
        for func in functions
    )
    
    # Check for model loading
    loading_keywords = ['torch.load', 'load_state_dict']
    patterns['has_model_loading'] = any(
        any(keyword in str(func) for keyword in loading_keywords)
        for func in functions
    )
    
    # Check for training loop patterns
    training_keywords = ['forward', 'backward', 'zero_grad', 'step']
    patterns['has_training_loop'] = any(
        any(keyword in str(func) for keyword in training_keywords)
        for func in functions
    )
    
    return patterns


def detect_sklearn_patterns(script_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect Scikit-learn-specific patterns in script analysis.
    
    Args:
        script_analysis: Script analysis results
        
    Returns:
        Dictionary containing detected sklearn patterns
    """
    patterns = {
        'has_sklearn_import': False,
        'has_preprocessing': False,
        'has_model_training': False,
        'has_model_evaluation': False,
        'has_pipeline': False
    }
    
    imports = script_analysis.get('imports', [])
    functions = script_analysis.get('functions', [])
    
    # Check for sklearn imports
    sklearn_import_keywords = ['sklearn', 'scikit-learn']
    patterns['has_sklearn_import'] = any(
        any(keyword.lower() in str(imp).lower() for keyword in sklearn_import_keywords)
        for imp in imports
    )
    
    # Check for preprocessing usage
    preprocessing_keywords = ['preprocessing', 'StandardScaler', 'LabelEncoder', 'fit_transform']
    patterns['has_preprocessing'] = any(
        any(keyword in str(func) for keyword in preprocessing_keywords)
        for func in functions
    )
    
    # Check for model training
    training_keywords = ['fit', 'train', 'RandomForestClassifier', 'SVC']
    patterns['has_model_training'] = any(
        any(keyword in str(func) for keyword in training_keywords)
        for func in functions
    )
    
    # Check for model evaluation
    evaluation_keywords = ['score', 'predict', 'accuracy_score', 'classification_report']
    patterns['has_model_evaluation'] = any(
        any(keyword in str(func) for keyword in evaluation_keywords)
        for func in functions
    )
    
    # Check for pipeline usage
    pipeline_keywords = ['Pipeline', 'make_pipeline']
    patterns['has_pipeline'] = any(
        any(keyword in str(func) for keyword in pipeline_keywords)
        for func in functions
    )
    
    return patterns


def detect_pandas_patterns(script_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect Pandas-specific patterns in script analysis.
    
    Args:
        script_analysis: Script analysis results
        
    Returns:
        Dictionary containing detected pandas patterns
    """
    patterns = {
        'has_pandas_import': False,
        'has_dataframe_operations': False,
        'has_data_loading': False,
        'has_data_saving': False,
        'has_data_transformation': False
    }
    
    imports = script_analysis.get('imports', [])
    functions = script_analysis.get('functions', [])
    
    # Check for pandas imports
    pandas_import_keywords = ['pandas', 'pd']
    patterns['has_pandas_import'] = any(
        any(keyword.lower() in str(imp).lower() for keyword in pandas_import_keywords)
        for imp in imports
    )
    
    # Check for DataFrame operations
    dataframe_keywords = ['DataFrame', 'pd.DataFrame', 'df.']
    patterns['has_dataframe_operations'] = any(
        any(keyword in str(func) for keyword in dataframe_keywords)
        for func in functions
    )
    
    # Check for data loading
    loading_keywords = ['read_csv', 'read_json', 'read_excel', 'pd.read']
    patterns['has_data_loading'] = any(
        any(keyword in str(func) for keyword in loading_keywords)
        for func in functions
    )
    
    # Check for data saving
    saving_keywords = ['to_csv', 'to_json', 'to_excel']
    patterns['has_data_saving'] = any(
        any(keyword in str(func) for keyword in saving_keywords)
        for func in functions
    )
    
    # Check for data transformation
    transform_keywords = ['groupby', 'merge', 'join', 'pivot', 'apply', 'map']
    patterns['has_data_transformation'] = any(
        any(keyword in str(func) for keyword in transform_keywords)
        for func in functions
    )
    
    return patterns


def get_framework_patterns(framework: str, script_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get framework-specific patterns for the given framework.
    
    Args:
        framework: Framework name
        script_analysis: Script analysis results
        
    Returns:
        Dictionary containing framework-specific patterns
    """
    framework_detectors = {
        'xgboost': detect_xgboost_patterns,
        'pytorch': detect_pytorch_patterns,
        'sklearn': detect_sklearn_patterns,
        'pandas': detect_pandas_patterns,
        'training': detect_training_patterns
    }
    
    detector = framework_detectors.get(framework.lower())
    if detector:
        return detector(script_analysis)
    
    return {}


def get_all_framework_patterns(script_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get patterns for all supported frameworks.
    
    Args:
        script_analysis: Script analysis results
        
    Returns:
        Dictionary mapping framework names to their patterns
    """
    frameworks = ['xgboost', 'pytorch', 'sklearn', 'pandas', 'training']
    all_patterns = {}
    
    for framework in frameworks:
        all_patterns[framework] = get_framework_patterns(framework, script_analysis)
    
    return all_patterns


def detect_framework_from_script_content(script_content: str) -> Optional[str]:
    """
    Detect the primary framework used in script content.
    
    Args:
        script_content: The content of the script as a string
        
    Returns:
        The detected framework name or None if no framework is detected
    """
    if not script_content:
        return None
    
    # Framework detection patterns with scoring
    framework_scores = {
        'xgboost': 0,
        'pytorch': 0,
        'sklearn': 0,
        'pandas': 0
    }
    
    # XGBoost patterns
    xgb_patterns = [
        'import xgboost', 'from xgboost', 'xgb.', 'DMatrix', 'xgb.train',
        'XGBClassifier', 'XGBRegressor', 'xgboost.train'
    ]
    for pattern in xgb_patterns:
        if pattern in script_content:
            framework_scores['xgboost'] += 2 if 'import' in pattern else 1
    
    # PyTorch patterns
    pytorch_patterns = [
        'import torch', 'from torch', 'torch.', 'nn.Module', 'torch.nn',
        'torch.optim', 'torch.save', 'torch.load', 'pytorch'
    ]
    for pattern in pytorch_patterns:
        if pattern in script_content:
            framework_scores['pytorch'] += 2 if 'import' in pattern else 1
    
    # Scikit-learn patterns
    sklearn_patterns = [
        'import sklearn', 'from sklearn', 'scikit-learn', 'fit_transform',
        'RandomForestClassifier', 'SVC', 'StandardScaler'
    ]
    for pattern in sklearn_patterns:
        if pattern in script_content:
            framework_scores['sklearn'] += 2 if 'import' in pattern else 1
    
    # Pandas patterns
    pandas_patterns = [
        'import pandas', 'from pandas', 'pd.', 'DataFrame', 'read_csv',
        'to_csv', 'pd.read'
    ]
    for pattern in pandas_patterns:
        if pattern in script_content:
            framework_scores['pandas'] += 2 if 'import' in pattern else 1
    
    # Return the framework with the highest score
    if max(framework_scores.values()) > 0:
        return max(framework_scores, key=framework_scores.get)
    
    return None


def detect_framework_from_imports(imports: List[str]) -> Optional[str]:
    """
    Detect framework from import statements.
    
    Args:
        imports: List of import statements
        
    Returns:
        The detected framework name or None if no framework is detected
    """
    if not imports:
        return None
    
    # Convert imports to a single string for pattern matching
    imports_str = ' '.join(str(imp) for imp in imports)
    
    # Priority order for framework detection (ML frameworks first)
    framework_patterns = [
        ('xgboost', ['xgboost', 'xgb']),
        ('pytorch', ['torch', 'pytorch']),
        ('sklearn', ['sklearn', 'scikit-learn']),
        ('pandas', ['pandas', 'pd'])
    ]
    
    for framework, patterns in framework_patterns:
        if any(pattern in imports_str.lower() for pattern in patterns):
            return framework
    
    return None
