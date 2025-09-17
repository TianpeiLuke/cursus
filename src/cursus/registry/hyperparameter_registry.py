"""
Central registry for all hyperparameter classes.
Single source of truth for hyperparameter configuration across the system.
"""

from typing import Dict, List, Optional

# Core hyperparameter registry
HYPERPARAMETER_REGISTRY = {
    # Base hyperparameter class
    "ModelHyperparameters": {
        "class_name": "ModelHyperparameters",
        "module_path": "cursus.core.base.hyperparameters_base",
        "model_type": None,  # Base class is not specific to any model
        "description": "Base class for all model hyperparameters",
    },
    # XGBoost hyperparameters
    "XGBoostHyperparameters": {
        "class_name": "XGBoostHyperparameters",
        "module_path": "cursus.steps.hyperparams.xgboost_hyperparameters",
        "model_type": "xgboost",
        "description": "Hyperparameters for XGBoost models",
    },
    # BSM hyperparameters
    "BSMModelHyperparameters": {
        "class_name": "BSMModelHyperparameters",
        "module_path": "cursus.steps.hyperparams.pytorch_hyperparameters",
        "model_type": "pytorch",
        "description": "Hyperparameters for BSM PyTorch models",
    },
}


# Helper functions
def get_all_hyperparameter_classes() -> List[str]:
    """Get all registered hyperparameter class names."""
    return list(HYPERPARAMETER_REGISTRY.keys())


def get_hyperparameter_class_by_model_type(model_type: str) -> Optional[str]:
    """Find a hyperparameter class for a specific model type."""
    for class_name, info in HYPERPARAMETER_REGISTRY.items():
        if info["model_type"] == model_type:
            return class_name
    return None


def get_module_path(class_name: str) -> Optional[str]:
    """Get the module path for a hyperparameter class."""
    if class_name not in HYPERPARAMETER_REGISTRY:
        return None
    return HYPERPARAMETER_REGISTRY[class_name]["module_path"]


def get_all_hyperparameter_info() -> Dict[str, Dict[str, str]]:
    """Get complete information for all registered hyperparameter classes."""
    return HYPERPARAMETER_REGISTRY.copy()


def validate_hyperparameter_class(class_name: str) -> bool:
    """Validate that a hyperparameter class exists in the registry."""
    return class_name in HYPERPARAMETER_REGISTRY
