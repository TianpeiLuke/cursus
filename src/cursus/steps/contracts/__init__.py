"""
Script Contracts Module.

This module contains script contracts that define the expected input and output
paths for scripts used in pipeline steps, as well as required environment variables.
These contracts are used by step specifications to map logical names to container paths.
"""

# Base contract classes - import from core module
from ...core.base.contract_base import ScriptContract, ValidationResult, ScriptAnalyzer
from .training_script_contract import TrainingScriptContract, TrainingScriptAnalyzer
from .contract_validator import ContractValidationReport, ScriptContractValidator

# Processing script contracts
from .currency_conversion_contract import CURRENCY_CONVERSION_CONTRACT
from .dummy_training_contract import DUMMY_TRAINING_CONTRACT
from .package_contract import PACKAGE_CONTRACT
from .payload_contract import PAYLOAD_CONTRACT
from .mims_registration_contract import MIMS_REGISTRATION_CONTRACT
from .model_calibration_contract import MODEL_CALIBRATION_CONTRACT
from .xgboost_model_eval_contract import XGBOOST_MODEL_EVAL_CONTRACT
from .risk_table_mapping_contract import RISK_TABLE_MAPPING_CONTRACT
from .tabular_preprocessing_contract import TABULAR_PREPROCESSING_CONTRACT

# Training script contracts
from .pytorch_training_contract import PYTORCH_TRAIN_CONTRACT
from .xgboost_training_contract import XGBOOST_TRAIN_CONTRACT

# Data loading contracts
from .cradle_data_loading_contract import CRADLE_DATA_LOADING_CONTRACT

__all__ = [
    # Base classes
    "ScriptContract",
    "ValidationResult",
    "ScriptAnalyzer",
    "TrainingScriptContract",
    "TrainingScriptAnalyzer",
    "ContractValidationReport",
    "ScriptContractValidator",
    # Processing contracts
    "CURRENCY_CONVERSION_CONTRACT",
    "DUMMY_TRAINING_CONTRACT",
    "PACKAGE_CONTRACT",
    "PAYLOAD_CONTRACT",
    "MIMS_REGISTRATION_CONTRACT",
    "MODEL_CALIBRATION_CONTRACT",
    "XGBOOST_MODEL_EVAL_CONTRACT",
    "RISK_TABLE_MAPPING_CONTRACT",
    "TABULAR_PREPROCESSING_CONTRACT",
    # Training contracts
    "PYTORCH_TRAIN_CONTRACT",
    "XGBOOST_TRAIN_CONTRACT",
    # Data loading contracts
    "CRADLE_DATA_LOADING_CONTRACT",
]
