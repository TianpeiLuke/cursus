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
from .missing_value_imputation_contract import MISSING_VALUE_IMPUTATION_CONTRACT
from .model_calibration_contract import MODEL_CALIBRATION_CONTRACT
from .model_metrics_computation_contract import MODEL_METRICS_COMPUTATION_CONTRACT
from .model_wiki_generator_contract import MODEL_WIKI_GENERATOR_CONTRACT
from .package_contract import PACKAGE_CONTRACT
from .payload_contract import PAYLOAD_CONTRACT
from .mims_registration_contract import MIMS_REGISTRATION_CONTRACT
from .risk_table_mapping_contract import RISK_TABLE_MAPPING_CONTRACT
from .stratified_sampling_contract import STRATIFIED_SAMPLING_CONTRACT
from .tabular_preprocessing_contract import TABULAR_PREPROCESSING_CONTRACT
from .temporal_sequence_normalization_contract import TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT
from .temporal_feature_engineering_contract import TEMPORAL_FEATURE_ENGINEERING_CONTRACT
from .xgboost_model_eval_contract import XGBOOST_MODEL_EVAL_CONTRACT
from .xgboost_model_inference_contract import XGBOOST_MODEL_INFERENCE_CONTRACT

# Training script contracts
from .lightgbm_training_contract import LIGHTGBM_TRAINING_CONTRACT
from .pytorch_training_contract import PYTORCH_TRAIN_CONTRACT
from .xgboost_training_contract import XGBOOST_TRAIN_CONTRACT

# Data loading contracts
from .cradle_data_loading_contract import CRADLE_DATA_LOADING_CONTRACT
from .dummy_data_loading_contract import DUMMY_DATA_LOADING_CONTRACT

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
    "MISSING_VALUE_IMPUTATION_CONTRACT",
    "MODEL_CALIBRATION_CONTRACT",
    "MODEL_METRICS_COMPUTATION_CONTRACT",
    "MODEL_WIKI_GENERATOR_CONTRACT",
    "PACKAGE_CONTRACT",
    "PAYLOAD_CONTRACT",
    "MIMS_REGISTRATION_CONTRACT",
    "RISK_TABLE_MAPPING_CONTRACT",
    "STRATIFIED_SAMPLING_CONTRACT",
    "TABULAR_PREPROCESSING_CONTRACT",
    "TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT",
    "TEMPORAL_FEATURE_ENGINEERING_CONTRACT",
    "XGBOOST_MODEL_EVAL_CONTRACT",
    "XGBOOST_MODEL_INFERENCE_CONTRACT",
    # Training contracts
    "LIGHTGBM_TRAINING_CONTRACT",
    "PYTORCH_TRAIN_CONTRACT",
    "XGBOOST_TRAIN_CONTRACT",
    # Data loading contracts
    "CRADLE_DATA_LOADING_CONTRACT",
    "DUMMY_DATA_LOADING_CONTRACT",
]
