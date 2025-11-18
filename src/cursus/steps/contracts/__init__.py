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
from .active_sample_selection_contract import ACTIVE_SAMPLE_SELECTION_CONTRACT
from .bedrock_batch_processing_contract import BEDROCK_BATCH_PROCESSING_CONTRACT
from .bedrock_processing_contract import BEDROCK_PROCESSING_CONTRACT
from .bedrock_prompt_template_generation_contract import (
    BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT,
)
from .currency_conversion_contract import CURRENCY_CONVERSION_CONTRACT
from .dummy_training_contract import DUMMY_TRAINING_CONTRACT
from .feature_selection_contract import FEATURE_SELECTION_CONTRACT
from .label_ruleset_execution_contract import LABEL_RULESET_EXECUTION_CONTRACT
from .label_ruleset_generation_contract import LABEL_RULESET_GENERATION_CONTRACT
from .lightgbm_model_eval_contract import LIGHTGBM_MODEL_EVAL_CONTRACT
from .lightgbm_model_inference_contract import LIGHTGBM_MODEL_INFERENCE_CONTRACT
from .missing_value_imputation_contract import MISSING_VALUE_IMPUTATION_CONTRACT
from .model_calibration_contract import MODEL_CALIBRATION_CONTRACT
from .model_metrics_computation_contract import MODEL_METRICS_COMPUTATION_CONTRACT
from .model_wiki_generator_contract import MODEL_WIKI_GENERATOR_CONTRACT
from .package_contract import PACKAGE_CONTRACT
from .payload_contract import PAYLOAD_CONTRACT
from .percentile_model_calibration_contract import PERCENTILE_MODEL_CALIBRATION_CONTRACT
from .pytorch_model_eval_contract import PYTORCH_MODEL_EVAL_CONTRACT
from .pytorch_model_inference_contract import PYTORCH_MODEL_INFERENCE_CONTRACT
from .mims_registration_contract import MIMS_REGISTRATION_CONTRACT
from .risk_table_mapping_contract import RISK_TABLE_MAPPING_CONTRACT
from .stratified_sampling_contract import STRATIFIED_SAMPLING_CONTRACT
from .tabular_preprocessing_contract import TABULAR_PREPROCESSING_CONTRACT
from .temporal_sequence_normalization_contract import (
    TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT,
)
from .temporal_feature_engineering_contract import TEMPORAL_FEATURE_ENGINEERING_CONTRACT
from .xgboost_model_eval_contract import XGBOOST_MODEL_EVAL_CONTRACT
from .xgboost_model_inference_contract import XGBOOST_MODEL_INFERENCE_CONTRACT

# Training script contracts
from .lightgbm_training_contract import LIGHTGBM_TRAIN_CONTRACT
from .lightgbmmt_training_contract import LIGHTGBMMT_TRAIN_CONTRACT
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
    "ACTIVE_SAMPLE_SELECTION_CONTRACT",
    "BEDROCK_BATCH_PROCESSING_CONTRACT",
    "BEDROCK_PROCESSING_CONTRACT",
    "BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT",
    "CURRENCY_CONVERSION_CONTRACT",
    "DUMMY_TRAINING_CONTRACT",
    "FEATURE_SELECTION_CONTRACT",
    "LABEL_RULESET_EXECUTION_CONTRACT",
    "LABEL_RULESET_GENERATION_CONTRACT",
    "LIGHTGBM_MODEL_EVAL_CONTRACT",
    "LIGHTGBM_MODEL_INFERENCE_CONTRACT",
    "MISSING_VALUE_IMPUTATION_CONTRACT",
    "MODEL_CALIBRATION_CONTRACT",
    "MODEL_METRICS_COMPUTATION_CONTRACT",
    "MODEL_WIKI_GENERATOR_CONTRACT",
    "PACKAGE_CONTRACT",
    "PAYLOAD_CONTRACT",
    "PERCENTILE_MODEL_CALIBRATION_CONTRACT",
    "PYTORCH_MODEL_EVAL_CONTRACT",
    "PYTORCH_MODEL_INFERENCE_CONTRACT",
    "MIMS_REGISTRATION_CONTRACT",
    "RISK_TABLE_MAPPING_CONTRACT",
    "STRATIFIED_SAMPLING_CONTRACT",
    "TABULAR_PREPROCESSING_CONTRACT",
    "TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT",
    "TEMPORAL_FEATURE_ENGINEERING_CONTRACT",
    "XGBOOST_MODEL_EVAL_CONTRACT",
    "XGBOOST_MODEL_INFERENCE_CONTRACT",
    # Training contracts
    "LIGHTGBM_TRAIN_CONTRACT",
    "LIGHTGBMMT_TRAIN_CONTRACT",
    "PYTORCH_TRAIN_CONTRACT",
    "XGBOOST_TRAIN_CONTRACT",
    # Data loading contracts
    "CRADLE_DATA_LOADING_CONTRACT",
    "DUMMY_DATA_LOADING_CONTRACT",
]
