"""
Step Specifications Module.

This module contains declarative specifications for pipeline steps, defining their
dependencies, outputs, and script contracts. These specifications serve as the
single source of truth for step behavior and connectivity.
"""

# Batch Transform specifications
from .batch_transform_calibration_spec import BATCH_TRANSFORM_CALIBRATION_SPEC
from .batch_transform_testing_spec import BATCH_TRANSFORM_TESTING_SPEC
from .batch_transform_training_spec import BATCH_TRANSFORM_TRAINING_SPEC
from .batch_transform_validation_spec import BATCH_TRANSFORM_VALIDATION_SPEC

# Currency Conversion specifications
from .currency_conversion_spec import CURRENCY_CONVERSION_SPEC
from .currency_conversion_calibration_spec import CURRENCY_CONVERSION_CALIBRATION_SPEC
from .currency_conversion_testing_spec import CURRENCY_CONVERSION_TESTING_SPEC
from .currency_conversion_training_spec import CURRENCY_CONVERSION_TRAINING_SPEC
from .currency_conversion_validation_spec import CURRENCY_CONVERSION_VALIDATION_SPEC

# Data Loading specifications
from .cradle_data_loading_spec import DATA_LOADING_SPEC
from .cradle_data_loading_calibration_spec import DATA_LOADING_CALIBRATION_SPEC
from .cradle_data_loading_testing_spec import DATA_LOADING_TESTING_SPEC
from .cradle_data_loading_training_spec import DATA_LOADING_TRAINING_SPEC
from .cradle_data_loading_validation_spec import DATA_LOADING_VALIDATION_SPEC

# Dummy Data Loading specifications
from .dummy_data_loading_spec import DUMMY_DATA_LOADING_SPEC
from .dummy_data_loading_calibration_spec import DUMMY_DATA_LOADING_CALIBRATION_SPEC
from .dummy_data_loading_testing_spec import DUMMY_DATA_LOADING_TESTING_SPEC
from .dummy_data_loading_training_spec import DUMMY_DATA_LOADING_TRAINING_SPEC
from .dummy_data_loading_validation_spec import DUMMY_DATA_LOADING_VALIDATION_SPEC

# Training specifications
from .dummy_training_spec import DUMMY_TRAINING_SPEC
from .lightgbm_training_spec import LIGHTGBM_TRAINING_SPEC
from .pytorch_training_spec import PYTORCH_TRAINING_SPEC
from .xgboost_training_spec import XGBOOST_TRAINING_SPEC

# Model specifications
from .pytorch_model_spec import PYTORCH_MODEL_SPEC
from .xgboost_model_spec import XGBOOST_MODEL_SPEC

# Missing Value Imputation specifications
from .missing_value_imputation_spec import MISSING_VALUE_IMPUTATION_SPEC
from .missing_value_imputation_calibration_spec import MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC
from .missing_value_imputation_testing_spec import MISSING_VALUE_IMPUTATION_TESTING_SPEC
from .missing_value_imputation_training_spec import MISSING_VALUE_IMPUTATION_TRAINING_SPEC
from .missing_value_imputation_validation_spec import MISSING_VALUE_IMPUTATION_VALIDATION_SPEC

# Model operations specifications
from .model_calibration_spec import MODEL_CALIBRATION_SPEC
from .model_calibration_calibration_spec import MODEL_CALIBRATION_CALIBRATION_SPEC
from .model_calibration_testing_spec import MODEL_CALIBRATION_TESTING_SPEC
from .model_calibration_training_spec import MODEL_CALIBRATION_TRAINING_SPEC
from .model_calibration_validation_spec import MODEL_CALIBRATION_VALIDATION_SPEC
from .model_metrics_computation_spec import MODEL_METRICS_COMPUTATION_SPEC
from .model_wiki_generator_spec import MODEL_WIKI_GENERATOR_SPEC
from .xgboost_model_eval_spec import MODEL_EVAL_SPEC
from .xgboost_model_inference_spec import XGBOOST_MODEL_INFERENCE_SPEC

# Packaging and deployment specifications
from .package_spec import PACKAGE_SPEC
from .payload_spec import PAYLOAD_SPEC
from .registration_spec import REGISTRATION_SPEC

# Stratified Sampling specifications
from .stratified_sampling_spec import STRATIFIED_SAMPLING_SPEC
from .stratified_sampling_calibration_spec import STRATIFIED_SAMPLING_CALIBRATION_SPEC
from .stratified_sampling_testing_spec import STRATIFIED_SAMPLING_TESTING_SPEC
from .stratified_sampling_training_spec import STRATIFIED_SAMPLING_TRAINING_SPEC
from .stratified_sampling_validation_spec import STRATIFIED_SAMPLING_VALIDATION_SPEC

# Preprocessing specifications
from .tabular_preprocessing_spec import TABULAR_PREPROCESSING_SPEC
from .tabular_preprocessing_calibration_spec import (
    TABULAR_PREPROCESSING_CALIBRATION_SPEC,
)
from .tabular_preprocessing_testing_spec import TABULAR_PREPROCESSING_TESTING_SPEC
from .tabular_preprocessing_training_spec import TABULAR_PREPROCESSING_TRAINING_SPEC
from .tabular_preprocessing_validation_spec import TABULAR_PREPROCESSING_VALIDATION_SPEC

# Risk Table Mapping specifications
from .risk_table_mapping_calibration_spec import RISK_TABLE_MAPPING_CALIBRATION_SPEC
from .risk_table_mapping_testing_spec import RISK_TABLE_MAPPING_TESTING_SPEC
from .risk_table_mapping_training_spec import RISK_TABLE_MAPPING_TRAINING_SPEC
from .risk_table_mapping_validation_spec import RISK_TABLE_MAPPING_VALIDATION_SPEC

# Temporal Sequence Normalization specifications
from .temporal_sequence_normalization_spec import TEMPORAL_SEQUENCE_NORMALIZATION_SPEC
from .temporal_sequence_normalization_calibration_spec import TEMPORAL_SEQUENCE_NORMALIZATION_CALIBRATION_SPEC
from .temporal_sequence_normalization_testing_spec import TEMPORAL_SEQUENCE_NORMALIZATION_TESTING_SPEC
from .temporal_sequence_normalization_training_spec import TEMPORAL_SEQUENCE_NORMALIZATION_TRAINING_SPEC
from .temporal_sequence_normalization_validation_spec import TEMPORAL_SEQUENCE_NORMALIZATION_VALIDATION_SPEC

# Temporal Feature Engineering specifications
from .temporal_feature_engineering_spec import TEMPORAL_FEATURE_ENGINEERING_SPEC
from .temporal_feature_engineering_calibration_spec import TEMPORAL_FEATURE_ENGINEERING_CALIBRATION_SPEC
from .temporal_feature_engineering_testing_spec import TEMPORAL_FEATURE_ENGINEERING_TESTING_SPEC
from .temporal_feature_engineering_training_spec import TEMPORAL_FEATURE_ENGINEERING_TRAINING_SPEC
from .temporal_feature_engineering_validation_spec import TEMPORAL_FEATURE_ENGINEERING_VALIDATION_SPEC

__all__ = [
    # Batch Transform specifications
    "BATCH_TRANSFORM_CALIBRATION_SPEC",
    "BATCH_TRANSFORM_TESTING_SPEC",
    "BATCH_TRANSFORM_TRAINING_SPEC",
    "BATCH_TRANSFORM_VALIDATION_SPEC",
    # Currency Conversion specifications
    "CURRENCY_CONVERSION_SPEC",
    "CURRENCY_CONVERSION_CALIBRATION_SPEC",
    "CURRENCY_CONVERSION_TESTING_SPEC",
    "CURRENCY_CONVERSION_TRAINING_SPEC",
    "CURRENCY_CONVERSION_VALIDATION_SPEC",
    # Data Loading specifications
    "DATA_LOADING_SPEC",
    "DATA_LOADING_CALIBRATION_SPEC",
    "DATA_LOADING_TESTING_SPEC",
    "DATA_LOADING_TRAINING_SPEC",
    "DATA_LOADING_VALIDATION_SPEC",
    # Dummy Data Loading specifications
    "DUMMY_DATA_LOADING_SPEC",
    "DUMMY_DATA_LOADING_CALIBRATION_SPEC",
    "DUMMY_DATA_LOADING_TESTING_SPEC",
    "DUMMY_DATA_LOADING_TRAINING_SPEC",
    "DUMMY_DATA_LOADING_VALIDATION_SPEC",
    # Training specifications
    "DUMMY_TRAINING_SPEC",
    "LIGHTGBM_TRAINING_SPEC",
    "PYTORCH_TRAINING_SPEC",
    "XGBOOST_TRAINING_SPEC",
    # Model specifications
    "PYTORCH_MODEL_SPEC",
    "XGBOOST_MODEL_SPEC",
    # Missing Value Imputation specifications
    "MISSING_VALUE_IMPUTATION_SPEC",
    "MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC",
    "MISSING_VALUE_IMPUTATION_TESTING_SPEC",
    "MISSING_VALUE_IMPUTATION_TRAINING_SPEC",
    "MISSING_VALUE_IMPUTATION_VALIDATION_SPEC",
    # Model operations specifications
    "MODEL_CALIBRATION_SPEC",
    "MODEL_CALIBRATION_CALIBRATION_SPEC",
    "MODEL_CALIBRATION_TESTING_SPEC",
    "MODEL_CALIBRATION_TRAINING_SPEC",
    "MODEL_CALIBRATION_VALIDATION_SPEC",
    "MODEL_METRICS_COMPUTATION_SPEC",
    "MODEL_WIKI_GENERATOR_SPEC",
    "MODEL_EVAL_SPEC",
    "XGBOOST_MODEL_INFERENCE_SPEC",
    # Packaging and deployment specifications
    "PACKAGE_SPEC",
    "PAYLOAD_SPEC",
    "REGISTRATION_SPEC",
    # Stratified Sampling specifications
    "STRATIFIED_SAMPLING_SPEC",
    "STRATIFIED_SAMPLING_CALIBRATION_SPEC",
    "STRATIFIED_SAMPLING_TESTING_SPEC",
    "STRATIFIED_SAMPLING_TRAINING_SPEC",
    "STRATIFIED_SAMPLING_VALIDATION_SPEC",
    # Preprocessing specifications
    "TABULAR_PREPROCESSING_SPEC",
    "TABULAR_PREPROCESSING_CALIBRATION_SPEC",
    "TABULAR_PREPROCESSING_TESTING_SPEC",
    "TABULAR_PREPROCESSING_TRAINING_SPEC",
    "TABULAR_PREPROCESSING_VALIDATION_SPEC",
    # Risk Table Mapping specifications
    "RISK_TABLE_MAPPING_CALIBRATION_SPEC",
    "RISK_TABLE_MAPPING_TESTING_SPEC",
    "RISK_TABLE_MAPPING_TRAINING_SPEC",
    "RISK_TABLE_MAPPING_VALIDATION_SPEC",
    # Temporal Sequence Normalization specifications
    "TEMPORAL_SEQUENCE_NORMALIZATION_SPEC",
    "TEMPORAL_SEQUENCE_NORMALIZATION_CALIBRATION_SPEC",
    "TEMPORAL_SEQUENCE_NORMALIZATION_TESTING_SPEC",
    "TEMPORAL_SEQUENCE_NORMALIZATION_TRAINING_SPEC",
    "TEMPORAL_SEQUENCE_NORMALIZATION_VALIDATION_SPEC",
    # Temporal Feature Engineering specifications
    "TEMPORAL_FEATURE_ENGINEERING_SPEC",
    "TEMPORAL_FEATURE_ENGINEERING_CALIBRATION_SPEC",
    "TEMPORAL_FEATURE_ENGINEERING_TESTING_SPEC",
    "TEMPORAL_FEATURE_ENGINEERING_TRAINING_SPEC",
    "TEMPORAL_FEATURE_ENGINEERING_VALIDATION_SPEC",
]
