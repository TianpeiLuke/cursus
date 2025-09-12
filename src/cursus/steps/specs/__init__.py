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

# Training specifications
from .dummy_training_spec import DUMMY_TRAINING_SPEC
from .pytorch_training_spec import PYTORCH_TRAINING_SPEC
from .xgboost_training_spec import XGBOOST_TRAINING_SPEC

# Model specifications
from .pytorch_model_spec import PYTORCH_MODEL_SPEC
from .xgboost_model_spec import XGBOOST_MODEL_SPEC

# Model operations specifications
from .model_calibration_spec import MODEL_CALIBRATION_SPEC
from .xgboost_model_eval_spec import MODEL_EVAL_SPEC

# Packaging and deployment specifications
from .package_spec import PACKAGE_SPEC
from .payload_spec import PAYLOAD_SPEC
from .registration_spec import REGISTRATION_SPEC

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

__all__ = [
    # Batch Transform specifications
    "BATCH_TRANSFORM_CALIBRATION_SPEC",
    "BATCH_TRANSFORM_TESTING_SPEC",
    "BATCH_TRANSFORM_TRAINING_SPEC",
    "BATCH_TRANSFORM_VALIDATION_SPEC",
    # Currency Conversion specifications
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
    # Training specifications
    "DUMMY_TRAINING_SPEC",
    "PYTORCH_TRAINING_SPEC",
    "XGBOOST_TRAINING_SPEC",
    # Model specifications
    "PYTORCH_MODEL_SPEC",
    "XGBOOST_MODEL_SPEC",
    # Model operations specifications
    "MODEL_CALIBRATION_SPEC",
    "MODEL_EVAL_SPEC",
    # Packaging and deployment specifications
    "PACKAGE_SPEC",
    "PAYLOAD_SPEC",
    "REGISTRATION_SPEC",
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
]
