"""
Step builders module.

This module contains step builder classes that create SageMaker pipeline steps
using the specification-driven architecture. Each builder is responsible for
creating a specific type of step (processing, training, etc.) and integrates
with step specifications and script contracts.
"""

import logging

logger = logging.getLogger(__name__)

from ...core.base.builder_base import StepBuilderBase
from .builder_active_sample_selection_step import ActiveSampleSelectionStepBuilder
from .builder_batch_transform_step import BatchTransformStepBuilder
from .builder_bedrock_batch_processing_step import BedrockBatchProcessingStepBuilder
from .builder_bedrock_processing_step import BedrockProcessingStepBuilder
from .builder_bedrock_prompt_template_generation_step import (
    BedrockPromptTemplateGenerationStepBuilder,
)

# Import CradleDataLoadingStepBuilder with exception handling
# This builder depends on secure_ai_sandbox_workflow_python_sdk which may not be available
try:
    from .builder_cradle_data_loading_step import CradleDataLoadingStepBuilder

    CRADLE_DATA_LOADING_AVAILABLE = True
except ImportError as e:
    logger.warning(
        "CradleDataLoadingStepBuilder not available. "
        "This requires secure_ai_sandbox_workflow_python_sdk package. "
        f"Import error: {e}"
    )
    CradleDataLoadingStepBuilder = None
    CRADLE_DATA_LOADING_AVAILABLE = False

from .builder_currency_conversion_step import CurrencyConversionStepBuilder
from .builder_dummy_data_loading_step import DummyDataLoadingStepBuilder
from .builder_dummy_training_step import DummyTrainingStepBuilder
from .builder_feature_selection_step import FeatureSelectionStepBuilder
from .builder_label_ruleset_execution_step import LabelRulesetExecutionStepBuilder
from .builder_label_ruleset_generation_step import LabelRulesetGenerationStepBuilder
from .builder_lightgbm_model_eval_step import LightGBMModelEvalStepBuilder
from .builder_lightgbm_model_inference_step import LightGBMModelInferenceStepBuilder
from .builder_lightgbm_training_step import LightGBMTrainingStepBuilder
from .builder_lightgbmmt_training_step import LightGBMMTTrainingStepBuilder
from .builder_missing_value_imputation_step import MissingValueImputationStepBuilder
from .builder_model_calibration_step import ModelCalibrationStepBuilder
from .builder_model_metrics_computation_step import ModelMetricsComputationStepBuilder
from .builder_model_wiki_generator_step import ModelWikiGeneratorStepBuilder
from .builder_percentile_model_calibration_step import (
    PercentileModelCalibrationStepBuilder,
)
from .builder_pytorch_model_eval_step import PyTorchModelEvalStepBuilder
from .builder_pytorch_model_inference_step import PyTorchModelInferenceStepBuilder
from .builder_stratified_sampling_step import StratifiedSamplingStepBuilder
from .builder_xgboost_model_eval_step import XGBoostModelEvalStepBuilder
from .builder_pytorch_model_step import PyTorchModelStepBuilder
from .builder_xgboost_model_step import XGBoostModelStepBuilder
from .builder_package_step import PackageStepBuilder
from .builder_payload_step import PayloadStepBuilder

# Import RegistrationStepBuilder with exception handling
# This builder depends on secure_ai_sandbox_workflow_python_sdk which may not be available
try:
    from .builder_registration_step import RegistrationStepBuilder

    REGISTRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(
        "RegistrationStepBuilder not available. "
        "This requires secure_ai_sandbox_workflow_python_sdk package. "
        f"Import error: {e}"
    )
    RegistrationStepBuilder = None
    REGISTRATION_AVAILABLE = False

from .builder_risk_table_mapping_step import RiskTableMappingStepBuilder
from .builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from .builder_temporal_sequence_normalization_step import (
    TemporalSequenceNormalizationStepBuilder,
)
from .builder_temporal_feature_engineering_step import (
    TemporalFeatureEngineeringStepBuilder,
)
from .builder_pytorch_training_step import PyTorchTrainingStepBuilder
from .builder_xgboost_training_step import XGBoostTrainingStepBuilder
from .s3_utils import S3PathHandler

# Build __all__ list dynamically based on available imports
__all__ = [
    # Base class
    "StepBuilderBase",
    # Step builders
    "ActiveSampleSelectionStepBuilder",
    "BatchTransformStepBuilder",
    "BedrockBatchProcessingStepBuilder",
    "BedrockProcessingStepBuilder",
    "BedrockPromptTemplateGenerationStepBuilder",
    "CurrencyConversionStepBuilder",
    "DummyDataLoadingStepBuilder",
    "DummyTrainingStepBuilder",
    "FeatureSelectionStepBuilder",
    "LabelRulesetExecutionStepBuilder",
    "LabelRulesetGenerationStepBuilder",
    "LightGBMModelEvalStepBuilder",
    "LightGBMModelInferenceStepBuilder",
    "LightGBMTrainingStepBuilder",
    "LightGBMMTTrainingStepBuilder",
    "MissingValueImputationStepBuilder",
    "ModelCalibrationStepBuilder",
    "ModelMetricsComputationStepBuilder",
    "ModelWikiGeneratorStepBuilder",
    "PercentileModelCalibrationStepBuilder",
    "PyTorchModelEvalStepBuilder",
    "PyTorchModelInferenceStepBuilder",
    "StratifiedSamplingStepBuilder",
    "XGBoostModelEvalStepBuilder",
    "PyTorchModelStepBuilder",
    "XGBoostModelStepBuilder",
    "PackageStepBuilder",
    "PayloadStepBuilder",
    "RiskTableMappingStepBuilder",
    "TabularPreprocessingStepBuilder",
    "TemporalSequenceNormalizationStepBuilder",
    "TemporalFeatureEngineeringStepBuilder",
    "PyTorchTrainingStepBuilder",
    "XGBoostTrainingStepBuilder",
    # Utilities
    "S3PathHandler",
]

# Add optional builders to __all__ if available
if CRADLE_DATA_LOADING_AVAILABLE:
    __all__.append("CradleDataLoadingStepBuilder")

if REGISTRATION_AVAILABLE:
    __all__.append("RegistrationStepBuilder")
