"""
Original Central registry for all pipeline step names - BACKUP.
Single source of truth for step naming across config, builders, and specifications.
"""

from typing import Dict, List

# Core step name registry - canonical names used throughout the system
STEP_NAMES = {
    "Base": {
        "config_class": "BasePipelineConfig",
        "builder_step_name": "StepBuilderBase",
        "spec_type": "Base",
        "sagemaker_step_type": "Base",  # Special case
        "description": "Base pipeline configuration",
    },
    # Processing Steps (keep Processing as-is)
    "Processing": {
        "config_class": "ProcessingStepConfigBase",
        "builder_step_name": "ProcessingStepBuilder",
        "spec_type": "Processing",
        "sagemaker_step_type": "Processing",
        "description": "Base processing step",
    },
    # Data Loading Steps
    "CradleDataLoading": {
        "config_class": "CradleDataLoadingConfig",
        "builder_step_name": "CradleDataLoadingStepBuilder",
        "spec_type": "CradleDataLoading",
        "sagemaker_step_type": "CradleDataLoading",
        "description": "Cradle data loading step",
    },
    "DummyDataLoading": {
        "config_class": "DummyDataLoadingConfig",
        "builder_step_name": "DummyDataLoadingStepBuilder",
        "spec_type": "DummyDataLoading",
        "sagemaker_step_type": "Processing",
        "description": "Dummy data loading step that processes user-provided data instead of calling Cradle services",
    },
    # Processing Steps
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStepBuilder",
        "spec_type": "TabularPreprocessing",
        "sagemaker_step_type": "Processing",
        "description": "Tabular data preprocessing step",
    },
    "TemporalSequenceNormalization": {
        "config_class": "TemporalSequenceNormalizationConfig",
        "builder_step_name": "TemporalSequenceNormalizationStepBuilder",
        "spec_type": "TemporalSequenceNormalization",
        "sagemaker_step_type": "Processing",
        "description": "Temporal sequence normalization step for machine learning models with configurable sequence operations",
    },
    "TemporalFeatureEngineering": {
        "config_class": "TemporalFeatureEngineeringConfig",
        "builder_step_name": "TemporalFeatureEngineeringStepBuilder",
        "spec_type": "TemporalFeatureEngineering",
        "sagemaker_step_type": "Processing",
        "description": "Temporal feature engineering step that extracts comprehensive temporal features from normalized sequences for machine learning models",
    },
    "StratifiedSampling": {
        "config_class": "StratifiedSamplingConfig",
        "builder_step_name": "StratifiedSamplingStepBuilder",
        "spec_type": "StratifiedSampling",
        "sagemaker_step_type": "Processing",
        "description": "Stratified sampling step with multiple allocation strategies for class imbalance, causal analysis, and variance optimization",
    },
    "RiskTableMapping": {
        "config_class": "RiskTableMappingConfig",
        "builder_step_name": "RiskTableMappingStepBuilder",
        "spec_type": "RiskTableMapping",
        "sagemaker_step_type": "Processing",
        "description": "Risk table mapping step for categorical features",
    },
    "MissingValueImputation": {
        "config_class": "MissingValueImputationConfig",
        "builder_step_name": "MissingValueImputationStepBuilder",
        "spec_type": "MissingValueImputation",
        "sagemaker_step_type": "Processing",
        "description": "Missing value imputation step using statistical methods (mean, median, mode, constant) with pandas-safe values",
    },
    "FeatureSelection": {
        "config_class": "FeatureSelectionConfig",
        "builder_step_name": "FeatureSelectionStepBuilder",
        "spec_type": "FeatureSelection",
        "sagemaker_step_type": "Processing",
        "description": "Feature selection step using multiple statistical and ML-based methods with ensemble combination strategies",
    },
    "CurrencyConversion": {
        "config_class": "CurrencyConversionConfig",
        "builder_step_name": "CurrencyConversionStepBuilder",
        "spec_type": "CurrencyConversion",
        "sagemaker_step_type": "Processing",
        "description": "Currency conversion processing step",
    },
    "BedrockPromptTemplateGeneration": {
        "config_class": "BedrockPromptTemplateGenerationConfig",
        "builder_step_name": "BedrockPromptTemplateGenerationStepBuilder",
        "spec_type": "BedrockPromptTemplateGeneration",
        "sagemaker_step_type": "Processing",
        "description": "Bedrock prompt template generation step that creates structured prompt templates for classification tasks using the 5-component architecture pattern",
    },
    "BedrockProcessing": {
        "config_class": "BedrockProcessingConfig",
        "builder_step_name": "BedrockProcessingStepBuilder",
        "spec_type": "BedrockProcessing",
        "sagemaker_step_type": "Processing",
        "description": "Bedrock processing step that processes input data through AWS Bedrock models using generated prompt templates and validation schemas",
    },
    "BedrockBatchProcessing": {
        "config_class": "BedrockBatchProcessingConfig",
        "builder_step_name": "BedrockBatchProcessingStepBuilder",
        "spec_type": "BedrockBatchProcessing",
        "sagemaker_step_type": "Processing",
        "description": "Bedrock batch processing step that provides AWS Bedrock batch inference capabilities with automatic fallback to real-time processing for cost-efficient large dataset processing",
    },
    "LabelRulesetGeneration": {
        "config_class": "LabelRulesetGenerationConfig",
        "builder_step_name": "LabelRulesetGenerationStepBuilder",
        "spec_type": "LabelRulesetGeneration",
        "sagemaker_step_type": "Processing",
        "description": "Label ruleset generation step that validates and optimizes user-defined classification rules for transparent, maintainable rule-based label mapping in ML training pipelines",
    },
    "LabelRulesetExecution": {
        "config_class": "LabelRulesetExecutionConfig",
        "builder_step_name": "LabelRulesetExecutionStepBuilder",
        "spec_type": "LabelRulesetExecution",
        "sagemaker_step_type": "Processing",
        "description": "Label ruleset execution step that applies validated rulesets to processed data to generate classification labels using priority-based rule evaluation with execution-time field validation",
    },
    "ActiveSampleSelection": {
        "config_class": "ActiveSampleSelectionConfig",
        "builder_step_name": "ActiveSampleSelectionStepBuilder",
        "spec_type": "ActiveSampleSelection",
        "sagemaker_step_type": "Processing",
        "description": "Active sample selection step that intelligently selects high-value samples from model predictions for Semi-Supervised Learning (SSL) or Active Learning workflows using confidence-based, uncertainty-based, diversity-based, or hybrid strategies",
    },
    "PseudoLabelMerge": {
        "config_class": "PseudoLabelMergeConfig",
        "builder_step_name": "PseudoLabelMergeStepBuilder",
        "spec_type": "PseudoLabelMerge",
        "sagemaker_step_type": "Processing",
        "description": "Pseudo label merge step that intelligently combines labeled base data with pseudo-labeled or augmented samples for Semi-Supervised Learning (SSL) and Active Learning workflows with split-aware merge, auto-inferred split ratios, and provenance tracking",
    },
    # Training Steps
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",
        "builder_step_name": "PyTorchTrainingStepBuilder",
        "spec_type": "PyTorchTraining",
        "sagemaker_step_type": "Training",
        "description": "PyTorch model training step",
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost model training step",
    },
    "LightGBMTraining": {
        "config_class": "LightGBMTrainingConfig",
        "builder_step_name": "LightGBMTrainingStepBuilder",
        "spec_type": "LightGBMTraining",
        "sagemaker_step_type": "Training",
        "description": "LightGBM model training step using built-in algorithm",
    },
    "LightGBMMTTraining": {
        "config_class": "LightGBMMTTrainingConfig",
        "builder_step_name": "LightGBMMTTrainingStepBuilder",
        "spec_type": "LightGBMMTTraining",
        "sagemaker_step_type": "Training",
        "description": "LightGBM multi-task training with adaptive weighting and knowledge distillation",
    },
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "sagemaker_step_type": "Processing",
        "description": "Training step that uses a pretrained model",
    },
    # Evaluation Steps
    "XGBoostModelEval": {
        "config_class": "XGBoostModelEvalConfig",
        "builder_step_name": "XGBoostModelEvalStepBuilder",
        "spec_type": "XGBoostModelEval",
        "sagemaker_step_type": "Processing",
        "description": "XGBoost model evaluation step",
    },
    "XGBoostModelInference": {
        "config_class": "XGBoostModelInferenceConfig",
        "builder_step_name": "XGBoostModelInferenceStepBuilder",
        "spec_type": "XGBoostModelInference",
        "sagemaker_step_type": "Processing",
        "description": "XGBoost model inference step for prediction generation without metrics",
    },
    "LightGBMModelEval": {
        "config_class": "LightGBMModelEvalConfig",
        "builder_step_name": "LightGBMModelEvalStepBuilder",
        "spec_type": "LightGBMModelEval",
        "sagemaker_step_type": "Processing",
        "description": "LightGBM model evaluation step",
    },
    "LightGBMModelInference": {
        "config_class": "LightGBMModelInferenceConfig",
        "builder_step_name": "LightGBMModelInferenceStepBuilder",
        "spec_type": "LightGBMModelInference",
        "sagemaker_step_type": "Processing",
        "description": "LightGBM model inference step for prediction generation without metrics",
    },
    "LightGBMMTModelEval": {
        "config_class": "LightGBMMTModelEvalConfig",
        "builder_step_name": "LightGBMMTModelEvalStepBuilder",
        "spec_type": "LightGBMMTModelEval",
        "sagemaker_step_type": "Processing",
        "description": "LightGBM multi-task model evaluation step",
    },
    "LightGBMMTModelInference": {
        "config_class": "LightGBMMTModelInferenceConfig",
        "builder_step_name": "LightGBMMTModelInferenceStepBuilder",
        "spec_type": "LightGBMMTModelInference",
        "sagemaker_step_type": "Processing",
        "description": "LightGBM multi-task model inference step for prediction generation without metrics",
    },
    "PyTorchModelEval": {
        "config_class": "PyTorchModelEvalConfig",
        "builder_step_name": "PyTorchModelEvalStepBuilder",
        "spec_type": "PyTorchModelEval",
        "sagemaker_step_type": "Processing",
        "description": "PyTorch model evaluation step",
    },
    "PyTorchModelInference": {
        "config_class": "PyTorchModelInferenceConfig",
        "builder_step_name": "PyTorchModelInferenceStepBuilder",
        "spec_type": "PyTorchModelInference",
        "sagemaker_step_type": "Processing",
        "description": "PyTorch model inference step for prediction generation without metrics",
    },
    "ModelMetricsComputation": {
        "config_class": "ModelMetricsComputationConfig",
        "builder_step_name": "ModelMetricsComputationStepBuilder",
        "spec_type": "ModelMetricsComputation",
        "sagemaker_step_type": "Processing",
        "description": "Model metrics computation step for comprehensive performance evaluation",
    },
    "ModelWikiGenerator": {
        "config_class": "ModelWikiGeneratorConfig",
        "builder_step_name": "ModelWikiGeneratorStepBuilder",
        "spec_type": "ModelWikiGenerator",
        "sagemaker_step_type": "Processing",
        "description": "Model wiki generator step for automated documentation creation",
    },
    # Model Steps
    "PyTorchModel": {
        "config_class": "PyTorchModelConfig",
        "builder_step_name": "PyTorchModelStepBuilder",
        "spec_type": "PyTorchModel",
        "sagemaker_step_type": "CreateModel",
        "description": "PyTorch model creation step",
    },
    "XGBoostModel": {
        "config_class": "XGBoostModelConfig",
        "builder_step_name": "XGBoostModelStepBuilder",
        "spec_type": "XGBoostModel",
        "sagemaker_step_type": "CreateModel",
        "description": "XGBoost model creation step",
    },
    # Model Processing Steps
    "ModelCalibration": {
        "config_class": "ModelCalibrationConfig",
        "builder_step_name": "ModelCalibrationStepBuilder",
        "spec_type": "ModelCalibration",
        "sagemaker_step_type": "Processing",
        "description": "Calibrates model prediction scores to accurate probabilities",
    },
    "PercentileModelCalibration": {
        "config_class": "PercentileModelCalibrationConfig",
        "builder_step_name": "PercentileModelCalibrationStepBuilder",
        "spec_type": "PercentileModelCalibration",
        "sagemaker_step_type": "Processing",
        "description": "Creates percentile mapping from model scores using ROC curve analysis for consistent risk interpretation",
    },
    # Deployment Steps
    "Package": {
        "config_class": "PackageConfig",
        "builder_step_name": "PackageStepBuilder",
        "spec_type": "Package",
        "sagemaker_step_type": "Processing",
        "description": "Model packaging step",
    },
    "Registration": {
        "config_class": "RegistrationConfig",
        "builder_step_name": "RegistrationStepBuilder",
        "spec_type": "Registration",
        "sagemaker_step_type": "MimsModelRegistrationProcessing",
        "description": "Model registration step",
    },
    "Payload": {
        "config_class": "PayloadConfig",
        "builder_step_name": "PayloadStepBuilder",
        "spec_type": "Payload",
        "sagemaker_step_type": "Processing",
        "description": "Payload testing step",
    },
    # Utility Steps
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStepBuilder",
        "spec_type": "HyperparameterPrep",
        "sagemaker_step_type": "Lambda",  # Special classification
        "description": "Hyperparameter preparation step",
    },
    # Transform Steps
    "BatchTransform": {
        "config_class": "BatchTransformStepConfig",
        "builder_step_name": "BatchTransformStepBuilder",
        "spec_type": "BatchTransform",
        "sagemaker_step_type": "Transform",
        "description": "Batch transform step",
    },
}

# Generate the mappings that existing code expects
CONFIG_STEP_REGISTRY = {
    info["config_class"]: step_name for step_name, info in STEP_NAMES.items()
}

BUILDER_STEP_NAMES = {
    step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()
}

# Generate step specification types
SPEC_STEP_TYPES = {
    step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()
}
