"""
Step Type Test Variant Registry

This module defines the mapping between SageMaker step types and their corresponding
universal test variant classes. It provides the registry for automatic detection
and instantiation of appropriate test variants based on step type.

The registry supports the hierarchical universal tester system where each SageMaker
step type gets specialized validation through dedicated test variant classes.
"""

from typing import Dict, List, Type, Any, Optional
from dataclasses import dataclass


@dataclass
class StepTypeRequirements:
    """Requirements specification for a SageMaker step type."""
    required_methods: List[str]
    optional_methods: List[str]
    required_attributes: List[str]
    step_class: str
    sagemaker_objects: List[str]
    validation_rules: Optional[Dict[str, Any]] = None


# Step type requirements mapping
STEP_TYPE_REQUIREMENTS: Dict[str, StepTypeRequirements] = {
    "Processing": StepTypeRequirements(
        required_methods=[
            "_create_processor",
            "_get_inputs", 
            "_get_outputs"
        ],
        optional_methods=[
            "_get_property_files",
            "_get_job_arguments",
            "_get_code_path",
            "_get_environment_variables"
        ],
        required_attributes=[
            "processor_class"
        ],
        step_class="ProcessingStep",
        sagemaker_objects=[
            "Processor",
            "ProcessingInput", 
            "ProcessingOutput",
            "PropertyFile"
        ],
        validation_rules={
            "min_inputs": 1,
            "max_inputs": 10,
            "min_outputs": 1,
            "max_outputs": 10,
            "required_processor_types": ["ScriptProcessor", "FrameworkProcessor"]
        }
    ),
    
    "Training": StepTypeRequirements(
        required_methods=[
            "_create_estimator",
            "_get_training_inputs"
        ],
        optional_methods=[
            "_get_hyperparameters",
            "_get_metric_definitions",
            "_get_checkpoint_config",
            "_get_debugger_config"
        ],
        required_attributes=[
            "estimator_class"
        ],
        step_class="TrainingStep",
        sagemaker_objects=[
            "Estimator",
            "TrainingInput"
        ],
        validation_rules={
            "min_training_inputs": 1,
            "max_training_inputs": 20,
            "required_estimator_types": ["Framework", "Algorithm"]
        }
    ),
    
    "Transform": StepTypeRequirements(
        required_methods=[
            "_create_transformer",
            "_get_transform_inputs"
        ],
        optional_methods=[
            "_get_transform_strategy",
            "_get_assemble_with",
            "_get_accept_type"
        ],
        required_attributes=[
            "transformer_class"
        ],
        step_class="TransformStep",
        sagemaker_objects=[
            "Transformer",
            "TransformInput"
        ],
        validation_rules={
            "valid_strategies": ["SingleRecord", "MultiRecord"],
            "valid_assemble_with": ["Line", "None"]
        }
    ),
    
    "CreateModel": StepTypeRequirements(
        required_methods=[
            "_create_model",
            "_get_model_data"
        ],
        optional_methods=[
            "_get_container_definitions",
            "_get_inference_code",
            "_get_environment_variables"
        ],
        required_attributes=[
            "model_class"
        ],
        step_class="CreateModelStep",
        sagemaker_objects=[
            "Model",
            "ModelPackage"
        ],
        validation_rules={
            "max_containers": 15,
            "required_model_types": ["Model", "PipelineModel"]
        }
    ),
    
    "Tuning": StepTypeRequirements(
        required_methods=[
            "_create_tuner",
            "_get_tuning_inputs",
            "_get_hyperparameter_ranges"
        ],
        optional_methods=[
            "_get_objective_metric",
            "_get_tuning_strategy",
            "_get_early_stopping_config"
        ],
        required_attributes=[
            "tuner_class"
        ],
        step_class="TuningStep",
        sagemaker_objects=[
            "HyperparameterTuner",
            "TrainingInput"
        ],
        validation_rules={
            "max_parallel_jobs": 100,
            "max_training_jobs": 500,
            "valid_strategies": ["Bayesian", "Random", "Hyperband"]
        }
    ),
    
    "Lambda": StepTypeRequirements(
        required_methods=[
            "_create_lambda_function",
            "_get_lambda_inputs"
        ],
        optional_methods=[
            "_get_lambda_outputs",
            "_get_function_timeout",
            "_get_function_memory"
        ],
        required_attributes=[
            "lambda_function_arn"
        ],
        step_class="LambdaStep",
        sagemaker_objects=[
            "Lambda"
        ],
        validation_rules={
            "max_timeout": 900,  # 15 minutes
            "max_memory": 10240,  # 10GB
            "min_memory": 128
        }
    ),
    
    "Callback": StepTypeRequirements(
        required_methods=[
            "_get_sqs_queue_url",
            "_get_callback_inputs"
        ],
        optional_methods=[
            "_get_callback_outputs",
            "_get_callback_timeout"
        ],
        required_attributes=[
            "sqs_queue_url"
        ],
        step_class="CallbackStep",
        sagemaker_objects=[
            "CallbackOutput"
        ],
        validation_rules={
            "max_timeout": 86400  # 24 hours
        }
    ),
    
    "Condition": StepTypeRequirements(
        required_methods=[
            "_get_conditions",
            "_get_if_steps",
            "_get_else_steps"
        ],
        optional_methods=[
            "_get_condition_logic"
        ],
        required_attributes=[
            "condition_type"
        ],
        step_class="ConditionStep",
        sagemaker_objects=[
            "Condition",
            "ConditionEquals",
            "ConditionGreaterThan",
            "ConditionLessThan"
        ],
        validation_rules={
            "valid_condition_types": ["Equals", "GreaterThan", "LessThan", "In", "Not"]
        }
    ),
    
    "Fail": StepTypeRequirements(
        required_methods=[
            "_get_error_message"
        ],
        optional_methods=[
            "_get_failure_reason"
        ],
        required_attributes=[],
        step_class="FailStep",
        sagemaker_objects=[],
        validation_rules={
            "max_error_message_length": 1024
        }
    ),
    
    "EMR": StepTypeRequirements(
        required_methods=[
            "_get_cluster_id",
            "_get_step_config"
        ],
        optional_methods=[
            "_get_cluster_config",
            "_get_execution_role_arn"
        ],
        required_attributes=[
            "cluster_id"
        ],
        step_class="EMRStep",
        sagemaker_objects=[
            "EMRStepConfig"
        ],
        validation_rules={
            "valid_step_types": ["HadoopJarStep", "SparkStep", "HiveStep"]
        }
    ),
    
    "AutoML": StepTypeRequirements(
        required_methods=[
            "_create_automl_job",
            "_get_automl_inputs"
        ],
        optional_methods=[
            "_get_target_attribute",
            "_get_problem_type",
            "_get_automl_config"
        ],
        required_attributes=[
            "automl_job_class"
        ],
        step_class="AutoMLStep",
        sagemaker_objects=[
            "AutoML",
            "AutoMLInput"
        ],
        validation_rules={
            "valid_problem_types": ["BinaryClassification", "MulticlassClassification", "Regression"],
            "max_candidates": 250
        }
    ),
    
    "NotebookJob": StepTypeRequirements(
        required_methods=[
            "_get_input_notebook",
            "_get_image_uri",
            "_get_kernel_name"
        ],
        optional_methods=[
            "_get_parameters",
            "_get_environment_variables",
            "_get_initialization_script"
        ],
        required_attributes=[
            "notebook_path",
            "image_uri",
            "kernel_name"
        ],
        step_class="NotebookJobStep",
        sagemaker_objects=[],
        validation_rules={
            "valid_kernels": ["python3", "conda_python3", "r", "scala"],
            "max_runtime": 172800  # 48 hours
        }
    )
}


# Step type variant mapping - will be populated when test classes are imported
STEP_TYPE_VARIANT_MAP: Dict[str, Type] = {}


# Reference examples for each step type pattern
STEP_TYPE_EXAMPLES: Dict[str, Dict[str, List[str]]] = {
    "Processing": {
        "standard_patterns": [
            "builder_tabular_preprocessing_step",
            "builder_package_step"
        ],
        "custom_package_patterns": [
            "builder_model_eval_step_xgboost"
        ],
        "custom_step_patterns": [
            "builder_data_load_step_cradle"
        ]
    },
    "Training": {
        "standard_patterns": [
            "builder_training_step_xgboost"
        ]
    },
    "Transform": {
        "standard_patterns": [
            "builder_batch_transform_step"
        ]
    },
    "CreateModel": {
        "standard_patterns": [
            "builder_model_step_xgboost",
            "builder_model_step_pytorch"
        ]
    },
    "RegisterModel": {
        "custom_step_patterns": [
            "builder_registration_step"
        ]
    }
}


# Custom step detection patterns
CUSTOM_STEP_DETECTION: Dict[str, str] = {
    "CradleDataLoadingStep": "basic_interface_only",
    "MimsModelRegistrationProcessingStep": "basic_interface_only"
}


# Framework-specific processor mappings
FRAMEWORK_PROCESSOR_MAP: Dict[str, List[str]] = {
    "sklearn": ["SKLearnProcessor"],
    "xgboost": ["XGBoostProcessor"],
    "pytorch": ["PyTorchProcessor"],
    "tensorflow": ["TensorFlowProcessor"],
    "huggingface": ["HuggingFaceProcessor"],
    "spark": ["SparkMLProcessor"]
}


def register_step_type_variant(step_type: str, variant_class: Type) -> None:
    """
    Register a test variant class for a specific step type.
    
    Args:
        step_type: The SageMaker step type (e.g., "Processing", "Training")
        variant_class: The test variant class to register
    """
    STEP_TYPE_VARIANT_MAP[step_type] = variant_class


def get_step_type_variant(step_type: str) -> Optional[Type]:
    """
    Get the test variant class for a specific step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        The test variant class if registered, None otherwise
    """
    return STEP_TYPE_VARIANT_MAP.get(step_type)


def get_step_type_requirements(step_type: str) -> Optional[StepTypeRequirements]:
    """
    Get the requirements specification for a specific step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        The requirements specification if defined, None otherwise
    """
    return STEP_TYPE_REQUIREMENTS.get(step_type)


def get_all_step_types() -> List[str]:
    """
    Get all registered step types.
    
    Returns:
        List of all step type names
    """
    return list(STEP_TYPE_REQUIREMENTS.keys())


def validate_step_type(step_type: str) -> bool:
    """
    Validate if a step type is supported.
    
    Args:
        step_type: The SageMaker step type to validate
        
    Returns:
        True if step type is supported, False otherwise
    """
    return step_type in STEP_TYPE_REQUIREMENTS


def get_step_type_examples(step_type: str) -> Optional[Dict[str, List[str]]]:
    """
    Get reference examples for a specific step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        Dictionary of example patterns if available, None otherwise
    """
    return STEP_TYPE_EXAMPLES.get(step_type)


def is_custom_step(step_class_name: str) -> bool:
    """
    Check if a step class is a custom step that requires basic interface testing only.
    
    Args:
        step_class_name: The name of the step class
        
    Returns:
        True if it's a custom step, False otherwise
    """
    return step_class_name in CUSTOM_STEP_DETECTION


def get_custom_step_test_level(step_class_name: str) -> Optional[str]:
    """
    Get the test level for a custom step.
    
    Args:
        step_class_name: The name of the step class
        
    Returns:
        Test level string if it's a custom step, None otherwise
    """
    return CUSTOM_STEP_DETECTION.get(step_class_name)


def detect_framework_from_processor(processor_class_name: str) -> Optional[str]:
    """
    Detect the framework based on processor class name.
    
    Args:
        processor_class_name: The name of the processor class
        
    Returns:
        Framework name if detected, None otherwise
    """
    for framework, processors in FRAMEWORK_PROCESSOR_MAP.items():
        if processor_class_name in processors:
            return framework
    return None


def get_test_pattern_for_builder(builder_class_name: str, step_type: str) -> str:
    """
    Determine the appropriate test pattern for a step builder.
    
    Args:
        builder_class_name: The name of the step builder class
        step_type: The SageMaker step type
        
    Returns:
        Test pattern string: "standard", "custom_package", "custom_step", or "unknown"
    """
    # Check if it's a custom step first
    if is_custom_step(builder_class_name):
        return "custom_step"
    
    # Get examples for the step type
    examples = get_step_type_examples(step_type)
    if not examples:
        return "unknown"
    
    # Check each pattern category
    for pattern_type, pattern_examples in examples.items():
        if builder_class_name in pattern_examples:
            if pattern_type == "standard_patterns":
                return "standard"
            elif pattern_type == "custom_package_patterns":
                return "custom_package"
            elif pattern_type == "custom_step_patterns":
                return "custom_step"
    
    # Default to standard if not found in examples but step type is known
    return "standard"


def should_run_advanced_tests(builder_class_name: str, step_type: str) -> bool:
    """
    Determine if advanced tests should be run for a step builder.
    
    Args:
        builder_class_name: The name of the step builder class
        step_type: The SageMaker step type
        
    Returns:
        True if advanced tests should be run, False for basic interface only
    """
    test_pattern = get_test_pattern_for_builder(builder_class_name, step_type)
    
    # Custom steps should only run basic interface tests
    if test_pattern == "custom_step":
        return False
    
    # All other patterns should run advanced tests
    return True


def get_reference_examples_for_pattern(step_type: str, pattern: str) -> List[str]:
    """
    Get reference examples for a specific step type and pattern.
    
    Args:
        step_type: The SageMaker step type
        pattern: The pattern type ("standard", "custom_package", "custom_step")
        
    Returns:
        List of reference example names
    """
    examples = get_step_type_examples(step_type)
    if not examples:
        return []
    
    pattern_key = f"{pattern}_patterns"
    return examples.get(pattern_key, [])


# Default variant registration function - to be called during module initialization
def _register_default_variants():
    """Register default test variant classes when they become available."""
    try:
        # Import will be done when the actual test variant classes are implemented
        # from cursus.validation.builders.variants.processing import ProcessingStepBuilderTest
        # from cursus.validation.builders.variants.training import TrainingStepBuilderTest
        # from cursus.validation.builders.variants.transform import TransformStepBuilderTest
        # from cursus.validation.builders.variants.create_model import CreateModelStepBuilderTest
        # from cursus.validation.builders.variants.tuning import TuningStepBuilderTest
        # from cursus.validation.builders.variants.lambda_step import LambdaStepBuilderTest
        # from cursus.validation.builders.variants.callback import CallbackStepBuilderTest
        # from cursus.validation.builders.variants.condition import ConditionStepBuilderTest
        # from cursus.validation.builders.variants.fail import FailStepBuilderTest
        # from cursus.validation.builders.variants.emr import EMRStepBuilderTest
        # from cursus.validation.builders.variants.automl import AutoMLStepBuilderTest
        # from cursus.validation.builders.variants.notebook_job import NotebookJobStepBuilderTest
        
        # Register variants when classes are available
        # register_step_type_variant("Processing", ProcessingStepBuilderTest)
        # register_step_type_variant("Training", TrainingStepBuilderTest)
        # register_step_type_variant("Transform", TransformStepBuilderTest)
        # register_step_type_variant("CreateModel", CreateModelStepBuilderTest)
        # register_step_type_variant("Tuning", TuningStepBuilderTest)
        # register_step_type_variant("Lambda", LambdaStepBuilderTest)
        # register_step_type_variant("Callback", CallbackStepBuilderTest)
        # register_step_type_variant("Condition", ConditionStepBuilderTest)
        # register_step_type_variant("Fail", FailStepBuilderTest)
        # register_step_type_variant("EMR", EMRStepBuilderTest)
        # register_step_type_variant("AutoML", AutoMLStepBuilderTest)
        # register_step_type_variant("NotebookJob", NotebookJobStepBuilderTest)
        
        pass  # Placeholder until variant classes are implemented
        
    except ImportError:
        # Variant classes not yet implemented - this is expected during initial setup
        pass


# Initialize default variants
_register_default_variants()


# Export public interface
__all__ = [
    "StepTypeRequirements",
    "STEP_TYPE_REQUIREMENTS", 
    "STEP_TYPE_VARIANT_MAP",
    "STEP_TYPE_EXAMPLES",
    "CUSTOM_STEP_DETECTION",
    "FRAMEWORK_PROCESSOR_MAP",
    "register_step_type_variant",
    "get_step_type_variant",
    "get_step_type_requirements",
    "get_all_step_types",
    "validate_step_type",
    "get_step_type_examples",
    "is_custom_step",
    "get_custom_step_test_level",
    "detect_framework_from_processor",
    "get_test_pattern_for_builder",
    "should_run_advanced_tests",
    "get_reference_examples_for_pattern"
]
