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
    "register_step_type_variant",
    "get_step_type_variant",
    "get_step_type_requirements",
    "get_all_step_types",
    "validate_step_type"
]
