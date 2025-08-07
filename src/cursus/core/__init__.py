"""
AutoPipe Core module.

This module provides the core functionality for AutoPipe, including:
- Pipeline assembling and template management
- DAG compilation and configuration resolution
- Configuration field management and three-tier architecture
- Dependency resolution and specification management
- Base classes for configurations, contracts, specifications, and builders
"""

# Import from submodules
from .base import (
    DependencyType,
    NodeType,
    ScriptContract,
    ValidationResult,
    ScriptAnalyzer,
    ModelHyperparameters,
    BasePipelineConfig,
    DependencySpec,
    OutputSpec,
    StepSpecification,
    StepBuilderBase
)
from .assembler import PipelineAssembler, PipelineTemplateBase
from .compiler import (
    compile_dag_to_pipeline,
    PipelineDAGCompiler,
    StepConfigResolver,
    ValidationResult,
    ResolutionPreview,
    ConversionReport,
    ValidationEngine,
    generate_random_word,
    validate_pipeline_name,
    sanitize_pipeline_name,
    generate_pipeline_name,
    PipelineAPIError,
    ConfigurationError,
    AmbiguityError,
    ValidationError,
    ResolutionError
)

def _get_dynamic_pipeline_template():
    """Lazy import to avoid circular import issues."""
    from .compiler import DynamicPipelineTemplate
    return DynamicPipelineTemplate

# Make DynamicPipelineTemplate available through lazy loading
def __getattr__(name):
    if name == "DynamicPipelineTemplate":
        return _get_dynamic_pipeline_template()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
from .config_fields import (
    merge_and_save_configs,
    load_configs,
    serialize_config,
    deserialize_config,
    ConfigClassStore,
    register_config_class,
    CircularReferenceTracker,
    ConfigFieldTierRegistry
    # These modules are not currently available
    # DefaultValuesProvider,
    # FieldDerivationEngine,
    # DataConfig,
    # ModelConfig,
    # RegistrationConfig,
    # EssentialInputs
)
from .deps import (
    DependencyType,
    NodeType,
    DependencySpec,
    OutputSpec,
    PropertyReference,
    StepSpecification,
    SpecificationRegistry,
    RegistryManager,
    get_registry,
    get_pipeline_registry,
    get_default_registry,
    integrate_with_pipeline_builder,
    list_contexts,
    clear_context,
    get_context_stats,
    UnifiedDependencyResolver,
    DependencyResolutionError,
    create_dependency_resolver,
    SemanticMatcher,
    create_pipeline_components
)

__all__ = [
    # Base classes
    "DependencyType",
    "NodeType", 
    "ScriptContract",
    "ValidationResult",
    "ScriptAnalyzer",
    "ModelHyperparameters",
    "BasePipelineConfig",
    "DependencySpec",
    "OutputSpec", 
    "StepSpecification",
    "StepBuilderBase",
    
    # Assembler
    "PipelineAssembler",
    "PipelineTemplateBase",
    
    # Compiler
    "compile_dag_to_pipeline",
    "PipelineDAGCompiler",
    "DynamicPipelineTemplate",
    "StepConfigResolver",
    "ValidationResult",
    "ResolutionPreview",
    "ConversionReport",
    "ValidationEngine",
    "generate_random_word",
    "validate_pipeline_name",
    "sanitize_pipeline_name",
    "generate_pipeline_name",
    "PipelineAPIError",
    "ConfigurationError",
    "AmbiguityError",
    "ValidationError",
    "ResolutionError",
    
    # Config Fields
    "merge_and_save_configs",
    "load_configs",
    "serialize_config",
    "deserialize_config",
    "ConfigClassStore",
    "register_config_class",
    "CircularReferenceTracker",
    "ConfigFieldTierRegistry",
    # These modules are not currently available
    # "DefaultValuesProvider",
    # "FieldDerivationEngine",
    # "DataConfig",
    # "ModelConfig",
    # "RegistrationConfig",
    # "EssentialInputs",
    
    # Dependencies
    "DependencyType",
    "NodeType",
    "DependencySpec",
    "OutputSpec",
    "PropertyReference",
    "StepSpecification",
    "SpecificationRegistry",
    "RegistryManager",
    "get_registry",
    "get_pipeline_registry",
    "get_default_registry",
    "integrate_with_pipeline_builder",
    "list_contexts",
    "clear_context",
    "get_context_stats",
    "UnifiedDependencyResolver",
    "DependencyResolutionError",
    "create_dependency_resolver",
    "SemanticMatcher",
    "create_pipeline_components",
]
