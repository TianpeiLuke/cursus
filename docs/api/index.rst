API Reference
=============

This section provides comprehensive API documentation for all Cursus modules, automatically generated from the source code docstrings.

Overview
--------

The Cursus API is organized into several key modules:

- **Core Framework** (:doc:`core`): Core compilation and configuration engine
- **Public API** (:doc:`api`): High-level interfaces for pipeline creation
- **CLI Interface** (:doc:`cli`): Command-line tools and utilities
- **Pipeline Steps** (:doc:`steps`): Step builders, configurations, contracts, and specifications
- **Data Processing** (:doc:`processing`): Data processing components
- **Registry System** (:doc:`registry`): Component discovery and management
- **Validation Framework** (:doc:`validation`): Pipeline validation and testing
- **Pipeline Catalog** (:doc:`pipeline_catalog`): Pre-built pipeline templates
- **Workspace Management** (:doc:`workspace`): Multi-environment support
- **MODS Integration** (:doc:`mods`): MODS-specific implementations

Quick Reference
---------------

**Main Entry Points**

.. currentmodule:: cursus

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compile_dag
   compile_dag_to_pipeline
   create_pipeline_from_dag
   PipelineDAGCompiler
   PipelineDAG
   EnhancedPipelineDAG

**Core Classes**

.. currentmodule:: cursus.core.base

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BasePipelineConfig
   StepSpecification
   StepBuilderBase
   ScriptContract

**Configuration Management**

.. currentmodule:: cursus.core.config_fields

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ConfigFieldCategorizer
   ConfigMerger
   CircularReferenceTracker

**Pipeline Compilation**

.. currentmodule:: cursus.core.compiler

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PipelineDAGCompiler
   DynamicPipelineTemplate
   ConfigResolver

Module Documentation
--------------------

.. toctree::
   :maxdepth: 2

   core
   api
   cli
   steps
   processing
   registry
   validation
   pipeline_catalog
   workspace
   mods

Package Structure
-----------------

The Cursus package follows a hierarchical structure designed for clarity and extensibility:

.. code-block:: text

   cursus/
   ├── __init__.py              # Main package exports
   ├── __version__.py           # Version information
   ├── api/                     # Public API interfaces
   │   └── dag/                 # DAG manipulation classes
   │       ├── base_dag.py      # Base DAG implementation
   │       ├── edge_types.py    # Edge type definitions
   │       ├── enhanced_dag.py  # Enhanced DAG features
   │       ├── pipeline_dag_resolver.py  # DAG resolution
   │       └── workspace_dag.py # Workspace-aware DAG
   ├── core/                    # Core framework components
   │   ├── base/                # Base classes and contracts
   │   │   ├── builder_base.py  # Step builder base class
   │   │   ├── config_base.py   # Configuration base class
   │   │   ├── contract_base.py # Contract validation base
   │   │   ├── enums.py         # Core enumerations
   │   │   ├── hyperparameters_base.py  # Hyperparameter base
   │   │   └── specification_base.py    # Specification base
   │   ├── assembler/           # Pipeline assembly components
   │   │   ├── pipeline_assembler.py    # Main assembler
   │   │   └── pipeline_template_base.py # Template base
   │   ├── compiler/            # Pipeline compilation logic
   │   │   ├── config_resolver.py       # Configuration resolution
   │   │   ├── dag_compiler.py          # DAG compilation
   │   │   ├── dynamic_template.py      # Dynamic templates
   │   │   ├── exceptions.py            # Compiler exceptions
   │   │   ├── name_generator.py        # Name generation
   │   │   └── validation.py            # Compilation validation
   │   └── config_fields/       # Configuration management
   │       ├── circular_reference_tracker.py  # Circular reference detection
   │       ├── config_class_detector.py       # Config class detection
   │       ├── config_class_store.py          # Config class storage
   │       ├── config_field_categorizer.py    # Field categorization
   │       ├── config_merger.py               # Configuration merging
   │       ├── constants.py                   # Configuration constants
   │       ├── cradle_config_factory.py       # Cradle config factory
   │       ├── tier_registry.py               # Tier registry
   │       └── type_aware_config_serializer.py # Type-aware serialization
   ├── cli/                     # Command-line interface
   │   ├── __main__.py          # CLI entry point
   │   ├── alignment_cli.py     # Alignment validation CLI
   │   ├── builder_test_cli.py  # Builder testing CLI
   │   ├── catalog_cli.py       # Catalog management CLI
   │   ├── registry_cli.py      # Registry management CLI
   │   ├── runtime_testing_cli.py # Runtime testing CLI
   │   ├── validation_cli.py    # Validation CLI
   │   └── workspace_cli.py     # Workspace management CLI
   ├── steps/                   # Pipeline step implementations
   │   ├── builders/            # Step builder classes
   │   │   ├── builder_batch_transform_step.py
   │   │   ├── builder_cradle_data_loading_step.py
   │   │   ├── builder_currency_conversion_step.py
   │   │   ├── builder_dummy_training_step.py
   │   │   ├── builder_model_calibration_step.py
   │   │   ├── builder_package_step.py
   │   │   ├── builder_payload_step.py
   │   │   ├── builder_pytorch_model_step.py
   │   │   ├── builder_pytorch_training_step.py
   │   │   ├── builder_registration_step.py
   │   │   ├── builder_risk_table_mapping_step.py
   │   │   ├── builder_tabular_preprocessing_step.py
   │   │   ├── builder_xgboost_model_eval_step.py
   │   │   ├── builder_xgboost_model_step.py
   │   │   ├── builder_xgboost_training_step.py
   │   │   └── s3_utils.py      # S3 utilities
   │   ├── configs/             # Step configuration classes
   │   │   ├── config_batch_transform_step.py
   │   │   ├── config_cradle_data_loading_step.py
   │   │   ├── config_currency_conversion_step.py
   │   │   ├── config_dummy_training_step.py
   │   │   ├── config_model_calibration_step.py
   │   │   ├── config_package_step.py
   │   │   ├── config_payload_step.py
   │   │   ├── config_processing_step_base.py
   │   │   ├── config_pytorch_model_step.py
   │   │   ├── config_pytorch_training_step.py
   │   │   ├── config_registration_step.py
   │   │   ├── config_risk_table_mapping_step.py
   │   │   ├── config_tabular_preprocessing_step.py
   │   │   ├── config_xgboost_model_eval_step.py
   │   │   ├── config_xgboost_model_step.py
   │   │   ├── config_xgboost_training_step.py
   │   │   └── utils.py          # Configuration utilities
   │   ├── contracts/           # Step validation contracts
   │   │   ├── contract_validator.py
   │   │   ├── cradle_data_loading_contract.py
   │   │   ├── currency_conversion_contract.py
   │   │   ├── dummy_training_contract.py
   │   │   ├── mims_registration_contract.py
   │   │   ├── model_calibration_contract.py
   │   │   ├── package_contract.py
   │   │   ├── payload_contract.py
   │   │   ├── pytorch_training_contract.py
   │   │   ├── risk_table_mapping_contract.py
   │   │   ├── tabular_preprocess_contract.py
   │   │   ├── training_script_contract.py
   │   │   ├── xgboost_model_eval_contract.py
   │   │   └── xgboost_training_contract.py
   │   ├── hyperparams/         # Hyperparameter definitions
   │   │   ├── hyperparameters_bsm.py
   │   │   └── hyperparameters_xgboost.py
   │   ├── registry/            # Step registry components
   │   ├── scripts/             # Step implementation scripts
   │   │   ├── currency_conversion.py
   │   │   ├── dummy_training.py
   │   │   ├── model_calibration.py
   │   │   ├── package.py
   │   │   ├── payload.py
   │   │   ├── risk_table_mapping.py
   │   │   ├── tabular_preprocessing.py
   │   │   ├── xgboost_model_evaluation.py
   │   │   └── xgboost_training.py
   │   └── specs/               # Step specifications
   │       ├── batch_transform_*.py     # Batch transform specs
   │       ├── cradle_data_loading_*.py # Data loading specs
   │       ├── currency_conversion_*.py # Currency conversion specs
   │       ├── dummy_training_spec.py
   │       ├── model_calibration_*.py   # Model calibration specs
   │       ├── package_spec.py
   │       ├── payload_spec.py
   │       ├── pytorch_*.py             # PyTorch specs
   │       ├── registration_spec.py
   │       ├── risk_table_mapping_*.py  # Risk table mapping specs
   │       ├── tabular_preprocessing_*.py # Preprocessing specs
   │       ├── xgboost_model_eval_spec.py
   │       ├── xgboost_model_spec.py
   │       └── xgboost_training_spec.py
   ├── processing/              # Data processing components
   │   ├── bert_tokenize_processor.py
   │   ├── bsm_dataloader.py
   │   ├── bsm_datasets.py
   │   ├── bsm_processor.py
   │   ├── categorical_label_processor.py
   │   ├── cs_processor.py
   │   ├── gensim_tokenize_processor.py
   │   ├── multiclass_label_processor.py
   │   ├── numerical_binning_processor.py
   │   ├── numerical_imputation_processor.py
   │   ├── processors.py        # Main processor classes
   │   └── risk_table_processor.py
   ├── registry/                # Component registries
   │   ├── builder_registry.py  # Step builder registry
   │   ├── exceptions.py        # Registry exceptions
   │   ├── hyperparameter_registry.py # Hyperparameter registry
   │   ├── step_names_original.py
   │   ├── step_names.py        # Step name definitions
   │   ├── step_type_test_variants.py
   │   ├── validation_utils.py  # Registry validation utilities
   │   └── hybrid/              # Hybrid registry implementation
   │       ├── manager.py       # Registry manager
   │       ├── models.py        # Registry models
   │       ├── setup.py         # Registry setup
   │       └── utils.py         # Registry utilities
   ├── validation/              # Validation frameworks
   │   ├── simple_integration.py # Simple integration tests
   │   ├── alignment/           # Contract-specification alignment
   │   │   ├── alignment_reporter.py
   │   │   ├── alignment_scorer.py
   │   │   ├── alignment_utils.py
   │   │   ├── builder_config_alignment.py
   │   │   ├── contract_spec_alignment.py
   │   │   ├── core_models.py
   │   │   ├── dependency_classifier.py
   │   │   ├── enhanced_reporter.py
   │   │   ├── file_resolver.py
   │   │   ├── framework_patterns.py
   │   │   ├── level3_validation_config.py
   │   │   ├── property_path_validator.py
   │   │   ├── script_analysis_models.py
   │   │   ├── script_contract_alignment.py
   │   │   ├── smart_spec_selector.py
   │   │   ├── spec_dependency_alignment.py
   │   │   ├── step_type_detection.py
   │   │   ├── step_type_enhancement_router.py
   │   │   ├── testability_validator.py
   │   │   ├── unified_alignment_tester.py
   │   │   ├── utils.py
   │   │   ├── workflow_integration.py
   │   │   ├── analyzers/       # Static analysis components
   │   │   ├── discovery/       # Contract discovery
   │   │   ├── loaders/         # Contract and spec loaders
   │   │   ├── orchestration/   # Validation orchestration
   │   │   ├── patterns/        # Pattern recognition
   │   │   ├── processors/      # Spec file processors
   │   │   ├── static_analysis/ # Static analysis tools
   │   │   ├── step_type_enhancers/ # Step type enhancers
   │   │   └── validators/      # Validation components
   │   ├── builders/            # Builder validation
   │   │   ├── base_test.py
   │   │   ├── builder_reporter.py
   │   │   ├── example_enhanced_usage.py
   │   │   ├── example_usage.py
   │   │   ├── generic_test.py
   │   │   ├── integration_tests.py
   │   │   ├── interface_tests.py
   │   │   ├── mock_factory.py
   │   │   ├── registry_discovery.py
   │   │   ├── sagemaker_step_type_validator.py
   │   │   ├── scoring.py
   │   │   ├── specification_tests.py
   │   │   ├── step_creation_tests.py
   │   │   ├── step_info_detector.py
   │   │   ├── test_factory.py
   │   │   ├── universal_test.py
   │   │   └── variants/        # Step type specific tests
   │   ├── interface/           # Interface validation
   │   │   └── interface_standard_validator.py
   │   ├── naming/              # Naming validation
   │   │   └── naming_standard_validator.py
   │   ├── runtime/             # Runtime validation
   │   │   ├── runtime_models.py
   │   │   ├── runtime_spec_builder.py
   │   │   └── runtime_testing.py
   │   └── shared/              # Shared validation utilities
   │       └── chart_utils.py
   ├── pipeline_catalog/        # Pre-built pipeline templates
   │   ├── catalog_index.json   # Catalog index
   │   ├── utils.py             # Catalog utilities
   │   ├── mods_pipelines/      # MODS-specific pipelines
   │   │   ├── dummy_mods_e2e_basic.py
   │   │   ├── pytorch_mods_e2e_standard.py
   │   │   ├── pytorch_mods_training_basic.py
   │   │   ├── xgb_mods_e2e_comprehensive.py
   │   │   ├── xgb_mods_training_calibrated.py
   │   │   ├── xgb_mods_training_evaluation.py
   │   │   └── xgb_mods_training_simple.py
   │   ├── pipelines/           # Standard pipelines
   │   │   ├── dummy_e2e_basic.py
   │   │   ├── pytorch_e2e_standard.py
   │   │   ├── pytorch_training_basic.py
   │   │   ├── xgb_e2e_comprehensive.py
   │   │   ├── xgb_training_calibrated.py
   │   │   ├── xgb_training_evaluation.py
   │   │   └── xgb_training_simple.py
   │   ├── shared_dags/         # Shared DAG components
   │   │   ├── enhanced_metadata.py
   │   │   ├── registry_sync.py
   │   │   ├── dummy/           # Dummy pipeline DAGs
   │   │   ├── pytorch/         # PyTorch pipeline DAGs
   │   │   └── xgboost/         # XGBoost pipeline DAGs
   │   └── utils/               # Catalog utilities
   │       ├── catalog_registry.py
   │       ├── connection_traverser.py
   │       ├── recommendation_engine.py
   │       ├── registry_validator.py
   │       └── tag_discovery.py
   ├── workspace/               # Workspace management
   │   ├── api.py               # Workspace API
   │   ├── templates.py         # Workspace templates
   │   ├── utils.py             # Workspace utilities
   │   ├── core/                # Core workspace components
   │   │   ├── assembler.py
   │   │   ├── compiler.py
   │   │   ├── config.py
   │   │   ├── discovery.py
   │   │   ├── integration.py
   │   │   ├── isolation.py
   │   │   ├── lifecycle.py
   │   │   ├── manager.py
   │   │   └── registry.py
   │   ├── quality/             # Quality assurance
   │   │   ├── documentation_validator.py
   │   │   ├── quality_monitor.py
   │   │   └── user_experience_validator.py
   │   └── validation/          # Workspace validation
   │       ├── base_validation_result.py
   │       ├── cross_workspace_validator.py
   │       ├── legacy_adapters.py
   │       ├── unified_report_generator.py
   │       ├── unified_result_structures.py
   │       ├── unified_validation_core.py
   │       ├── workspace_alignment_tester.py
   │       ├── workspace_builder_test.py
   │       ├── workspace_file_resolver.py
   │       ├── workspace_isolation.py
   │       ├── workspace_manager.py
   │       ├── workspace_module_loader.py
   │       ├── workspace_test_manager.py
   │       └── workspace_type_detector.py
   └── mods/                    # MODS-specific implementations
       └── compiler/            # MODS compiler components
           └── mods_dag_compiler.py

Usage Patterns
--------------

**Basic Pipeline Creation**

.. code-block:: python

   from cursus import PipelineDAG, compile_dag_to_pipeline
   
   # Create DAG
   dag = PipelineDAG()
   dag.add_node("preprocessing")
   dag.add_node("training")
   dag.add_node("evaluation")
   dag.add_edge("preprocessing", "training")
   dag.add_edge("training", "evaluation")
   
   # Compile to pipeline
   pipeline = compile_dag_to_pipeline(dag, "ml-pipeline")

**Advanced Configuration**

.. code-block:: python

   from cursus.core.compiler import PipelineDAGCompiler
   from cursus.core.config_fields import ConfigMerger
   
   # Advanced compilation with custom configuration
   compiler = PipelineDAGCompiler()
   config_merger = ConfigMerger()
   
   pipeline = compiler.compile(
       dag, 
       pipeline_name="advanced-pipeline",
       config_merger=config_merger
   )

**Step Builder Usage**

.. code-block:: python

   from cursus.steps.builders import BuilderXGBoostTrainingStep
   from cursus.steps.configs import ConfigXGBoostTrainingStep
   
   # Create step configuration
   config = ConfigXGBoostTrainingStep(
       model_name="fraud-detector",
       hyperparameters={"max_depth": 6, "eta": 0.3}
   )
   
   # Build step
   builder = BuilderXGBoostTrainingStep()
   step = builder.build(config)

**Validation Framework**

.. code-block:: python

   from cursus.validation.alignment import UnifiedAlignmentTester
   from cursus.validation.builders import UniversalStepBuilderTester
   
   # Validate alignment
   alignment_tester = UnifiedAlignmentTester()
   results = alignment_tester.validate_step_alignment("xgboost_training")
   
   # Test step builders
   builder_tester = UniversalStepBuilderTester()
   test_results = builder_tester.test_builder("xgboost_training")

Type Annotations
----------------

Cursus makes extensive use of Python type annotations to provide clear API contracts and enable better IDE support. All public APIs include comprehensive type hints.

.. code-block:: python

   from typing import Dict, List, Optional, Union
   from cursus.api.dag import PipelineDAG
   from cursus.core.base import BasePipelineConfig
   
   def compile_dag_to_pipeline(
       dag: PipelineDAG,
       pipeline_name: Optional[str] = None,
       config: Optional[BasePipelineConfig] = None,
       **kwargs: Dict[str, Any]
   ) -> sagemaker.workflow.pipeline.Pipeline:
       """Compile DAG to SageMaker pipeline with type safety."""

Error Handling
--------------

Cursus provides structured exception handling with specific exception types for different error conditions:

.. currentmodule:: cursus.core.compiler

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CompilationError
   ValidationError
   ConfigurationError

.. currentmodule:: cursus.registry

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RegistryError
   BuilderNotFoundError

Extension Points
----------------

Cursus is designed for extensibility. Key extension points include:

**Custom Step Builders**

.. code-block:: python

   from cursus.core.base import StepBuilderBase
   from cursus.steps.configs import BasePipelineConfig
   
   class CustomStepBuilder(StepBuilderBase):
       def build(self, config: BasePipelineConfig) -> sagemaker.workflow.steps.Step:
           # Custom step implementation
           pass

**Custom Validators**

.. code-block:: python

   from cursus.validation.alignment.validators import BaseValidator
   
   class CustomValidator(BaseValidator):
       def validate(self, component) -> ValidationResult:
           # Custom validation logic
           pass

**Custom Configuration Classes**

.. code-block:: python

   from cursus.core.base import BasePipelineConfig
   
   class CustomConfig(BasePipelineConfig):
       custom_parameter: str
       advanced_settings: Dict[str, Any]

Performance Considerations
--------------------------

- **Lazy Loading**: Many components use lazy loading to minimize startup time
- **Caching**: Configuration resolution and validation results are cached
- **Parallel Processing**: Where applicable, operations are parallelized
- **Memory Management**: Large objects are cleaned up automatically

See the individual module documentation for detailed API information and usage examples.
