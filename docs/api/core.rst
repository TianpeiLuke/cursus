Core Framework
==============

The core framework provides the foundational components for pipeline compilation, configuration management, and base abstractions.

.. currentmodule:: cursus.core

Overview
--------

The core framework is organized into several key submodules:

- **Base Classes** (:mod:`cursus.core.base`): Foundational abstractions and contracts
- **Compiler** (:mod:`cursus.core.compiler`): Pipeline compilation and template generation
- **Assembler** (:mod:`cursus.core.assembler`): Pipeline assembly components
- **Configuration Fields** (:mod:`cursus.core.config_fields`): Advanced configuration management

Base Classes (:mod:`cursus.core.base`)
--------------------------------------

.. automodule:: cursus.core.base
   :members:
   :undoc-members:
   :show-inheritance:

Base Configuration
~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.base.config_base
   :members:
   :undoc-members:
   :show-inheritance:

Step Builder Base
~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.base.builder_base
   :members:
   :undoc-members:
   :show-inheritance:

Contract Base
~~~~~~~~~~~~~

.. automodule:: cursus.core.base.contract_base
   :members:
   :undoc-members:
   :show-inheritance:

Specification Base
~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.base.specification_base
   :members:
   :undoc-members:
   :show-inheritance:

Hyperparameters Base
~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.base.hyperparameters_base
   :members:
   :undoc-members:
   :show-inheritance:

Core Enumerations
~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.base.enums
   :members:
   :undoc-members:
   :show-inheritance:

Compiler (:mod:`cursus.core.compiler`)
--------------------------------------

.. automodule:: cursus.core.compiler
   :members:
   :undoc-members:
   :show-inheritance:

DAG Compiler
~~~~~~~~~~~~

.. automodule:: cursus.core.compiler.dag_compiler
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Resolver
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.compiler.config_resolver
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Template
~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.compiler.dynamic_template
   :members:
   :undoc-members:
   :show-inheritance:

Name Generator
~~~~~~~~~~~~~~

.. automodule:: cursus.core.compiler.name_generator
   :members:
   :undoc-members:
   :show-inheritance:

Validation
~~~~~~~~~~

.. automodule:: cursus.core.compiler.validation
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
~~~~~~~~~~

.. automodule:: cursus.core.compiler.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Assembler (:mod:`cursus.core.assembler`)
----------------------------------------

The Pipeline Assembler is a specification-driven system for creating SageMaker pipelines. It provides a declarative approach to defining pipeline structure and leverages intelligent dependency resolution to automatically connect steps, eliminating the need for manual wiring of inputs and outputs.

.. automodule:: cursus.core.assembler
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Template Base
~~~~~~~~~~~~~~~~~~~~~~

The Pipeline Template Base is an abstract base class that provides a consistent structure and common functionality for all pipeline templates. It handles configuration loading, component lifecycle management, and pipeline generation.

**Key Features:**
- **Abstract Template Structure**: Enforces consistent implementation across pipeline templates
- **Configuration Management**: Loads and validates pipeline configurations from JSON files
- **Component Lifecycle**: Manages dependency resolution components (registry manager, dependency resolver)
- **Factory Methods**: Provides multiple creation patterns for different use cases
- **Context Management**: Supports scoped contexts and thread-local components for thread safety

.. automodule:: cursus.core.assembler.pipeline_template_base
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Assembler
~~~~~~~~~~~~~~~~~~

The Pipeline Assembler is responsible for assembling pipeline steps using a DAG structure and specification-based dependency resolution.

**Key Features:**
- **Step Builder Management**: Initializes and manages step builders for all pipeline steps
- **Specification-Based Matching**: Uses step specifications to intelligently match inputs to outputs
- **Message Propagation**: Propagates dependency information between connected steps
- **Runtime Property Handling**: Creates SageMaker property references for step connections
- **Topological Assembly**: Instantiates steps in dependency order using topological sorting

.. automodule:: cursus.core.assembler.pipeline_assembler
   :members:
   :undoc-members:
   :show-inheritance:

Specification-Driven Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each step builder provides a specification that declares its inputs and outputs:

.. code-block:: python

   self.spec = StepSpecification(
       step_type="XGBoostTrainingStep",
       node_type=NodeType.INTERNAL,
       dependencies={
           "training_data": DependencySpec(
               logical_name="training_data",
               dependency_type=DependencyType.PROCESSING_OUTPUT,
               required=True,
               compatible_sources=["PreprocessingStep"],
               semantic_keywords=["data", "training", "processed"],
               data_type="S3Uri"
           )
       },
       outputs={
           "model_output": OutputSpec(
               logical_name="model_output",
               output_type=DependencyType.MODEL_ARTIFACTS,
               property_path="properties.ModelArtifacts.S3ModelArtifacts",
               data_type="S3Uri",
               aliases=["ModelArtifacts", "model_data"]
           )
       }
   )

**Dependency Resolution Process:**

1. **Specification Registration**: Each step registers its specification with the registry
2. **Dependency Analysis**: The dependency resolver analyzes the specifications of all steps
3. **Compatibility Scoring**: The resolver calculates compatibility scores between dependencies and outputs
4. **Message Propagation**: Messages are propagated from source steps to destination steps based on the DAG structure
5. **Property Reference Creation**: Property references are created to bridge definition-time and runtime

Creating Custom Pipeline Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Use Dynamic Pipeline Template (Recommended)**

.. code-block:: python

   from cursus.core.compiler import DynamicPipelineTemplate
   from cursus.api import PipelineDAG

   # Create your DAG structure
   dag = PipelineDAG()
   dag.add_node("CradleDataLoading_data_loading")
   dag.add_node("XGBoostTraining_training")
   dag.add_node("Package_packaging")
   dag.add_edge("CradleDataLoading_data_loading", "XGBoostTraining_training")
   dag.add_edge("XGBoostTraining_training", "Package_packaging")

   # Create dynamic template - automatically detects required config classes
   template = DynamicPipelineTemplate(
       dag=dag,
       config_path="configs/my_pipeline.json",
       sagemaker_session=sagemaker_session,
       role=execution_role
   )

   # Generate pipeline
   pipeline = template.generate_pipeline()

**Option 2: Create Custom Template Class**

.. code-block:: python

   from cursus.core.assembler import PipelineTemplateBase
   from cursus.core.base import BasePipelineConfig
   from cursus.steps.configs import (
       CradleDataLoadConfig,
       XGBoostTrainingConfig,
       PackageConfig
   )

   class MyCustomTemplate(PipelineTemplateBase):
       # Define the configuration classes expected in the config file
       CONFIG_CLASSES = {
           'BasePipelineConfig': BasePipelineConfig,
           'CradleDataLoadConfig': CradleDataLoadConfig,
           'XGBoostTrainingConfig': XGBoostTrainingConfig,
           'PackageConfig': PackageConfig,
       }
       
       def _validate_configuration(self) -> None:
           """Validate that required configurations are present."""
           required_configs = ['CradleDataLoadConfig', 'XGBoostTrainingConfig']
           for config_type in required_configs:
               if not any(isinstance(cfg, self.CONFIG_CLASSES[config_type]) 
                         for cfg in self.configs.values()):
                   raise ValueError(f"Missing required configuration: {config_type}")
       
       def _create_pipeline_dag(self) -> PipelineDAG:
           """Create the DAG structure for the pipeline."""
           dag = PipelineDAG()
           dag.add_node("data_loading")
           dag.add_node("training")
           dag.add_node("packaging")
           dag.add_edge("data_loading", "training")
           dag.add_edge("training", "packaging")
           return dag
       
       def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
           """Map step names to configuration instances."""
           config_map = {}
           
           # Find configurations by type
           for config_name, config in self.configs.items():
               if isinstance(config, CradleDataLoadConfig):
                   config_map["data_loading"] = config
               elif isinstance(config, XGBoostTrainingConfig):
                   config_map["training"] = config
               elif isinstance(config, PackageConfig):
                   config_map["packaging"] = config
                   
           return config_map
       
       def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
           """Map step types to builder classes."""
           from cursus.steps.builders import (
               CradleDataLoadingStepBuilder,
               XGBoostTrainingStepBuilder,
               PackageStepBuilder
           )
           
           return {
               "CradleDataLoading": CradleDataLoadingStepBuilder,
               "XGBoostTraining": XGBoostTrainingStepBuilder,
               "Package": PackageStepBuilder,
           }

   # Usage
   template = MyCustomTemplate(
       config_path="configs/my_pipeline.json",
       sagemaker_session=sagemaker_session,
       role=execution_role
   )

   pipeline = template.generate_pipeline()

Pipeline Assembly Process
~~~~~~~~~~~~~~~~~~~~~~~~~

The Pipeline Assembler follows a systematic approach to build SageMaker pipelines:

**1. Step Builder Initialization**

.. code-block:: python

   # Initialize step builders for all steps in the DAG
   for step_name in self.dag.nodes:
       config = self.config_map[step_name]
       step_type = CONFIG_STEP_REGISTRY.get(type(config).__name__)
       builder_cls = self.step_builder_map[step_type]
       
       builder = builder_cls(
           config=config,
           sagemaker_session=self.sagemaker_session,
           role=self.role,
           registry_manager=self._registry_manager,
           dependency_resolver=self._dependency_resolver
       )
       self.step_builders[step_name] = builder

**2. Message Propagation**

.. code-block:: python

   # Propagate messages between steps using specifications
   for src_step, dst_step in self.dag.edges:
       src_builder = self.step_builders[src_step]
       dst_builder = self.step_builders[dst_step]
       
       # Match outputs to inputs based on compatibility
       for dep_name, dep_spec in dst_builder.spec.dependencies.items():
           for out_name, out_spec in src_builder.spec.outputs.items():
               compatibility = resolver._calculate_compatibility(dep_spec, out_spec, src_builder.spec)
               if compatibility > 0.5:
                   # Store connection information
                   self.step_messages[dst_step][dep_name] = {
                       'source_step': src_step,
                       'source_output': out_name,
                       'compatibility': compatibility
                   }

**3. Step Instantiation**

.. code-block:: python

   # Instantiate steps in topological order
   build_order = self.dag.topological_sort()
   for step_name in build_order:
       builder = self.step_builders[step_name]
       
       # Extract inputs from message connections
       inputs = {}
       for input_name, message in self.step_messages[step_name].items():
           src_step = message['source_step']
           src_output = message['source_output']
           
           # Create runtime property reference
           prop_ref = PropertyReference(
               step_name=src_step,
               output_spec=src_builder.spec.get_output_by_name_or_alias(src_output)
           )
           inputs[input_name] = prop_ref.to_runtime_property(self.step_instances)
       
       # Generate outputs and create step
       outputs = self._generate_outputs(step_name)
       step = builder.create_step(inputs=inputs, outputs=outputs)
       self.step_instances[step_name] = step

Configuration Fields (:mod:`cursus.core.config_fields`)
-------------------------------------------------------

.. automodule:: cursus.core.config_fields
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Field Categorizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.config_field_categorizer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Merger
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.config_merger
   :members:
   :undoc-members:
   :show-inheritance:

Circular Reference Tracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.circular_reference_tracker
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Class Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.config_class_detector
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Class Store
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.config_class_store
   :members:
   :undoc-members:
   :show-inheritance:

Type-Aware Configuration Serializer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.type_aware_config_serializer
   :members:
   :undoc-members:
   :show-inheritance:

Cradle Configuration Factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.cradle_config_factory
   :members:
   :undoc-members:
   :show-inheritance:

Tier Registry
~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.tier_registry
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Constants
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.config_fields.constants
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Pipeline Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.core.compiler import PipelineDAGCompiler
   from cursus.api import PipelineDAG
   
   # Create a DAG
   dag = PipelineDAG()
   dag.add_node("preprocessing")
   dag.add_node("training")
   dag.add_edge("preprocessing", "training")
   
   # Compile the DAG
   compiler = PipelineDAGCompiler()
   pipeline = compiler.compile(dag, pipeline_name="example-pipeline")

Advanced Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.core.config_fields import load_configs, serialize_config
   from cursus.core.base import BasePipelineConfig
   
   # Load and serialize configurations
   config = load_configs("pipeline_config.yaml")
   serialized = serialize_config(config)
   
   # Work with base configuration
   base_config = BasePipelineConfig()
   step_config = {"hyperparameters": {"learning_rate": 0.01}}

Custom Step Builder
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.core.base import StepBuilderBase, BasePipelineConfig
   from sagemaker.workflow.steps import ProcessingStep
   
   class CustomStepBuilder(StepBuilderBase):
       """Custom step builder implementation."""
       
       def create_step(self, config: BasePipelineConfig) -> ProcessingStep:
           """Build a custom processing step."""
           # Implementation details
           return ProcessingStep(
               name=config.step_name,
               # ... other parameters
           )

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.core import ScriptContract, ValidationResult
   
   class CustomContract(ScriptContract):
       """Custom validation contract."""
       
       def validate(self, config) -> ValidationResult:
           """Validate configuration against contract."""
           errors = []
           warnings = []
           
           # Validation logic
           if not config.get("required_param"):
               errors.append("required_param is missing")
           
           return ValidationResult(
               is_valid=len(errors) == 0,
               errors=errors,
               warnings=warnings
           )

See Also
--------

- :doc:`../guides/advanced_usage` - Advanced usage patterns
- :doc:`api` - Public API interfaces  
- :doc:`steps` - Pipeline step implementations
- :doc:`validation` - Validation framework
