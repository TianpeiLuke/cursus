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

.. automodule:: cursus.core.assembler
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Assembler
~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.assembler.pipeline_assembler
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Template Base
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.core.assembler.pipeline_template_base
   :members:
   :undoc-members:
   :show-inheritance:

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
