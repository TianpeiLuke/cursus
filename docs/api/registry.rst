Registry System
===============

The registry system provides centralized management of step builders, hyperparameters, and validation utilities for the Cursus pipeline framework.

.. currentmodule:: cursus.registry

Overview
--------

The registry system consists of several key components:

- **Builder Registry** (:mod:`cursus.registry.builder_registry`): Manages step builder registration and discovery
- **Hyperparameter Registry** (:mod:`cursus.registry.hyperparameter_registry`): Manages hyperparameter configurations
- **Validation Utils** (:mod:`cursus.registry.validation_utils`): Provides validation utilities for registry operations
- **Hybrid Registry** (:mod:`cursus.registry.hybrid`): Advanced registry patterns and integrations

Builder Registry (:mod:`cursus.registry.builder_registry`)
---------------------------------------------------------

.. automodule:: cursus.registry.builder_registry
   :members:
   :undoc-members:
   :show-inheritance:

Hyperparameter Registry (:mod:`cursus.registry.hyperparameter_registry`)
------------------------------------------------------------------------

.. automodule:: cursus.registry.hyperparameter_registry
   :members:
   :undoc-members:
   :show-inheritance:

Validation Utilities (:mod:`cursus.registry.validation_utils`)
-------------------------------------------------------------

.. automodule:: cursus.registry.validation_utils
   :members:
   :undoc-members:
   :show-inheritance:

Step Names (:mod:`cursus.registry.step_names`)
----------------------------------------------

.. automodule:: cursus.registry.step_names
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions (:mod:`cursus.registry.exceptions`)
---------------------------------------------

.. automodule:: cursus.registry.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Hybrid Registry (:mod:`cursus.registry.hybrid`)
-----------------------------------------------

.. automodule:: cursus.registry.hybrid
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Registry Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.registry import get_builder_registry
   
   # Get the builder registry
   registry = get_builder_registry()
   
   # List available step types
   step_types = registry.list_step_types()
   print(f"Available step types: {step_types}")
   
   # Get a specific builder
   builder_class = registry.get_builder("XGBoostTraining")
   builder = builder_class()

Registering Custom Builders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.registry import register_builder
   from cursus.steps.builders import StepBuilderBase
   
   class CustomStepBuilder(StepBuilderBase):
       """Custom step builder implementation."""
       
       def create_step(self, config):
           # Implementation details
           pass
   
   # Register the custom builder
   register_builder("CustomStep", CustomStepBuilder)

Hyperparameter Management
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.registry import get_hyperparameter_registry
   
   # Get hyperparameter registry
   hp_registry = get_hyperparameter_registry()
   
   # Register hyperparameters for a step type
   hp_registry.register_hyperparameters(
       "XGBoostTraining",
       {
           "max_depth": {"type": "int", "range": [1, 20], "default": 6},
           "eta": {"type": "float", "range": [0.01, 1.0], "default": 0.3},
           "objective": {"type": "str", "choices": ["binary:logistic", "reg:squarederror"]}
       }
   )

Validation Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.registry.validation_utils import validate_step_configuration
   
   # Validate a step configuration
   config = {
       "step_type": "XGBoostTraining",
       "hyperparameters": {
           "max_depth": 6,
           "eta": 0.3,
           "objective": "binary:logistic"
       }
   }
   
   validation_result = validate_step_configuration(config)
   if validation_result.is_valid:
       print("Configuration is valid")
   else:
       print(f"Validation errors: {validation_result.errors}")

Advanced Registry Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.registry.hybrid import HybridRegistry
   
   # Create a hybrid registry with multiple backends
   hybrid_registry = HybridRegistry()
   
   # Add multiple registry sources
   hybrid_registry.add_source("local", local_registry)
   hybrid_registry.add_source("remote", remote_registry)
   
   # Query across all sources
   all_builders = hybrid_registry.get_all_builders()

See Also
--------

- :doc:`../guides/developer_guide/step_builder_registry_guide` - Detailed guide on using the registry system
- :doc:`steps` - Step builders and configurations
- :doc:`validation` - Validation framework integration
- :doc:`../design/registry_system` - Registry system architecture
