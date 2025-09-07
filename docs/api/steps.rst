Pipeline Steps
==============

The steps module provides comprehensive step builders, configurations, and contracts for creating SageMaker pipeline steps in the Cursus framework.

.. currentmodule:: cursus.steps

Overview
--------

The steps system is organized into several key components:

- **Builders** (:mod:`cursus.steps.builders`): Step builder classes that create SageMaker pipeline steps
- **Configurations** (:mod:`cursus.steps.configs`): Configuration classes for step parameters and settings
- **Contracts** (:mod:`cursus.steps.contracts`): Script contracts that define step requirements and validation
- **Specifications** (:mod:`cursus.steps.specs`): Step specifications for dependency resolution
- **Hyperparameters** (:mod:`cursus.steps.hyperparams`): Hyperparameter management for ML steps
- **Scripts** (:mod:`cursus.steps.scripts`): Training and processing scripts for step execution

Step Builders (:mod:`cursus.steps.builders`)
--------------------------------------------

.. automodule:: cursus.steps.builders
   :members:
   :undoc-members:
   :show-inheritance:

Training Step Builders
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.builders.builder_xgboost_training_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.builders.builder_pytorch_training_step
   :members:
   :undoc-members:
   :show-inheritance:

Processing Step Builders
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.builders.builder_tabular_preprocessing_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.builders.builder_batch_transform_step
   :members:
   :undoc-members:
   :show-inheritance:

Model Step Builders
~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.builders.builder_xgboost_model_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.builders.builder_pytorch_model_step
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Step Builders
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.builders.builder_xgboost_model_eval_step
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Step Builders
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.builders.builder_currency_conversion_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.builders.builder_risk_table_mapping_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.builders.builder_package_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.builders.builder_payload_step
   :members:
   :undoc-members:
   :show-inheritance:

Step Configurations (:mod:`cursus.steps.configs`)
-------------------------------------------------

.. automodule:: cursus.steps.configs
   :members:
   :undoc-members:
   :show-inheritance:

Training Configurations
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.configs.config_xgboost_training_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.configs.config_pytorch_training_step
   :members:
   :undoc-members:
   :show-inheritance:

Processing Configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.configs.config_tabular_preprocessing_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.configs.config_batch_transform_step
   :members:
   :undoc-members:
   :show-inheritance:

Model Configurations
~~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.steps.configs.config_xgboost_model_step
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cursus.steps.configs.config_pytorch_model_step
   :members:
   :undoc-members:
   :show-inheritance:

Step Contracts (:mod:`cursus.steps.contracts`)
----------------------------------------------

.. automodule:: cursus.steps.contracts
   :members:
   :undoc-members:
   :show-inheritance:

Step Specifications (:mod:`cursus.steps.specs`)
-----------------------------------------------

.. automodule:: cursus.steps.specs
   :members:
   :undoc-members:
   :show-inheritance:

Hyperparameters (:mod:`cursus.steps.hyperparams`)
-------------------------------------------------

.. automodule:: cursus.steps.hyperparams
   :members:
   :undoc-members:
   :show-inheritance:

Scripts (:mod:`cursus.steps.scripts`)
-------------------------------------

.. automodule:: cursus.steps.scripts
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Creating a Training Step
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.steps.builders import XGBoostTrainingStepBuilder
   from cursus.steps.configs import XGBoostTrainingConfig
   
   # Configure XGBoost training step
   config = XGBoostTrainingConfig(
       input_path="s3://my-bucket/preprocessed-data/",
       output_path="s3://my-bucket/model-artifacts/",
       training_instance_type="ml.m5.xlarge",
       training_instance_count=1,
       hyperparameters={
           "max_depth": 6,
           "eta": 0.3,
           "objective": "binary:logistic",
           "num_round": 100
       }
   )
   
   # Create the step builder
   builder = XGBoostTrainingStepBuilder(config=config)
   training_step = builder.create_step()

Creating a Processing Step
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.steps.builders import TabularPreprocessingStepBuilder
   from cursus.steps.configs import TabularPreprocessingConfig
   
   # Configure preprocessing step
   config = TabularPreprocessingConfig(
       input_path="s3://my-bucket/raw-data/",
       output_path="s3://my-bucket/processed-data/",
       processing_instance_type="ml.m5.large",
       processing_instance_count=1
   )
   
   # Create the step builder
   builder = TabularPreprocessingStepBuilder(config=config)
   preprocessing_step = builder.create_step()

Creating a Batch Transform Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.steps.builders import BatchTransformStepBuilder
   from cursus.steps.configs import BatchTransformStepConfig
   
   # Configure batch transform step
   config = BatchTransformStepConfig(
       model_name="my-trained-model",
       input_path="s3://my-bucket/inference-data/",
       output_path="s3://my-bucket/predictions/",
       transform_instance_type="ml.m5.large",
       transform_instance_count=2
   )
   
   # Create the step builder
   builder = BatchTransformStepBuilder(config=config)
   transform_step = builder.create_step()

Using Step Contracts
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.steps.contracts import XGBoostTrainingContract
   
   # Create and validate a step contract
   contract = XGBoostTrainingContract()
   
   # Validate step configuration against contract
   validation_result = contract.validate(config)
   
   if validation_result.is_valid:
       print("Step configuration is valid")
   else:
       print(f"Validation errors: {validation_result.errors}")

Working with Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.steps.hyperparams import XGBoostModelHyperparameters
   
   # Create hyperparameter configuration
   hyperparams = XGBoostModelHyperparameters(
       id_name="customer_id",
       label_name="target",
       full_field_list=["customer_id", "target", "feature1", "feature2"],
       tab_field_list=["feature1", "feature2"],
       cat_field_list=[],
       xgb_params={
           "objective": "binary:logistic",
           "max_depth": 6,
           "eta": 0.3,
           "num_round": 100
       }
   )
   
   # Use with training configuration
   training_config = XGBoostTrainingConfig(
       input_path="s3://my-bucket/data/",
       output_path="s3://my-bucket/models/",
       hyperparameters=hyperparams
   )

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cursus.api import PipelineDAG
   from cursus.core.compiler import compile_dag_to_pipeline
   
   # Create a complete pipeline with multiple steps
   dag = PipelineDAG()
   dag.add_node("preprocessing")
   dag.add_node("training")
   dag.add_node("evaluation")
   dag.add_edge("preprocessing", "training")
   dag.add_edge("training", "evaluation")
   
   # Configure all steps
   step_configs = {
       "preprocessing": preprocessing_config,
       "training": training_config,
       "evaluation": evaluation_config
   }
   
   # Compile to SageMaker pipeline
   pipeline = compile_dag_to_pipeline(
       dag,
       pipeline_name="ml-pipeline",
       step_configs=step_configs
   )

See Also
--------

- :doc:`../guides/developer_guide/adding_new_pipeline_step` - Guide for creating custom steps
- :doc:`../guides/developer_guide/step_builder` - Step builder development guide
- :doc:`registry` - Registry system for step management
- :doc:`validation` - Validation framework integration
- :doc:`../design/step_builders` - Step builder architecture design
