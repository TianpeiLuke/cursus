Quick Start Guide
==================

This guide will help you get started with Cursus quickly and efficiently.

Installation
------------

Install Cursus with all dependencies:

.. code-block:: bash

   pip install cursus[all]

For a minimal installation:

.. code-block:: bash

   pip install cursus

Basic Usage
-----------

Creating Your First Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a simple example to create and compile a machine learning pipeline:

.. code-block:: python

   import cursus
   from cursus import PipelineDAG
   
   # Create a new DAG
   dag = PipelineDAG()
   
   # Add pipeline steps
   dag.add_node("data_preprocessing")
   dag.add_node("model_training")
   dag.add_node("model_evaluation")
   
   # Define dependencies
   dag.add_edge("data_preprocessing", "model_training")
   dag.add_edge("model_training", "model_evaluation")
   
   # Compile to SageMaker pipeline
   pipeline = cursus.compile_dag(dag, pipeline_name="my-first-pipeline")
   
   # Execute the pipeline
   execution = pipeline.start()
   print(f"Pipeline execution started: {execution.arn}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For more control over the compilation process:

.. code-block:: python

   from cursus.core.compiler import PipelineDAGCompiler
   from cursus.core.config_fields import ConfigMerger
   
   # Create compiler with custom configuration
   compiler = PipelineDAGCompiler()
   config_merger = ConfigMerger()
   
   # Compile with advanced options
   pipeline = compiler.compile(
       dag,
       pipeline_name="advanced-pipeline",
       config_merger=config_merger,
       enable_caching=True,
       role_arn="arn:aws:iam::123456789012:role/SageMakerRole"
   )

Working with Step Builders
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cursus provides pre-built step builders for common ML tasks:

.. code-block:: python

   from cursus.steps.builders import BuilderXGBoostTrainingStep
   from cursus.steps.configs import ConfigXGBoostTrainingStep
   
   # Configure XGBoost training step
   config = ConfigXGBoostTrainingStep(
       model_name="fraud-detector",
       hyperparameters={
           "max_depth": 6,
           "eta": 0.3,
           "objective": "binary:logistic",
           "num_round": 100
       },
       instance_type="ml.m5.xlarge",
       instance_count=1
   )
   
   # Build the step
   builder = BuilderXGBoostTrainingStep()
   training_step = builder.build(config)

Pipeline Validation
~~~~~~~~~~~~~~~~~~~

Validate your pipeline before execution:

.. code-block:: python

   from cursus.validation.alignment import UnifiedAlignmentTester
   
   # Create alignment tester
   tester = UnifiedAlignmentTester()
   
   # Validate step alignment
   results = tester.validate_step_alignment("xgboost_training")
   
   if results.is_valid:
       print("Pipeline validation passed!")
   else:
       print("Validation errors:")
       for error in results.errors:
           print(f"  - {error}")

Common Patterns
---------------

End-to-End ML Pipeline
~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of a typical ML pipeline:

.. code-block:: python

   from cursus import PipelineDAG, compile_dag_to_pipeline
   from cursus.steps.configs import (
       ConfigTabularPreprocessingStep,
       ConfigXGBoostTrainingStep,
       ConfigXGBoostModelEvalStep
   )
   
   # Create DAG
   dag = PipelineDAG()
   
   # Add steps
   dag.add_node("preprocessing")
   dag.add_node("training")
   dag.add_node("evaluation")
   
   # Define dependencies
   dag.add_edge("preprocessing", "training")
   dag.add_edge("training", "evaluation")
   
   # Configure steps
   preprocessing_config = ConfigTabularPreprocessingStep(
       input_data="s3://my-bucket/raw-data/",
       output_data="s3://my-bucket/processed-data/",
       processing_instance_type="ml.m5.large"
   )
   
   training_config = ConfigXGBoostTrainingStep(
       model_name="my-model",
       hyperparameters={"max_depth": 5, "eta": 0.2},
       instance_type="ml.m5.xlarge"
   )
   
   evaluation_config = ConfigXGBoostModelEvalStep(
       model_name="my-model",
       test_data="s3://my-bucket/test-data/",
       instance_type="ml.m5.large"
   )
   
   # Compile pipeline
   pipeline = compile_dag_to_pipeline(
       dag,
       pipeline_name="end-to-end-ml-pipeline",
       step_configs={
           "preprocessing": preprocessing_config,
           "training": training_config,
           "evaluation": evaluation_config
       }
   )
   
   # Execute
   execution = pipeline.start()

Batch Transform Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

For inference pipelines:

.. code-block:: python

   from cursus.steps.configs import ConfigBatchTransformStep
   
   # Create inference DAG
   inference_dag = PipelineDAG()
   inference_dag.add_node("batch_transform")
   
   # Configure batch transform
   transform_config = ConfigBatchTransformStep(
       model_name="my-trained-model",
       input_data="s3://my-bucket/inference-data/",
       output_path="s3://my-bucket/predictions/",
       instance_type="ml.m5.large",
       instance_count=2
   )
   
   # Compile and execute
   inference_pipeline = compile_dag_to_pipeline(
       inference_dag,
       pipeline_name="batch-inference",
       step_configs={"batch_transform": transform_config}
   )

CLI Usage
---------

Cursus provides a powerful CLI for pipeline management:

.. code-block:: bash

   # Validate a pipeline configuration
   cursus validate --config pipeline_config.yaml
   
   # Test step builders
   cursus test-builder xgboost_training
   
   # Check alignment between contracts and specifications
   cursus alignment-check --step-type xgboost_training
   
   # List available step types
   cursus registry list-steps

Configuration Management
------------------------

Cursus uses a three-tier configuration system:

1. **Base Configuration**: Common settings across all steps
2. **Step-Specific Configuration**: Settings specific to step types
3. **Runtime Configuration**: Dynamic settings provided at execution time

Example configuration file:

.. code-block:: yaml

   # pipeline_config.yaml
   pipeline_name: "my-ml-pipeline"
   role_arn: "arn:aws:iam::123456789012:role/SageMakerRole"
   
   base_config:
     region: "us-west-2"
     enable_network_isolation: true
     
   steps:
     preprocessing:
       type: "TabularPreprocessing"
       config:
         instance_type: "ml.m5.large"
         input_data: "s3://my-bucket/raw-data/"
         
     training:
       type: "XGBoostTraining"
       config:
         instance_type: "ml.m5.xlarge"
         hyperparameters:
           max_depth: 6
           eta: 0.3

Load and use the configuration:

.. code-block:: python

   from cursus.core.config_fields import load_configs
   
   # Load configuration
   config = load_configs("pipeline_config.yaml")
   
   # Use with compiler
   pipeline = compile_dag_to_pipeline(dag, config=config)

Next Steps
----------

- Read the :doc:`../api/index` for detailed API documentation
- Explore :doc:`advanced_usage` for more complex scenarios
- Check out :doc:`examples` for real-world use cases
- Learn about the :doc:`../design/architecture` for deeper understanding

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**
   Make sure you have installed all required dependencies:
   
   .. code-block:: bash
   
      pip install cursus[all]

**AWS Credentials**
   Ensure your AWS credentials are properly configured:
   
   .. code-block:: bash
   
      aws configure
      # or set environment variables
      export AWS_ACCESS_KEY_ID=your_access_key
      export AWS_SECRET_ACCESS_KEY=your_secret_key

**SageMaker Permissions**
   Your IAM role needs the following permissions:
   
   - ``sagemaker:CreatePipeline``
   - ``sagemaker:StartPipelineExecution``
   - ``sagemaker:DescribePipelineExecution``
   - ``s3:GetObject`` and ``s3:PutObject`` for your data buckets

Getting Help
~~~~~~~~~~~~

- Check the :doc:`../api/index` for API documentation
- Review existing issues on `GitHub <https://github.com/TianpeiLuke/cursus/issues>`_
- Create a new issue if you encounter problems

That's it! You're now ready to start building ML pipelines with Cursus.
