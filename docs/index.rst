Cursus Documentation
===================

.. image:: https://img.shields.io/pypi/v/cursus.svg
   :target: https://pypi.org/project/cursus/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/cursus.svg
   :target: https://pypi.org/project/cursus/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/TianpeiLuke/cursus.svg
   :target: https://github.com/TianpeiLuke/cursus/blob/main/LICENSE
   :alt: License

**Cursus** is an intelligent pipeline generation system that automatically creates complete SageMaker pipelines from user-provided pipeline graphs with intelligent dependency resolution and configuration management.

🎯 **Graph-to-Pipeline Automation**: Automatically generate complete SageMaker pipelines

⚡ **10x Faster Development**: Minutes to working pipeline vs. weeks of manual configuration  

🧠 **Intelligent Dependency Resolution**: Automatic step connections and data flow

🛡️ **Production Ready**: Built-in quality gates and validation

📈 **Proven Results**: 60% average code reduction across pipeline components

Quick Start
-----------

Install Cursus:

.. code-block:: bash

   pip install cursus[all]

Basic usage:

.. code-block:: python

   import cursus
   from cursus import PipelineDAG
   
   # Create a simple pipeline DAG
   dag = PipelineDAG()
   dag.add_node("training")
   dag.add_node("evaluation")
   dag.add_edge("training", "evaluation")
   
   # Compile to SageMaker pipeline
   pipeline = cursus.compile_dag(dag, pipeline_name="fraud-detection")
   
   # Execute the pipeline
   pipeline.start()

Key Features
------------

**Automatic Pipeline Generation**
   Transform pipeline graphs into production-ready SageMaker pipelines automatically with intelligent dependency resolution.

**Configuration Management**
   Sophisticated configuration system with three-tier design (base, step-specific, runtime) for maximum flexibility.

**Step Builder Registry**
   Extensible registry system for pipeline step builders with automatic discovery and validation.

**Validation Framework**
   Comprehensive validation system ensuring pipeline correctness and alignment with SageMaker requirements.

**CLI Interface**
   Powerful command-line interface for pipeline management, testing, and deployment operations.

**Workspace Management**
   Advanced workspace system for managing multiple pipeline configurations and environments.

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/quickstart
   guides/basic_usage
   guides/advanced_usage
   guides/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/api
   api/cli
   api/steps
   api/processing
   api/registry
   api/validation
   api/pipeline_catalog
   api/workspace

.. toctree::
   :maxdepth: 2
   :caption: Design Documentation

   design/architecture
   design/configuration_system
   design/validation_framework
   design/step_builders

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/testing
   development/release_process

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   license
   glossary

Architecture Overview
--------------------

Cursus follows a modular architecture with clear separation of concerns:

**Core Framework**
   - **API Layer**: Public interfaces for pipeline creation and management
   - **Compiler**: DAG compilation and template generation engine
   - **Configuration**: Multi-tier configuration management system
   - **Base Classes**: Foundational abstractions for extensibility

**Pipeline Components**
   - **Step Builders**: Extensible builders for different pipeline step types
   - **Step Configurations**: Type-safe configuration classes for each step
   - **Step Contracts**: Validation contracts ensuring step compatibility

**Validation & Quality**
   - **Alignment Validation**: Ensures consistency between contracts and specifications
   - **Builder Testing**: Comprehensive testing framework for step builders
   - **Runtime Validation**: Pipeline execution validation and monitoring

**Supporting Systems**
   - **Registry**: Component discovery and management
   - **Workspace**: Multi-environment configuration management
   - **CLI**: Command-line interface for all operations

Integration with Existing Systems
---------------------------------

Cursus integrates seamlessly with:

- **Amazon SageMaker**: Native SageMaker pipeline generation and execution
- **AWS Services**: S3, IAM, CloudWatch, and other AWS services
- **ML Frameworks**: PyTorch, XGBoost, Scikit-learn, and custom frameworks
- **Data Processing**: Pandas, NumPy, and custom data processors

Community and Support
---------------------

- **GitHub Repository**: `TianpeiLuke/cursus <https://github.com/TianpeiLuke/cursus>`_
- **Issue Tracker**: `GitHub Issues <https://github.com/TianpeiLuke/cursus/issues>`_
- **Documentation**: This documentation site
- **License**: MIT License

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
