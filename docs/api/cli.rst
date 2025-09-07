CLI Module
==========

.. currentmodule:: cursus.cli

The CLI module provides comprehensive command-line interface tools for pipeline development, validation, and management capabilities through a unified dispatcher architecture.

Overview
--------

The Cursus CLI system offers a complete interface for pipeline development, validation, and management tasks. It features a unified dispatcher architecture, consistent argument parsing, and robust error handling across all CLI modules.

The CLI system supports multiple command categories:

- Alignment validation across four levels
- Step builder testing and validation
- Pipeline catalog management
- Registry management and workspace operations
- Runtime testing and benchmarking
- Naming and interface validation
- Workspace lifecycle management

Quick Start
-----------

Installation and Basic Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Access main CLI help
   cursus --help

   # Or use module execution
   python -m cursus.cli --help

Common Command Patterns
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Alignment validation
   cursus alignment validate my_script --verbose
   cursus alignment validate-all --output-dir ./reports

   # Builder testing
   cursus builder-test validate MyStepBuilder
   cursus builder-test validate-all --scoring

   # Catalog management
   cursus catalog list --format json
   cursus catalog search --framework xgboost

   # Registry operations
   cursus registry list-steps --workspace my_project
   cursus registry validate-registry --check-conflicts

   # Runtime testing
   cursus runtime-testing test_script my_script.py
   cursus runtime-testing benchmark my_script.py --iterations 10

   # Validation tools
   cursus validation registry --verbose
   cursus validation interface src.cursus.steps.builders.MyBuilder

   # Workspace management
   cursus workspace create alice --template ml_pipeline
   cursus workspace list --show-components

CLI Architecture
----------------

Dispatcher Pattern
~~~~~~~~~~~~~~~~~~

The CLI system uses a dispatcher pattern for command routing:

1. **Main Dispatcher** - Routes commands to appropriate CLI modules
2. **Argument Parsing** - Comprehensive argument validation with help
3. **Command Forwarding** - Forwards arguments to selected CLI modules
4. **Exit Code Handling** - Preserves exit codes from sub-commands
5. **Error Management** - Consistent error handling across all commands

Integration Points
~~~~~~~~~~~~~~~~~~

- **Validation Framework** - Comprehensive validation capabilities
- **Registry System** - Step management and workspace operations
- **Pipeline Catalog** - Template management and discovery
- **Workspace System** - Developer environment setup and management
- **Builder Framework** - Step builder testing and validation

Command Categories
------------------

Alignment CLI
~~~~~~~~~~~~~

Comprehensive alignment validation across four levels of the framework architecture.

.. code-block:: bash

   # Validate single script alignment
   cursus alignment validate my_script --verbose
   
   # Validate all scripts with reporting
   cursus alignment validate-all --output-dir ./reports --format json

**Key Features:**

- Level 1: Script contract alignment validation
- Level 2: Property path validation implementation
- Level 3: Specification dependency alignment
- Level 4: Builder configuration alignment
- Comprehensive reporting with JSON and text formats
- Batch validation with error continuation

Builder Test CLI
~~~~~~~~~~~~~~~~

Step builder testing and validation tools for ensuring builder compliance.

.. code-block:: bash

   # Validate specific step builder
   cursus builder-test validate MyStepBuilder
   
   # Validate all builders with scoring
   cursus builder-test validate-all --scoring

**Key Features:**

- Individual builder validation
- Batch validation with scoring metrics
- Configuration compliance testing
- Integration testing capabilities

Catalog CLI
~~~~~~~~~~~

Pipeline catalog management with dual-compiler support for template discovery and management.

.. code-block:: bash

   # List available pipeline templates
   cursus catalog list --format json
   
   # Search templates by framework
   cursus catalog search --framework xgboost

**Key Features:**

- Template listing and discovery
- Framework-based filtering
- JSON output for programmatic access
- Template metadata management

Registry CLI
~~~~~~~~~~~~

Registry management and workspace tools for component organization.

.. code-block:: bash

   # List steps in workspace
   cursus registry list-steps --workspace my_project
   
   # Validate registry integrity
   cursus registry validate-registry --check-conflicts

**Key Features:**

- Step registry management
- Workspace-aware operations
- Conflict detection and resolution
- Registry integrity validation

Runtime Testing CLI
~~~~~~~~~~~~~~~~~~~~

Script runtime testing and benchmarking capabilities.

.. code-block:: bash

   # Test script execution
   cursus runtime-testing test_script my_script.py
   
   # Benchmark with iterations
   cursus runtime-testing benchmark my_script.py --iterations 10

**Key Features:**

- Script execution testing
- Performance benchmarking
- Iteration-based testing
- Runtime metrics collection

Validation CLI
~~~~~~~~~~~~~~

Naming and interface validation tools for framework compliance.

.. code-block:: bash

   # Validate registry naming standards
   cursus validation registry --verbose
   
   # Validate interface compliance
   cursus validation interface src.cursus.steps.builders.MyBuilder

**Key Features:**

- Registry naming validation
- Interface compliance checking
- Verbose reporting options
- Framework standard enforcement

Workspace CLI
~~~~~~~~~~~~~

Comprehensive workspace lifecycle management for development environments.

.. code-block:: bash

   # Create new workspace
   cursus workspace create alice --template ml_pipeline
   
   # List workspaces with components
   cursus workspace list --show-components
   
   # Validate workspace health
   cursus workspace validate --strict

**Key Features:**

- Workspace creation with templates
- Component discovery and listing
- Health checking and validation
- Template-based initialization

Error Handling
--------------

Standardized Exit Codes
~~~~~~~~~~~~~~~~~~~~~~~~

- **0** - Success
- **1** - General error or validation failure
- **2** - Invalid arguments or usage
- **3** - File or resource not found
- **4** - Permission or access error

Error Categories
~~~~~~~~~~~~~~~~

- **Command Routing** - Invalid command names with suggestions
- **Argument Parsing** - Comprehensive argument validation
- **Exception Handling** - Graceful handling with debugging information
- **Help Integration** - Automatic help display for invalid usage

Development Workflow Integration
--------------------------------

Pre-commit Validation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Validate alignment before commit
   cursus alignment validate-all --continue-on-error

   # Validate naming standards
   cursus validation registry

   # Test builders
   cursus builder-test validate-all

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # CI pipeline validation
   cursus alignment validate-all --format json --output-dir ./ci-reports
   cursus validation registry || exit 1
   cursus builder-test validate-all --scoring || exit 1

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Setup workspace
   cursus workspace create developer_name --template ml_pipeline

   # Validate workspace
   cursus workspace validate --strict

   # List available tools
   cursus workspace discover components

Advanced Usage
--------------

Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate multiple pipeline reports
   cursus alignment validate-all --output-dir ./reports --format both

   # Test all builders with custom configurations
   cursus builder-test validate-all --config-dir ./custom_configs

   # Comprehensive workspace health check
   cursus workspace health-check --fix-issues

Automation and Scripting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # JSON output for programmatic processing
   cursus catalog list --format json | jq '.[] | .id'
   cursus workspace list --format json | jq '.[] | select(.status == "healthy")'

   # Automated validation workflows
   cursus alignment validate-all --continue-on-error --output-dir ./validation
   cursus validation registry --verbose > validation_report.txt

Performance Considerations
--------------------------

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

- **Lazy Loading** - CLI modules loaded only when needed
- **Caching** - Validation results cached for repeated operations
- **Parallel Processing** - Multiple validations processed concurrently
- **Incremental Operations** - Only changed components validated when possible

Best Practices
~~~~~~~~~~~~~~

1. **Use specific filters** to reduce processing time
2. **Cache results** for batch operations
3. **Use JSON output** for programmatic processing
4. **Validate incrementally** during development
5. **Generate reports** for comprehensive analysis

API Reference
-------------

Main CLI Dispatcher
~~~~~~~~~~~~~~~~~~~

.. automodule:: cursus.cli
   :members:
   :undoc-members:
   :show-inheritance:

CLI Entry Point
~~~~~~~~~~~~~~~

.. automodule:: cursus.cli.__main__
   :members:
   :undoc-members:
   :show-inheritance:

Command Modules
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   generated/cursus.cli.alignment_cli
   generated/cursus.cli.builder_test_cli
   generated/cursus.cli.catalog_cli
   generated/cursus.cli.registry_cli
   generated/cursus.cli.runtime_testing_cli
   generated/cursus.cli.validation_cli
   generated/cursus.cli.workspace_cli
