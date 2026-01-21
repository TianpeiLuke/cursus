---
tags:
  - design
  - cli
  - pipeline_compilation
  - user_interface
  - automation
keywords:
  - CLI tools
  - pipeline compilation
  - DAG validation
  - project initialization
  - execution document generation
  - command line interface
  - user experience
  - automation
topics:
  - CLI design
  - pipeline compilation workflow
  - user interface design
  - execution document generation
language: python
date of note: 2025-09-21
---

# CLI Pipeline Compilation Tools Design

## Overview

This document describes the design and implementation of command-line interface (CLI) tools for the Cursus pipeline compilation system. The CLI provides user-friendly commands for initializing projects, validating DAGs, compiling pipelines, and generating execution documents, bridging the gap between the current development-focused CLI tools and the user-facing pipeline generation commands promised in the README.

## Current State Analysis

### Existing CLI Infrastructure

The current CLI system (`src/cursus/cli/`) provides a solid foundation with:

- **Entry Point**: Properly configured in `pyproject.toml` (`cursus = "cursus.cli:main"`)
- **Dispatcher Architecture**: Main dispatcher routes commands to specialized modules
- **Existing Commands**: 7 development/validation tools:
  - `alignment` - Alignment validation tools
  - `builder-test` - Step builder testing tools
  - `catalog` - Pipeline catalog management
  - `registry` - Registry management tools
  - `runtime-testing` - Runtime testing for pipeline scripts
  - `validation` - Naming and interface validation
  - `workspace` - Workspace management tools

### Missing User-Facing Commands

The README promises three key user-facing commands that are not yet implemented:

1. `cursus init --template xgboost --name fraud-detection` - Project generation
2. `cursus validate my_dag.py` - DAG validation
3. `cursus compile my_dag.py --name my-pipeline --output pipeline.json` - Pipeline compilation

### Available Core Functionality

The codebase provides robust underlying functionality:

- **`cursus.core.compiler.dag_compiler.compile_dag_to_pipeline()`** - Main compilation function
- **`cursus.core.compiler.dag_compiler.PipelineDAGCompiler`** - Advanced compiler with validation
- **`cursus.api.dag.base_dag.PipelineDAG`** - DAG construction API
- **`cursus.registry.builder_registry.StepBuilderRegistry`** - Step discovery and registration
- **`cursus.mods.exe_doc.generator.ExecutionDocumentGenerator`** - Execution document generation
- **Rich template system** in `cursus.pipeline_catalog.shared_dags`
- **Example patterns** in `slipbox/5_tutorials/main/sagemaker_pipeline_quick_start.md`

## Design Principles

### 1. User-Centric Design
- **Intuitive Commands**: Commands follow common CLI patterns (`init`, `validate`, `compile`)
- **Progressive Disclosure**: Simple commands with advanced options available
- **Clear Feedback**: Comprehensive error messages with actionable suggestions
- **Consistent Interface**: Uniform argument patterns across commands

### 2. Integration with Existing Architecture
- **Leverage Core Components**: Reuse existing compiler, registry, and validation systems
- **Maintain Separation**: CLI layer remains thin, delegating to core functionality
- **Preserve Development Tools**: Existing CLI commands remain unchanged
- **Unified Entry Point**: All commands accessible through single `cursus` command

### 3. Pipeline Catalog Integration
- **Catalog-Based Templates**: Use existing `cursus.pipeline_catalog.pipelines` as project templates
- **Framework-Specific**: Leverage existing XGBoost, PyTorch, and dummy pipeline implementations
- **Best Practices**: Templates are production-ready pipeline classes with proven patterns
- **Extensible**: Easy to add new templates by adding new pipeline classes to the catalog

### 4. Execution Document Integration
- **Automated Generation**: Leverage `ExecutionDocumentGenerator` for parameter extraction
- **MODS Compatibility**: Full integration with MODS execution patterns
- **Configuration Driven**: Use existing configuration system for parameter resolution

## Detailed Command Design

### 1. `cursus init` - Project Initialization

#### Purpose
Generate new pipeline projects from predefined templates, providing a complete starting point for pipeline development.

#### Command Signature
```bash
cursus init --template <template_name> --name <project_name> [options]
```

#### Arguments
- `--template, -t`: Template name (required)
  - `xgb_training_simple` - Basic XGBoost training pipeline
  - `xgb_training_evaluation` - XGBoost training with evaluation
  - `xgb_e2e_comprehensive` - Complete XGBoost end-to-end pipeline
  - `pytorch_training_basic` - Basic PyTorch training pipeline
  - `pytorch_e2e_standard` - Standard PyTorch end-to-end pipeline
  - `dummy_e2e_basic` - Simple processing pipeline template
- `--name, -n`: Project name (required)
- `--output-dir, -o`: Output directory (default: current directory)
- `--config-format`: Configuration format (`json` or `yaml`, default: `json`)
- `--include-sample-data`: Generate sample data files
- `--framework-version`: Specify framework version

#### Implementation Strategy

**Pipeline Catalog Integration:**
The CLI will use the existing pipeline catalog system (`cursus.pipeline_catalog.pipelines`) as the template source:

```python
# Available pipeline templates from catalog
from cursus.pipeline_catalog.pipelines import (
    XGBoostTrainingSimplePipeline,      # xgb_training_simple
    XGBoostTrainingEvaluationPipeline,  # xgb_training_evaluation
    XGBoostE2EComprehensivePipeline,    # xgb_e2e_comprehensive
    PyTorchTrainingBasicPipeline,       # pytorch_training_basic
    PyTorchE2EStandardPipeline,         # pytorch_e2e_standard
    DummyE2EBasicPipeline,              # dummy_e2e_basic
)
```

**Template Generation Process:**
1. **Pipeline Discovery**: Use `discover_pipelines()` to find available pipeline classes
2. **Pipeline Instantiation**: Create pipeline instance with user parameters
3. **DAG Extraction**: Extract DAG structure using `create_dag()` method
4. **Configuration Generation**: Generate configuration files using pipeline's config requirements
5. **Project Structure**: Create project directory with DAG file, config, and documentation
6. **Documentation**: Generate README with pipeline-specific usage instructions

**Template Sources:**
- **XGBoost Templates**: `xgb_training_simple`, `xgb_training_evaluation`, `xgb_e2e_comprehensive`
- **PyTorch Templates**: `pytorch_training_basic`, `pytorch_e2e_standard`
- **Simple Template**: `dummy_e2e_basic` for basic processing pipelines
- **Configuration Patterns**: Extracted from pipeline class metadata and requirements

#### Example Usage
```bash
# Generate basic XGBoost project
cursus init --template xgb_training_simple --name fraud-detection

# Generate comprehensive XGBoost project
cursus init --template xgb_e2e_comprehensive --name fraud-detection-full

# Generate PyTorch project with custom output directory
cursus init --template pytorch_training_basic --name image-classifier --output-dir ./projects

# Generate simple processing pipeline
cursus init --template dummy_e2e_basic --name data-pipeline --include-sample-data
```

#### Output Structure
```
fraud-detection/
├── __init__.py              # Python package initialization
├── dag.py                   # Pipeline DAG definition
├── config.json             # Pipeline configuration
├── requirements.txt         # Python dependencies
├── README.md               # Usage instructions
├── xgboost_training.py     # Training step implementation
├── xgboost_inference.py    # Inference step implementation
├── xgboost_model_evaluation.py  # Model evaluation implementation
├── hyperparams/            # Hyperparameters directory
│   ├── __init__.py
│   ├── hyperparameters_base.py
│   ├── hyperparameters_xgboost.py
│   └── hyperparameters.json
├── processing/             # Processing modules directory
│   ├── __init__.py
│   ├── processors.py
│   ├── bsm_processor.py
│   ├── categorical_label_processor.py
│   ├── numerical_binning_processor.py
│   └── constants.py
├── scripts/                # Processing scripts directory
│   ├── __init__.py
│   ├── tabular_preprocessing.py
│   ├── model_calibration.py
│   ├── package.py
│   └── payload.py
└── data/                   # Sample data (if requested)
    ├── training/
    └── calibration/
```

### 2. `cursus validate` - DAG Validation

#### Purpose
Validate DAG files before compilation, providing comprehensive checks for structure, configuration compatibility, and step resolution.

#### Command Signature
```bash
cursus validate <dag_file> [options]
```

#### Arguments
- `dag_file`: Path to Python file containing DAG definition (required)
- `--config, -c`: Configuration file path (optional)
- `--verbose, -v`: Verbose output with detailed validation results
- `--format`: Output format (`text`, `json`, `yaml`, default: `text`)
- `--strict`: Fail on warnings (default: false)

#### Implementation Strategy

**Validation Pipeline:**
1. **Syntax Validation**: Parse Python file and check for syntax errors
2. **DAG Structure Validation**: 
   - Check for cycles using `PipelineDAG.has_cycles()`
   - Validate node connectivity
   - Ensure DAG has entry and exit points
3. **Step Name Validation**: 
   - Use `StepBuilderRegistry` to validate step names
   - Check against known step types
   - Suggest corrections for typos
4. **Configuration Compatibility**:
   - Use `PipelineDAGCompiler.validate_dag_compatibility()`
   - Check configuration file structure
   - Validate step-to-config mappings
5. **Dependency Resolution**:
   - Use `PipelineDAGCompiler.preview_resolution()`
   - Check for unresolvable dependencies
   - Validate input/output compatibility

**Validation Components:**
- **DAG Loader**: Safe execution of DAG file in isolated environment
- **Validation Engine**: Leverage `cursus.core.compiler.validation.ValidationEngine`
- **Registry Integration**: Use `cursus.registry.builder_registry` for step validation
- **Configuration Resolver**: Use `cursus.core.compiler.config_resolver.StepConfigResolver`

#### Example Usage
```bash
# Basic validation
cursus validate my_dag.py

# Validate with configuration
cursus validate my_dag.py --config config.json

# Verbose output
cursus validate my_dag.py --verbose

# JSON output for CI/CD
cursus validate my_dag.py --format json
```

#### Output Examples
```bash
# Success case
✅ DAG validation passed!
📊 Confidence score: 0.95
🔧 Steps: 5 nodes, 4 edges
📋 All configurations resolved successfully

# Failure case
❌ DAG validation failed!
Issues found:
  - Missing configuration for step 'XGBoostTraining'
  - Unknown step type 'CustomProcessing'
  - Circular dependency detected: A → B → A

Suggestions:
  - Add XGBoostTrainingConfig to configuration file
  - Check spelling of 'CustomProcessing' (did you mean 'TabularPreprocessing'?)
  - Remove edge from B to A to break cycle
```

### 3. `cursus compile` - Pipeline Compilation

#### Purpose
Compile serialized DAG JSON files and configuration JSON files to SageMaker Pipeline objects, providing the core pipeline generation functionality with optional deployment and execution capabilities.

#### Command Signature
```bash
cursus compile --dag-file <dag.json> --config-file <config.json> [options]
```

#### Arguments

**Required:**
- `--dag-file, -d`: Path to serialized DAG JSON file (required)
- `--config-file, -c`: Path to configuration JSON file (required)

**Pipeline Configuration:**
- `--pipeline-name, -n`: Override pipeline name (optional)
- `--role`: IAM role ARN for pipeline execution (optional, can use default from config)

**SageMaker Operations:**
- `--upsert`: Create/update pipeline in SageMaker service (flag)
- `--start`: Start pipeline execution after upserting (flag, requires --upsert)

**Output Options:**
- `--output, -o`: Save pipeline definition to JSON file (optional)
- `--format`: Output format for console display (`text`, `json`, default: `text`)
- `--show-report`: Display detailed compilation report (flag)

**Validation:**
- `--validate-only`: Only validate compatibility, don't compile (flag)

#### Implementation Strategy

**Compilation Pipeline:**
1. **DAG Loading**: Load DAG from JSON using `import_dag_from_json()`
2. **Configuration Loading**: Load and validate configuration JSON file
3. **Validation** (if --validate-only): Run compatibility validation and exit
4. **Compilation**: Use `compile_dag_to_pipeline()` to generate SageMaker Pipeline object
5. **Pipeline Definition Export**: Save pipeline definition using `pipeline.definition()` if --output specified
6. **SageMaker Upsert**: Call `pipeline.upsert()` if --upsert flag provided
7. **Execution Start**: Call `pipeline.start()` if --start flag provided
8. **Report Generation**: Optional detailed compilation report if --show-report

**Core Components:**
- **DAG Serialization**: `cursus.api.dag.pipeline_dag_serializer.import_dag_from_json()`
- **DAG Compiler**: `cursus.core.compiler.dag_compiler.compile_dag_to_pipeline()`
- **Pipeline Object**: SageMaker `Pipeline` with `.definition()`, `.upsert()`, `.start()` methods
- **Validation**: `PipelineDAGCompiler.validate_dag_compatibility()`
- **Report Generation**: `PipelineDAGCompiler.compile_with_report()`

#### Example Usage

**Basic Compilation (Console Output Only):**
```bash
# Compile and display summary
cursus compile -d dag.json -c config.json
```

**Save Pipeline Definition:**
```bash
# Compile and save pipeline definition to file
cursus compile -d dag.json -c config.json -o pipeline_definition.json
```

**Deploy to SageMaker:**
```bash
# Compile and upsert to SageMaker (create/update pipeline)
cursus compile -d dag.json -c config.json --upsert
```

**Complete Workflow (Deploy + Execute):**
```bash
# Compile, upsert, and start execution in one command
cursus compile -d dag.json -c config.json --upsert --start
```

**Advanced Options:**
```bash
# Custom pipeline name with execution
cursus compile -d dag.json -c config.json -n "FraudDetection-v2" --upsert --start

# Save definition + upsert + start + detailed report
cursus compile -d dag.json -c config.json -o pipeline_def.json --upsert --start --show-report

# Validation only (no compilation)
cursus compile -d dag.json -c config.json --validate-only

# JSON output format for CI/CD integration
cursus compile -d dag.json -c config.json --format json
```

#### Output Examples

**Basic Compilation:**
```bash
$ cursus compile -d dag.json -c config.json

✓ DAG loaded: 4 nodes, 3 edges
✓ Config loaded: 4 step configurations
✓ Pipeline compiled successfully

Pipeline: FraudDetection-v1-20260120-225516
Steps: 4 SageMaker steps created
Validation: All configurations resolved successfully
```

**With Upsert:**
```bash
$ cursus compile -d dag.json -c config.json --upsert

✓ DAG loaded: 4 nodes, 3 edges
✓ Config loaded: 4 step configurations
✓ Pipeline compiled successfully

Upserting to SageMaker...
✓ Pipeline created/updated
  Pipeline Name: FraudDetection-v1-20260120-225516
  Pipeline ARN: arn:aws:sagemaker:us-east-1:123456789012:pipeline/frauddetection-v1-20260120-225516
```

**With Execution:**
```bash
$ cursus compile -d dag.json -c config.json --upsert --start

✓ DAG loaded: 4 nodes, 3 edges
✓ Config loaded: 4 step configurations
✓ Pipeline compiled successfully

Upserting to SageMaker...
✓ Pipeline created/updated
  Pipeline ARN: arn:aws:sagemaker:us-east-1:123456789012:pipeline/frauddetection-v1-20260120-225516

Starting execution...
✓ Execution started
  Execution ARN: arn:aws:sagemaker:us-east-1:123456789012:pipeline/frauddetection-v1-20260120-225516/execution/abc123
  Execution ID: abc123
  Status: Executing

Monitor execution at:
  https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/pipelines/frauddetection-v1-20260120-225516/executions/abc123
```

**With Report:**
```bash
$ cursus compile -d dag.json -c config.json --show-report

✓ DAG loaded: 4 nodes, 3 edges
✓ Config loaded: 4 step configurations
✓ Pipeline compiled successfully

📋 Compilation Report:
   Pipeline: FraudDetection-v1-20260120-225516
   Steps: 4
   Average confidence: 0.95
   Warnings: 0
   
   Resolution Details:
     data_load → CradleDataLoadConfig (ProcessingStepBuilder, confidence: 0.98)
     preprocess → TabularPreprocessingConfig (ProcessingStepBuilder, confidence: 0.95)
     train → XGBoostTrainingConfig (TrainingStepBuilder, confidence: 0.92)
     evaluate → XGBoostModelEvalConfig (ProcessingStepBuilder, confidence: 0.95)
```

**Validation Only:**
```bash
$ cursus compile -d dag.json -c config.json --validate-only

✓ DAG loaded: 4 nodes, 3 edges
✓ Config loaded: 4 step configurations

Validation Results:
✓ All DAG nodes have matching configurations
✓ All step builders resolved successfully
✓ No dependency issues found
✓ Average confidence: 0.95

Validation passed! Ready for compilation.
```

**JSON Output:**
```bash
$ cursus compile -d dag.json -c config.json --format json

{
  "status": "success",
  "pipeline_name": "FraudDetection-v1-20260120-225516",
  "dag_nodes": 4,
  "dag_edges": 3,
  "steps_created": 4,
  "validation": {
    "passed": true,
    "confidence": 0.95,
    "warnings": []
  }
}
```

#### DAG JSON Format

The DAG JSON file should be created using the `export_dag_to_json()` function or have the following structure:

```json
{
  "created_at": "2026-01-20T22:55:16Z",
  "metadata": {
    "description": "Fraud detection training pipeline",
    "author": "data-science-team",
    "version": "1.0.0"
  },
  "dag": {
    "nodes": ["data_load", "preprocess", "train", "evaluate"],
    "edges": [
      ["data_load", "preprocess"],
      ["preprocess", "train"],
      ["train", "evaluate"]
    ]
  },
  "statistics": {
    "node_count": 4,
    "edge_count": 3,
    "has_cycles": false,
    "entry_nodes": ["data_load"],
    "exit_nodes": ["evaluate"],
    "max_depth": 3
  }
}
```

#### Python API Alternative

For notebook/script usage, the Python API provides direct access to Pipeline objects:

```python
from cursus.core.compiler import compile_dag_to_pipeline

# Compile from JSON files
pipeline = compile_dag_to_pipeline(
    dag_path="dag.json",
    config_path="config.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Use the pipeline object directly
pipeline.upsert()
execution = pipeline.start()
```

### 4. `cursus catalog` - Step Catalog Management

#### Purpose
Discover, search, and inspect pipeline steps and their components (configs, builders, contracts, specs, scripts) using the unified StepCatalog system.

#### Overview
The catalog CLI provides comprehensive access to the StepCatalog system, enabling:
- Step discovery across packages and workspaces
- Component-specific search (configs, builders, contracts, specs)
- Field-level inspection for configuration classes
- Type-based filtering (Processing, Training, Transform, etc.)
- Cross-workspace component discovery

#### StepCatalog Integration
All catalog commands use the `StepCatalog` class from `cursus.step_catalog`, which provides:
- O(1) step lookups via dictionary indexing
- Lazy-loaded component discovery
- Multi-workspace support
- Intelligent framework detection
- Registry-based validation

---

#### Core Discovery Commands

##### `catalog list`
List available pipeline steps with optional filtering.

**Command Signature:**
```bash
cursus catalog list [options]
```

**Parameters:**
- `--workspace`: Filter by workspace ID
- `--job-type`: Filter by job type (training, validation, etc.)
- `--framework`: Filter by detected framework (xgboost, pytorch)
- `--format`: Output format (table, json)
- `--limit`: Maximum number of results

**StepCatalog API:**
- `list_available_steps(workspace_id, job_type)`
- `detect_framework(step_name)`

**Example Usage:**
```bash
# List all steps
cursus catalog list

# Filter by framework
cursus catalog list --framework xgboost --limit 10

# Filter by job type
cursus catalog list --job-type training

# JSON output for CI/CD
cursus catalog list --framework pytorch --format json
```

**Output Example:**
```
📂 Available Steps (12 found):
  1. XGBoostTraining (xgboost)
  2. XGBoostModelEval (xgboost)
  3. PyTorchTraining (pytorch)
  4. TabularPreprocessing
  5. CradleDataLoad
  ...

Filters applied: framework: xgboost
```

---

##### `catalog search`
Search steps by name with fuzzy matching.

**Command Signature:**
```bash
cursus catalog search <query> [options]
```

**Parameters:**
- `query`: Search query string (required)
- `--job-type`: Filter by job type
- `--format`: Output format (table, json)
- `--limit`: Maximum number of results (default: 10)

**StepCatalog API:**
- `search_steps(query, job_type)`

**Example Usage:**
```bash
# Search for training steps
cursus catalog search "training"

# Search with job type filter
cursus catalog search "xgboost" --job-type validation

# Limit results
cursus catalog search "preprocess" --limit 5
```

**Output Example:**
```
🔍 Search Results for 'training' (3 found):
  1. XGBoostTraining (score: 1.00) (3 components)
     Reason: name_match
  2. PyTorchTraining (score: 1.00) (4 components)
     Reason: name_match
  3. TabularPreprocessing_training (score: 0.80) (2 components)
     Reason: fuzzy_match
```

---

##### `catalog show`
Show detailed information about a specific step.

**Command Signature:**
```bash
cursus catalog show <step_name> [options]
```

**Parameters:**
- `step_name`: Name of the step (required)
- `--format`: Output format (text, json)
- `--show-components`: Show detailed component information

**StepCatalog API:**
- `get_step_info(step_name)`
- `detect_framework(step_name)`
- `get_job_type_variants(step_name)`

**Example Usage:**
```bash
# Show step details
cursus catalog show XGBoostTraining

# Show with components
cursus catalog show XGBoostTraining --show-components

# JSON output
cursus catalog show PyTorchTraining --format json
```

**Output Example:**
```
📋 Step: XGBoostTraining
Workspace: core
Framework: xgboost

📝 Registry Information:
  builder_step_name: XGBoostTraining
  sagemaker_step_type: Training
  framework: xgboost

🔧 Available Components:
  config: cursus/steps/configs/config_xgboost_training_step.py
  builder: cursus/steps/builders/builder_xgboost_training_step.py
  contract: cursus/steps/contracts/xgboost_training_contract.py
  spec: cursus/steps/specs/xgboost_training_spec.py

🔄 Job Type Variants:
  XGBoostTraining_calibration
  XGBoostTraining_validation
```

---

##### `catalog components`
Show components available for a specific step.

**Command Signature:**
```bash
cursus catalog components <step_name> [options]
```

**Parameters:**
- `step_name`: Name of the step (required)
- `--type`: Filter by component type (script, contract, spec, builder, config)
- `--format`: Output format (text, json)

**StepCatalog API:**
- `get_step_info(step_name)`

**Example Usage:**
```bash
# Show all components
cursus catalog components XGBoostTraining

# Filter by type
cursus catalog components XGBoostTraining --type contract

# JSON output
cursus catalog components PyTorchTraining --format json
```

**Output Example:**
```
🔧 Components for XGBoostTraining:

CONFIG:
  Path: src/cursus/steps/configs/config_xgboost_training_step.py
  Type: config
  Modified: 2026-01-15 10:30:45

BUILDER:
  Path: src/cursus/steps/builders/builder_xgboost_training_step.py
  Type: builder
  Modified: 2026-01-14 09:15:22

CONTRACT:
  Path: src/cursus/steps/contracts/xgboost_training_contract.py
  Type: contract
  Modified: 2026-01-13 14:20:10
```

---

#### Enhanced Discovery Commands (NEW)

##### `catalog list-configs`
List all configuration classes discovered across packages and workspaces.

**Command Signature:**
```bash
cursus catalog list-configs [options]
```

**Parameters:**
- `--project-id`: Filter by project/workspace
- `--framework`: Filter by framework
- `--format`: Output format (table, json)
- `--show-fields`: Show field count for each config

**StepCatalog API:**
- `discover_config_classes(project_id)`

**Example Usage:**
```bash
# List all config classes
cursus catalog list-configs

# Filter by framework
cursus catalog list-configs --framework xgboost

# Show with field counts
cursus catalog list-configs --show-fields
```

**Output Example:**
```
📋 Configuration Classes (15 found):
  1. BasePipelineConfig (12 fields)
  2. ProcessingStepConfigBase (18 fields)
  3. XGBoostTrainingConfig (25 fields)
  4. PyTorchTrainingConfig (28 fields)
  5. TabularPreprocessingConfig (20 fields)
  ...
```

---

##### `catalog list-builders`
List all builder classes with optional filtering.

**Command Signature:**
```bash
cursus catalog list-builders [options]
```

**Parameters:**
- `--step-type`: Filter by SageMaker step type (Processing, Training, etc.)
- `--framework`: Filter by framework
- `--format`: Output format (table, json)
- `--show-path`: Show file path for each builder

**StepCatalog API:**
- `get_all_builders()`
- `get_builders_by_step_type(step_type)`

**Example Usage:**
```bash
# List all builders
cursus catalog list-builders

# Filter by step type
cursus catalog list-builders --step-type Training

# Show with paths
cursus catalog list-builders --show-path
```

**Output Example:**
```
🔧 Builder Classes (20 found):
  1. XGBoostTrainingStepBuilder
     Type: Training | Framework: xgboost
  2. PyTorchTrainingStepBuilder
     Type: Training | Framework: pytorch
  3. TabularPreprocessingStepBuilder
     Type: Processing
  ...
```

---

##### `catalog list-contracts`
List all contract classes.

**Command Signature:**
```bash
cursus catalog list-contracts [options]
```

**Parameters:**
- `--with-scripts-only`: Only show contracts that have corresponding scripts
- `--format`: Output format (table, json)
- `--show-entry-points`: Show script entry points

**StepCatalog API:**
- `discover_contracts_with_scripts()`
- `get_contract_entry_points()`

**Example Usage:**
```bash
# List all contracts
cursus catalog list-contracts

# Only contracts with scripts
cursus catalog list-contracts --with-scripts-only

# Show entry points
cursus catalog list-contracts --show-entry-points
```

**Output Example:**
```
📜 Contract Classes (18 found):
  1. XGBoostTrainingContract
     Entry Point: xgboost_training.py
  2. PyTorchTrainingContract
     Entry Point: pytorch_training.py
  3. TabularPreprocessingContract
     Entry Point: tabular_preprocessing.py
  ...
```

---

##### `catalog list-specs`
List all specification classes.

**Command Signature:**
```bash
cursus catalog list-specs [options]
```

**Parameters:**
- `--job-type`: Filter by job type
- `--framework`: Filter by framework
- `--format`: Output format (table, json)
- `--show-dependencies`: Show dependency count

**StepCatalog API:**
- `list_steps_with_specs(job_type)`
- `load_all_specifications()`

**Example Usage:**
```bash
# List all specs
cursus catalog list-specs

# Filter by job type
cursus catalog list-specs --job-type training

# Show with dependencies
cursus catalog list-specs --show-dependencies
```

**Output Example:**
```
📐 Specification Classes (16 found):
  1. XGBoostTrainingSpec (5 dependencies)
  2. PyTorchTrainingSpec (7 dependencies)
  3. TabularPreprocessingSpec (3 dependencies)
  ...
```

---

##### `catalog list-scripts`
List all script files discovered.

**Command Signature:**
```bash
cursus catalog list-scripts [options]
```

**Parameters:**
- `--project-id`: Filter by project/workspace
- `--format`: Output format (table, json)
- `--show-path`: Show full file path

**StepCatalog API:**
- `list_available_scripts()`
- `discover_script_files(project_id)`

**Example Usage:**
```bash
# List all scripts
cursus catalog list-scripts

# Filter by project
cursus catalog list-scripts --project-id my_workspace

# Show with paths
cursus catalog list-scripts --show-path
```

**Output Example:**
```
📜 Script Files (22 found):
  1. xgboost_training.py
  2. pytorch_training.py
  3. tabular_preprocessing.py
  4. model_evaluation_xgb.py
  ...
```

---

#### Advanced Search Commands (NEW)

##### `catalog search-field`
Find steps with configs containing a specific field.

**Command Signature:**
```bash
cursus catalog search-field <field_name> [options]
```

**Parameters:**
- `field_name`: Name of the field to search for (required)
- `--field-type`: Filter by field type (str, int, bool, dict, list)
- `--format`: Output format (table, json)
- `--show-default`: Show default values

**StepCatalog API:**
- `discover_config_classes()`
- Uses Pydantic `model_fields` inspection

**Example Usage:**
```bash
# Find all steps with 'instance_type' field
cursus catalog search-field instance_type

# Find with type filter
cursus catalog search-field bucket --field-type str

# Show defaults
cursus catalog search-field instance_count --show-default
```

**Output Example:**
```
🔍 Steps with field 'instance_type':
  1. XGBoostTraining
     Config: XGBoostTrainingConfig
     Field Type: str
     Default: ml.m5.xlarge
  
  2. PyTorchTraining
     Config: PyTorchTrainingConfig
     Field Type: str
     Default: ml.p3.2xlarge
  
  3. TabularPreprocessing
     Config: TabularPreprocessingConfig
     Field Type: str
     Default: ml.m5.2xlarge
```

---

##### `catalog list-by-type`
Filter steps by SageMaker step type.

**Command Signature:**
```bash
cursus catalog list-by-type <type> [options]
```

**Parameters:**
- `type`: SageMaker step type (required)
  - Valid: Processing, Training, Transform, CreateModel, Model, Tuning, etc.
- `--framework`: Filter by framework
- `--format`: Output format (table, json)

**StepCatalog API:**
- `get_step_info(step_name)` with `registry_data['sagemaker_step_type']`

**Example Usage:**
```bash
# List all Processing steps
cursus catalog list-by-type Processing

# List Training steps for PyTorch
cursus catalog list-by-type Training --framework pytorch

# JSON output
cursus catalog list-by-type Transform --format json
```

**Output Example:**
```
📦 Steps with type 'Processing' (8 found):
  1. TabularPreprocessing
     Framework: N/A
  2. CradleDataLoad
     Framework: N/A
  3. ModelCalibration
     Framework: xgboost
  ...
```

---

##### `catalog fields`
Show all configuration fields for a step with inheritance.

**Command Signature:**
```bash
cursus catalog fields <step_name> [options]
```

**Parameters:**
- `step_name`: Name of the step (required)
- `--inherited`: Show inherited fields from parent classes
- `--format`: Output format (table, json)
- `--show-types`: Show field types
- `--show-defaults`: Show default values

**StepCatalog API:**
- `discover_config_classes()`
- `get_immediate_parent_config_class()`
- Uses Pydantic `model_fields` inspection

**Example Usage:**
```bash
# Show all fields
cursus catalog fields XGBoostTraining

# Show with inheritance
cursus catalog fields XGBoostTraining --inherited

# Show with types and defaults
cursus catalog fields PyTorchTraining --show-types --show-defaults
```

**Output Example:**
```
🔧 Configuration Fields for XGBoostTraining:
Config Class: XGBoostTrainingConfig
Parent Class: TrainingStepConfigBase

Direct Fields (10):
  - hyperparameters (dict, optional)
    Default: {}
  - max_depth (int, optional)
    Default: 6
  - num_round (int, optional)
    Default: 100
  ...

Inherited from TrainingStepConfigBase (8):
  - instance_type (str, required)
    Default: ml.m5.xlarge
  - instance_count (int, optional)
    Default: 1
  - volume_size (int, optional)
    Default: 30
  ...

Inherited from BasePipelineConfig (12):
  - author (str, required)
  - bucket (str, required)
  - role (str, optional)
  ...

Total: 30 fields (10 direct + 20 inherited)
```

---

##### `catalog component-info`
Get detailed information about a specific component type for a step.

**Command Signature:**
```bash
cursus catalog component-info <step_name> <component_type> [options]
```

**Parameters:**
- `step_name`: Name of the step (required)
- `component_type`: Type of component (required)
  - Valid: config, builder, contract, spec, script
- `--format`: Output format (text, json)
- `--load`: Load and inspect the actual class/instance

**StepCatalog API:**
- `load_config_class()`, `load_builder_class()`, `load_contract_class()`, `load_spec_class()`
- `get_script_info()`

**Example Usage:**
```bash
# Get config info
cursus catalog component-info XGBoostTraining config

# Get builder info with loading
cursus catalog component-info PyTorchTraining builder --load

# Get contract info
cursus catalog component-info TabularPreprocessing contract --load
```

**Output Example:**
```
📋 Component Info: XGBoostTraining (config)
Component Type: Configuration
File Path: src/cursus/steps/configs/config_xgboost_training_step.py
Class Name: XGBoostTrainingConfig
Modified: 2026-01-15 10:30:45

Class Details:
  Parent: TrainingStepConfigBase
  Fields: 30 total (10 direct, 20 inherited)
  Framework: xgboost
  
Key Fields:
  - hyperparameters (dict)
  - max_depth (int)
  - instance_type (str)
  - instance_count (int)
```

---

#### Workspace & Framework Commands

##### `catalog frameworks`
List detected frameworks across all steps.

**Command Signature:**
```bash
cursus catalog frameworks [options]
```

**Parameters:**
- `--format`: Output format (table, json)

**StepCatalog API:**
- `detect_framework(step_name)` for all steps

**Example Usage:**
```bash
# List frameworks
cursus catalog frameworks

# JSON output
cursus catalog frameworks --format json
```

**Output Example:**
```
🔧 Detected Frameworks (3 total):
xgboost: 8 steps
  - XGBoostTraining
  - XGBoostModelEval
  - XGBoostPackaging
  ... and 5 more

pytorch: 6 steps
  - PyTorchTraining
  - PyTorchModelEval
  - PyTorchPackaging
  ... and 3 more

generic: 4 steps
  - TabularPreprocessing
  - CradleDataLoad
  ... and 2 more
```

---

##### `catalog workspaces`
List available workspaces and their step counts.

**Command Signature:**
```bash
cursus catalog workspaces [options]
```

**Parameters:**
- `--format`: Output format (table, json)

**StepCatalog API:**
- `discover_cross_workspace_components()`

**Example Usage:**
```bash
# List workspaces
cursus catalog workspaces

# JSON output
cursus catalog workspaces --format json
```

**Output Example:**
```
🏢 Available Workspaces (3 total):

core:
  Steps: 45
  Components: 180
  Example steps:
    - XGBoostTraining
    - PyTorchTraining
    - TabularPreprocessing
    ... and 42 more

my_project:
  Steps: 8
  Components: 32
  Example steps:
    - CustomPreprocessing
    - CustomTraining
    - CustomEvaluation
    ... and 5 more
```

---

##### `catalog metrics`
Show step catalog performance metrics.

**Command Signature:**
```bash
cursus catalog metrics [options]
```

**Parameters:**
- `--format`: Output format (text, json)

**StepCatalog API:**
- `get_metrics_report()`

**Example Usage:**
```bash
# Show metrics
cursus catalog metrics

# JSON output
cursus catalog metrics --format json
```

**Output Example:**
```
📊 Step Catalog Metrics:
Total Queries: 1,247
Success Rate: 99.2%
Average Response Time: 2.34ms
Index Build Time: 0.145s
Total Steps Indexed: 53
Total Workspaces: 3
Last Index Build: 2026-01-20 22:30:15
```

---

##### `catalog discover`
Discover steps in a specific workspace directory.

**Command Signature:**
```bash
cursus catalog discover --workspace-dir <path> [options]
```

**Parameters:**
- `--workspace-dir`: Workspace directory to discover (required)
- `--format`: Output format (text, json)

**StepCatalog API:**
- `StepCatalog(workspace_dirs=[path])`
- `list_available_steps(workspace_id)`

**Example Usage:**
```bash
# Discover workspace
cursus catalog discover --workspace-dir /path/to/my_workspace

# JSON output
cursus catalog discover --workspace-dir /path/to/my_workspace --format json
```

**Output Example:**
```
🔍 Discovery Results for /path/to/my_workspace:
Workspace ID: my_workspace
Steps Found: 8

Discovered Steps:
  1. CustomPreprocessing (config, script, contract)
  2. CustomTraining (config, builder, contract, spec)
  3. CustomEvaluation (config, script)
  ...
```

---

### 5. `cursus exec-doc` - Execution Document Generation

#### Purpose
Generate execution documents from serialized DAG and configuration JSON files, providing parameter extraction and MODS integration. The execution document contains step-specific configurations enriched with runtime parameters, enabling MODS-compliant pipeline execution.

#### Implementation Status
✅ **IMPLEMENTED** - Available in `src/cursus/cli/exec_doc_cli.py`

#### Command Signature
```bash
cursus exec-doc generate -d <dag.json> -c <config.json> [options]
```

#### Arguments

**Required:**
- `--dag-file, -d`: Path to serialized DAG JSON file (required)
- `--config-file, -c`: Path to configuration JSON file (required)

**Output Options:**
- `--output, -o`: Output file path (default: `execution_doc.json`)
- `--format`: Output format (`json` or `yaml`, default: `json`)

**Template Options:**
- `--template`: Base execution document template file (optional)
  - If provided, loads and fills existing template structure
  - If not provided, auto-generates base template from DAG

**Runtime Configuration:**
- `--role`: IAM role ARN for AWS operations (optional)
- `--verbose, -v`: Verbose output with detailed processing logs (flag)

#### Implementation Strategy

**Execution Document Generation Pipeline:**

1. **Load DAG from JSON**
```python
from cursus.api.dag.pipeline_dag_serializer import import_dag_from_json
dag = import_dag_from_json(dag_file_path)
```

2. **Create Base Execution Document Template**
```python
# Option A: User provides template file
if template:
    with open(template) as f:
        execution_document = json.load(f)

# Option B: Auto-generate base template from DAG
else:
    execution_document = {
        "PIPELINE_STEP_CONFIGS": {
            node: {
                "STEP_CONFIG": {},
                "STEP_TYPE": []
            }
            for node in dag.nodes
        }
    }
```

3. **Initialize ExecutionDocumentGenerator**
```python
from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator

generator = ExecutionDocumentGenerator(
    config_path=config_file,
    role=role,  # optional
)
```

4. **Fill Execution Document**
```python
filled_doc = generator.fill_execution_document(dag, execution_document)
```

5. **Save Output**
```python
# JSON format (default)
with open(output_file, 'w') as f:
    json.dump(filled_doc, f, indent=2)

# YAML format (if specified)
if format == 'yaml':
    with open(output_file, 'w') as f:
        yaml.dump(filled_doc, f, default_flow_style=False)
```

**Core Components:**
- **ExecutionDocumentGenerator**: `cursus.mods.exe_doc.generator.ExecutionDocumentGenerator`
  - Main method: `fill_execution_document(dag, execution_document)`
  - Loads configurations from config file
  - Identifies relevant steps in DAG (Cradle, Registration, etc.)
  - Applies specialized helpers for different step types
- **Helper System**: 
  - `CradleDataLoadingHelper`: Handles Cradle data loading configurations
  - `RegistrationHelper`: Handles model registration configurations
  - Extensible for additional step types
- **DAG Serialization**: `cursus.api.dag.pipeline_dag_serializer`
  - Imports DAG structure from JSON format
  - Preserves node relationships and metadata

#### Example Usage

**Basic Usage:**
```bash
# Generate execution document with default output
cursus exec-doc generate -d dag.json -c config.json
```

**Custom Output Location:**
```bash
# Specify custom output file
cursus exec-doc generate -d dag.json -c config.json -o my_exec_doc.json
```

**With Template:**
```bash
# Use existing template as base
cursus exec-doc generate -d dag.json -c config.json --template base_template.json
```

**YAML Output:**
```bash
# Generate YAML format
cursus exec-doc generate -d dag.json -c config.json --format yaml -o exec_doc.yaml
```

**With IAM Role:**
```bash
# Specify IAM role for AWS operations
cursus exec-doc generate -d dag.json -c config.json --role arn:aws:iam::123456789012:role/MyRole
```

**Verbose Mode:**
```bash
# Show detailed processing logs
cursus exec-doc generate -d dag.json -c config.json --verbose
```

**Complete Workflow:**
```bash
# Full pipeline: compile DAG, then generate execution document
cursus compile -d dag.json -c config.json -o pipeline_def.json
cursus exec-doc generate -d dag.json -c config.json -o execution_doc.json
```

#### Output Structure

The generated execution document has the following structure:

```json
{
  "PIPELINE_STEP_CONFIGS": {
    "step_name_1": {
      "STEP_CONFIG": {
        // Step-specific configuration filled by helpers
        "param1": "value1",
        "param2": "value2"
      },
      "STEP_TYPE": ["PROCESSING_STEP", "CustomType"]
    },
    "step_name_2": {
      "STEP_CONFIG": {
        // Another step's configuration
      },
      "STEP_TYPE": ["TRAINING_STEP"]
    }
  }
}
```

#### Helper-Specific Configuration

**Cradle Data Loading Steps:**
- Automatically filled by `CradleDataLoadingHelper`
- Extracts Cradle-specific parameters from configuration
- Populates execution document with data loading settings

**Registration Steps:**
- Automatically filled by `RegistrationHelper`
- Handles model registration configurations
- Integrates with payload and package configurations if present

#### Output Examples

**Success Case:**
```bash
$ cursus exec-doc generate -d dag.json -c config.json

🔧 Execution Document Generation

📂 Loading DAG from: dag.json
✓ DAG loaded: 4 nodes, 3 edges

📋 Preparing execution document template
  Auto-generating base template from DAG
✓ Base template generated with 4 steps

⚙️  Initializing generator
  Config file: config.json
✓ Generator initialized with 4 configurations

🔄 Filling execution document
✓ Execution document generated successfully

📊 Processing Summary:
  Total steps: 4
  Steps with configuration: 2

💾 Saving execution document
  Output file: execution_doc.json
  Format: json
✓ Execution document saved to: execution_doc.json
  File size: 2,456 bytes

✅ Execution document generation complete!

Next steps:
  1. Review the generated execution document: execution_doc.json
  2. Use with MODS for pipeline execution
```

**With Template:**
```bash
$ cursus exec-doc generate -d dag.json -c config.json --template base_template.json

🔧 Execution Document Generation

📂 Loading DAG from: dag.json
✓ DAG loaded: 4 nodes, 3 edges

📋 Preparing execution document template
  Loading template from: base_template.json
✓ Template loaded

⚙️  Initializing generator
  Config file: config.json
✓ Generator initialized with 4 configurations

🔄 Filling execution document
✓ Execution document generated successfully

📊 Processing Summary:
  Total steps: 4
  Steps with configuration: 2

💾 Saving execution document
  Output file: execution_doc.json
  Format: json
✓ Execution document saved to: execution_doc.json
  File size: 2,789 bytes

✅ Execution document generation complete!
```

**Verbose Output:**
```bash
$ cursus exec-doc generate -d dag.json -c config.json --verbose

🔧 Execution Document Generation

📂 Loading DAG from: dag.json
✓ DAG loaded: 4 nodes, 3 edges
  Nodes: ['data_load', 'preprocess', 'train', 'evaluate']

📋 Preparing execution document template
  Auto-generating base template from DAG
✓ Base template generated with 4 steps

⚙️  Initializing generator
  Config file: config.json
✓ Generator initialized with 4 configurations
  Loaded configs: ['CradleDataLoadConfig', 'TabularPreprocessingConfig', 'XGBoostTrainingConfig', 'XGBoostModelEvalConfig']

🔄 Filling execution document
✓ Execution document generated successfully

📊 Processing Summary:
  Total steps: 4
  Steps with configuration: 2

  Configured steps:
    - data_load: 8 parameters
    - train: 15 parameters

💾 Saving execution document
  Output file: execution_doc.json
  Format: json
✓ Execution document saved to: execution_doc.json
  File size: 3,142 bytes

✅ Execution document generation complete!
```

#### Integration with MODS

The generated execution document is fully compatible with MODS execution patterns:

1. **Step Configuration Structure**: Matches MODS expected format
2. **Helper System**: Ensures proper configuration extraction for MODS steps
3. **Execution Document Template**: Compatible with existing MODS templates
4. **Parameter Resolution**: Automatic parameter extraction from configurations

#### Error Handling

**Missing Configuration:**
```bash
❌ Failed to initialize generator: Configuration file not found: config.json

Suggestion: Check the file path and ensure the configuration file exists
```

**Invalid DAG Structure:**
```bash
❌ Failed to load DAG: Invalid JSON format in dag.json

Suggestion: Validate JSON syntax using: jq . dag.json
```

**Helper Processing Warning:**
```bash
⚠ Warning: Step 'model_registration' configuration could not be fully processed
  Continuing with partial execution document...
```

#### CLI Integration

The command is fully integrated into the Cursus CLI dispatcher:
- **File**: `src/cursus/cli/exec_doc_cli.py`
- **Registration**: `src/cursus/cli/__init__.py`
- **Help**: `python -m cursus.cli exec-doc --help`

#### Python API Alternative

For notebook/script usage, direct API access is available:

```python
from cursus.api.dag.pipeline_dag_serializer import import_dag_from_json
from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
import json

# Load DAG
dag = import_dag_from_json("dag.json")

# Create base template
execution_document = {
    "PIPELINE_STEP_CONFIGS": {
        node: {"STEP_CONFIG": {}, "STEP_TYPE": []}
        for node in dag.nodes
    }
}

# Initialize generator and fill document
generator = ExecutionDocumentGenerator(config_path="config.json")
filled_doc = generator.fill_execution_document(dag, execution_document)

# Save output
with open("execution_doc.json", 'w') as f:
    json.dump(filled_doc, f, indent=2)
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

**1.1 CLI Dispatcher Enhancement**
- Update `src/cursus/cli/__init__.py` to include new commands
- Add routing logic for `init`, `validate`, `compile`, `exec-doc`
- Update help text and command descriptions

**1.2 Base Command Classes**
- Create `src/cursus/cli/base_command.py` with common functionality
- Implement error handling, logging, and output formatting
- Create argument parsing utilities

**1.3 DAG Loading Utilities**
- Create `src/cursus/cli/dag_loader.py` for safe DAG file execution
- Implement validation and error handling for DAG loading
- Support for different DAG file formats and patterns

### Phase 2: Pipeline Catalog Integration (Week 2-3)

**2.1 Pipeline Catalog Integration**
- Integrate with `cursus.pipeline_catalog.pipelines` discovery system
- Implement pipeline class loading and instantiation
- Create pipeline-to-project conversion utilities

**2.2 Template Generation from Pipelines**
- Extract DAG structures from pipeline classes using `create_dag()` method
- Generate configuration requirements from pipeline metadata
- Create project files based on pipeline specifications
- Implement parameter substitution for user-provided values

**2.3 Project Generation**
- Implement `src/cursus/cli/init_cli.py` with pipeline catalog integration
- Create project directory structure generation
- Implement DAG file generation from pipeline classes
- Add configuration file generation based on pipeline requirements
- Add sample data generation capabilities

### Phase 3: Validation Command (Week 3-4)

**3.1 Validation Infrastructure**
- Implement `src/cursus/cli/validate_cli.py`
- Integrate with existing `ValidationEngine` and `PipelineDAGCompiler`
- Create comprehensive validation pipeline

**3.2 Validation Reporting**
- Implement multiple output formats (text, JSON, YAML)
- Create detailed error messages with suggestions
- Add confidence scoring and resolution preview

**3.3 Integration Testing**
- Test validation with various DAG patterns
- Validate error handling and edge cases
- Ensure compatibility with existing validation systems

### Phase 4: Compilation Command (Week 4-5)

**4.1 Compilation Infrastructure**
- Implement `src/cursus/cli/compile_cli.py`
- Integrate with `PipelineDAGCompiler` and existing compilation system
- Create output serialization for different formats

**4.2 Report Generation**
- Implement compilation reporting using `compile_with_report()`
- Create detailed resolution and confidence reporting
- Add metadata and statistics collection

**4.3 Pipeline Output**
- Implement JSON/YAML serialization of SageMaker pipelines
- Create pipeline metadata extraction
- Add validation of generated pipeline structure

### Phase 5: Execution Document Command (Week 5-6)

**5.1 Execution Document CLI**
- Implement `src/cursus/cli/exec_doc_cli.py`
- Integrate with `ExecutionDocumentGenerator`
- Create template loading and parameter extraction

**5.2 MODS Integration**
- Ensure full compatibility with MODS execution patterns
- Test with existing execution document templates
- Validate helper system integration

**5.3 End-to-End Testing**
- Test complete workflow from init to execution document
- Validate integration with SageMaker pipeline execution
- Create comprehensive test suite

### Phase 6: Documentation and Polish (Week 6-7)

**6.1 Documentation**
- Update README.md with accurate CLI command descriptions
- Create comprehensive CLI documentation
- Add examples and tutorials

**6.2 Error Handling**
- Improve error messages and suggestions
- Add comprehensive logging and debugging support
- Create troubleshooting guides

**6.3 Testing and Validation**
- Create comprehensive test suite for all CLI commands
- Add integration tests with real pipeline compilation
- Validate performance and reliability

## Technical Architecture

### CLI Module Structure

```
src/cursus/cli/
├── __init__.py              # Main dispatcher (updated)
├── base_command.py          # Base command functionality
├── dag_loader.py           # DAG file loading utilities
├── init_cli.py             # Project initialization command
├── validate_cli.py         # DAG validation command
├── compile_cli.py          # Pipeline compilation command
├── exec_doc_cli.py         # Execution document command
├── catalog_integration.py  # Pipeline catalog integration utilities
└── utils/                  # CLI utilities
    ├── __init__.py
    ├── output.py          # Output formatting
    ├── validation.py      # Validation utilities
    └── errors.py          # Error handling
```

### Integration Points

**1. Core Compiler Integration**
- Direct use of `PipelineDAGCompiler` for compilation and validation
- Integration with `ValidationEngine` for comprehensive validation
- Use of `StepConfigResolver` for configuration resolution

**2. Pipeline Catalog Integration**
- Use of `cursus.pipeline_catalog.pipelines` for template discovery and loading
- Integration with pipeline catalog registry for template generation
- Validation against known pipeline types and patterns
- Leverage existing pipeline metadata and enhanced DAG metadata

**3. Configuration System Integration**
- Use of existing configuration loading and validation
- Integration with config field management system
- Support for portable path resolution

**4. Execution Document Integration**
- Direct use of `ExecutionDocumentGenerator`
- Integration with helper system for specialized step types
- Full MODS compatibility for execution patterns

### Error Handling Strategy

**1. Graceful Degradation**
- Commands continue with warnings when possible
- Clear distinction between errors and warnings
- Actionable error messages with suggestions

**2. Comprehensive Validation**
- Early validation of inputs before processing
- Clear error messages for common mistakes
- Suggestions for fixing validation errors

**3. Debug Support**
- Verbose modes for detailed debugging information
- Logging integration for troubleshooting
- Stack trace preservation for development

### Output Format Strategy

**1. Human-Readable Output**
- Clear, formatted text output for interactive use
- Progress indicators for long-running operations
- Color coding and emojis for visual clarity

**2. Machine-Readable Output**
- JSON and YAML formats for CI/CD integration
- Structured error reporting for automated processing
- Exit codes for script integration

**3. Flexible Formatting**
- Configurable output formats per command
- Support for different verbosity levels
- Integration with existing logging systems

## Usage Examples

### Complete Workflow Example

```bash
# 1. Initialize new XGBoost project using pipeline catalog
cursus init --template xgb_training_simple --name fraud-detection
cd fraud-detection

# 2. Customize configuration and DAG as needed
# Edit config.json and dag.py (generated from pipeline catalog)

# 3. Validate the DAG
cursus validate dag.py --config config.json --verbose

# 4. Compile to SageMaker pipeline
cursus compile dag.py --config config.json --name fraud-detection-v1 --output pipeline.json --report

# 5. Generate execution document
cursus exec-doc dag.py --config config.json --pipeline pipeline.json --output execution_doc.json

# 6. Deploy and execute (using existing MODS tools)
# Use SageMaker console or MODS helper to execute pipeline
```

### CI/CD Integration Example

```bash
# Validation in CI pipeline
cursus validate dag.py --config config.json --format json --strict > validation_report.json

# Compilation for deployment
cursus compile dag.py --config config.json --name ${PIPELINE_NAME}-${BUILD_NUMBER} --output pipeline.json

# Execution document generation
cursus exec-doc dag.py --config config.json --pipeline pipeline.json --output execution_doc.json
```

### Development Workflow Example

```bash
# Quick validation during development
cursus validate dag.py --config config.json

# Test compilation without output
cursus compile dag.py --config config.json --validate

# Generate execution document for testing
cursus exec-doc dag.py --config config.json --output test_exec_doc.json
```

## Benefits and Impact

### 1. User Experience Improvement
- **Simplified Onboarding**: New users can quickly generate working pipelines
- **Reduced Complexity**: Hide implementation details behind simple commands
- **Clear Feedback**: Comprehensive validation and error reporting
- **Consistent Interface**: Uniform command patterns across all operations

### 2. Development Velocity
- **Faster Prototyping**: Quick project initialization with proven templates
- **Early Validation**: Catch errors before compilation and deployment
- **Automated Generation**: Reduce manual configuration and setup work
- **Integration Ready**: Direct integration with CI/CD and deployment workflows

### 3. Quality Assurance
- **Validation First**: Comprehensive validation before compilation
- **Best Practices**: Templates embody proven patterns and configurations
- **Error Prevention**: Early detection of common mistakes and issues
- **Consistency**: Standardized project structure and configuration patterns

### 4. Ecosystem Integration
- **MODS Compatibility**: Full integration with existing MODS workflows
- **SageMaker Integration**: Direct compilation to SageMaker pipeline format
- **Tool Chain**: Seamless integration with existing development tools
- **Extensibility**: Easy to add new templates and validation rules

## Future Enhancements

### 1. Advanced Template System
- **Custom Templates**: User-defined templates for organization-specific patterns
- **Template Marketplace**: Shared templates for common use cases
- **Version Management**: Template versioning and compatibility tracking
- **Dynamic Templates**: Templates that adapt based on configuration

### 2. Interactive Features
- **Interactive Init**: Guided project initialization with prompts
- **Configuration Wizard**: Step-by-step configuration generation
- **Validation Assistant**: Interactive validation with fix suggestions
- **Pipeline Preview**: Visual representation of compiled pipelines

### 3. Integration Enhancements
- **IDE Integration**: Plugins for popular IDEs and editors
- **Git Integration**: Automatic git repository initialization and configuration
- **Cloud Integration**: Direct deployment to AWS from CLI
- **Monitoring Integration**: Pipeline monitoring and alerting setup

### 4. Advanced Validation
- **Semantic Validation**: Deep validation of pipeline logic and data flow
- **Performance Validation**: Cost and performance estimation
- **Security Validation**: Security best practices and compliance checking
- **Dependency Analysis**: Advanced dependency resolution and conflict detection

## References

### Core Implementation References
- **[DAG Compiler](../../src/cursus/core/compiler/dag_compiler.py)** - Main compilation engine and validation
- **[Pipeline DAG](../../src/cursus/api/dag/base_dag.py)** - DAG structure and manipulation
- **[Step Builder Registry](../../src/cursus/registry/builder_registry.py)** - Step discovery and validation
- **[Execution Document Generator](../../src/cursus/mods/exe_doc/generator.py)** - Execution document generation
- **[Configuration System](../../src/cursus/core/config_fields/)** - Configuration loading and management

### Pipeline Catalog and Template References
- **[Pipeline Catalog](../../src/cursus/pipeline_catalog/)** - Complete pipeline catalog system
- **[XGBoost Training Simple](../../src/cursus/pipeline_catalog/pipelines/xgb_training_simple.py)** - XGBoost template source
- **[Pipeline Discovery](../../src/cursus/pipeline_catalog/pipelines/__init__.py)** - Pipeline discovery and loading system
- **[Shared DAGs](../../src/cursus/pipeline_catalog/shared_dags/)** - Reusable DAG patterns
- **[Quick Start Tutorial](../../slipbox/5_tutorials/main/sagemaker_pipeline_quick_start.md)** - Usage patterns and examples
- **[Demo Configuration](../../demo/demo_config.ipynb)** - Configuration patterns and examples

### Design and Architecture References
- **[Specification-Driven Design](specification_driven_design.md)** - Core architectural principles
- **[Pipeline Compiler Design](pipeline_compiler.md)** - Compiler architecture and patterns
- **[Configuration Management](config_field_manager_refactoring.md)** - Configuration system design
- **[Dependency Resolution System](dependency_resolution_system.md)** - Dependency resolution architecture
- **[Execution Document Generator Design](standalone_execution_document_generator_design.md)** - Execution document architecture

### Development and Validation References
- **[Developer Guide](../../slipbox/0_developer_guide/README.md)** - Development patterns and best practices
- **[Validation Framework](../../slipbox/0_developer_guide/validation_framework_guide.md)** - Validation system architecture
- **[Step Builder Guide](../../slipbox/0_developer_guide/step_builder.md)** - Step builder patterns
- **[Best Practices](../../slipbox/0_developer_guide/best_practices.md)** - Development best practices

### CLI and User Interface References
- **[Workspace CLI Design](workspace_aware_cli_design.md)** - CLI design patterns and architecture
- **[Documentation Standard](documentation_yaml_frontmatter_standard.md)** - Documentation and metadata standards
- **[API Reference Documentation](api_reference_documentation_style_guide.md)** - Documentation style and patterns

## Conclusion

The CLI Pipeline Compilation Tools provide a comprehensive user interface for the Cursus pipeline system, bridging the gap between powerful core functionality and user-friendly operation. By implementing these tools, we enable users to quickly initialize projects, validate configurations, compile pipelines, and generate execution documents through simple, intuitive commands.

The design leverages existing core components while providing a clean, consistent interface that follows CLI best practices. The template system enables rapid prototyping and ensures best practices, while comprehensive validation prevents common errors and provides actionable feedback.

This CLI system transforms Cursus from a developer-focused library into a complete pipeline development platform, enabling both novice and expert users to efficiently create, validate, and deploy production-ready SageMaker pipelines.
