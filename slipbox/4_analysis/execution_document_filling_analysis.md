---
tags:
  - analysis
  - reference
  - execution_document
  - refactoring
  - cradle_data_loading
keywords:
  - execution document
  - refactoring plan
  - cradle data loading
  - pipeline generation
  - metadata collection
  - dependency separation
  - modular architecture
topics:
  - execution document filling
  - pipeline architecture
  - code refactoring
  - dependency management
language: python
date of note: 2025-09-16
---

# Execution Document Filling Analysis and Refactoring Plan

## Overview

This analysis examines the current execution document filling process throughout the cursus package, identifies the methods and data flow involved, and proposes a refactoring plan to separate execution document related methods from pipeline generation methods.

## Current Architecture Analysis

### 1. Data Flow from Bottom to Top

#### 1.1 Cradle Data Loading Step (Bottom Layer)
**File**: `src/cursus/steps/builders/builder_cradle_data_loading_step.py`

**Key Methods**:
- `_build_request()`: Converts config to `CreateCradleDataLoadJobRequest`
- `get_request_dict()`: Converts request to dictionary using `coral_utils`

**Data Generated**:
- Cradle request dictionary containing:
  - `data_sources`: DataSourcesSpecification with MDS/EDX/ANDES configs
  - `transform_specification`: SQL transforms and job split options
  - `output_specification`: Output paths, formats, and schemas
  - `cradle_job_specification`: Cluster type, account, retry settings

### 2. Pipeline Assembly Layer (Middle Layer)
**File**: `src/cursus/core/assembler/pipeline_assembler.py`

**Key Methods**:
- `_instantiate_step()`: Creates individual steps and stores Cradle requests
- `_store_pipeline_metadata()`: Stores step metadata

**Data Collection in `_instantiate_step()`**:
```python
# Special case for CradleDataLoading steps - store request dict for execution document
config = self.config_map[step_name]
step_type = BasePipelineConfig.get_step_name(type(config).__name__)
if "CradleDataLoad" in step_type and hasattr(builder, "get_request_dict"):
    self.cradle_loading_requests[step.name] = builder.get_request_dict()
    logger.info(f"Stored Cradle data loading request for step: {step.name}")
```

**Storage**: `cradle_loading_requests` class attribute dictionary on PipelineAssembler

**Class Attribute Declaration**:
```python
class PipelineAssembler:
    # Dictionary to store Cradle data loading requests
    cradle_loading_requests = {}
```

**Data Transfer in `_store_pipeline_metadata()` (called by template layer)**:
```python
def _store_pipeline_metadata(self, template: PipelineAssembler) -> None:
    if hasattr(template, "cradle_loading_requests"):
        self.pipeline_metadata["cradle_loading_requests"] = template.cradle_loading_requests
```

### 3. Template Layer (Middle-High Layer)
**File**: `src/cursus/core/assembler/pipeline_template_base.py`

**Key Methods**:
- `generate_pipeline()`: Coordinates pipeline generation
- `_store_pipeline_metadata()`: Stores assembler metadata
- `fill_execution_document()`: Base method (placeholder)

**Data Transfer**:
```python
def _store_pipeline_metadata(self, template: PipelineAssembler) -> None:
    if hasattr(template, "cradle_loading_requests"):
        self.pipeline_metadata["cradle_loading_requests"] = template.cradle_loading_requests
```

### 4. Dynamic Template Layer (High Layer)
**File**: `src/cursus/core/compiler/dynamic_template.py`

**Key Methods**:
- `_store_pipeline_metadata()`: Enhanced metadata collection from assembler
- `fill_execution_document()`: Main execution document filling logic
- `_fill_cradle_configurations()`: Fills Cradle-specific configs from stored requests
- `_fill_registration_configurations()`: Fills MIMS registration configs

**Note**: This is the primary implementation of execution document filling, located in the compiler folder and extending the base template functionality.

**Code Snippets for Execution Document Creation**:

**`fill_execution_document()` - Main entry point**:
```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in the execution document with pipeline metadata."""
    if "PIPELINE_STEP_CONFIGS" not in execution_document:
        self.logger.warning("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
        return execution_document

    pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

    # 1. Handle Cradle data loading requests
    self._fill_cradle_configurations(pipeline_configs)

    # 2. Handle Registration configurations
    self._fill_registration_configurations(pipeline_configs)

    return execution_document
```

**`_fill_cradle_configurations()` - Fills Cradle data loading configs**:
```python
def _fill_cradle_configurations(self, pipeline_configs: Dict[str, Any]) -> None:
    """Fill Cradle data loading configurations in the execution document."""
    cradle_requests = self.pipeline_metadata.get("cradle_loading_requests", {})

    if not cradle_requests:
        self.logger.debug("No Cradle loading requests found in metadata")
        return

    for step_name, request_dict in cradle_requests.items():
        if step_name not in pipeline_configs:
            self.logger.warning(f"Cradle step '{step_name}' not found in execution document")
            continue

        pipeline_configs[step_name]["STEP_CONFIG"] = request_dict
        self.logger.info(f"Updated execution config for Cradle step: {step_name}")
```

**`_fill_registration_configurations()` - Fills MIMS registration configs**:
```python
def _fill_registration_configurations(self, pipeline_configs: Dict[str, Any]) -> None:
    """Fill Registration configurations in the execution document."""
    # Use the resolved config map to find registration steps
    registration_nodes = self._find_registration_step_nodes()
    if not registration_nodes:
        self.logger.debug("No registration steps found in DAG")
        return

    # Get stored registration configs from metadata
    registration_configs = self.pipeline_metadata.get("registration_configs", {})

    # Generate search patterns for registration step names
    search_patterns = registration_nodes + ["model_registration", "Registration", "register_model"]

    # Process each potential registration step
    for pattern in search_patterns:
        if pattern in pipeline_configs:
            # Update the execution config
            if registration_configs:
                for step_name, config in registration_configs.items():
                    pipeline_configs[pattern]["STEP_CONFIG"] = config
                    self.logger.info(f"Updated execution config for registration step: {pattern}")
                    break
```

**`_store_pipeline_metadata()` - Enhanced metadata collection from assembler**:
```python
def _store_pipeline_metadata(self, assembler: "PipelineAssembler") -> None:
    """Store pipeline metadata from template."""
    # Store Cradle data loading requests if available
    if hasattr(assembler, "cradle_loading_requests"):
        self.pipeline_metadata["cradle_loading_requests"] = assembler.cradle_loading_requests
        self.logger.info(f"Stored {len(assembler.cradle_loading_requests)} Cradle loading requests")

    # Find and store registration steps and configurations
    try:
        registration_steps = []
        # Check step instances dictionary if available
        if hasattr(assembler, "step_instances"):
            for step_name, step_instance in assembler.step_instances.items():
                if "registration" in step_name.lower() or "registration" in str(type(step_instance)).lower():
                    registration_steps.append(step_instance)
                    self.logger.info(f"Found registration step: {step_name}")

        # If registration steps found, process and store configurations
        if registration_steps:
            # Store configs for all registration steps found
            registration_configs = {}
            for step in registration_steps:
                if hasattr(step, "name"):
                    exec_config = self._create_execution_doc_config("image-uri-placeholder")
                    registration_configs[step.name] = exec_config
                    self.logger.info(f"Stored execution doc config for registration step: {step.name}")
            
            self.pipeline_metadata["registration_configs"] = registration_configs

    except Exception as e:
        self.logger.warning(f"Error while processing registration steps: {e}")
```

**Execution Document Structure**:
```python
execution_document = {
    "PIPELINE_STEP_CONFIGS": {
        "step_name": {
            "STEP_CONFIG": {...},  # Filled by this process
            "STEP_TYPE": [...]
        }
    }
}
```

### 5. Compiler Layer (Top Layer)
**File**: `src/cursus/core/compiler/dag_compiler.py`

**Key Methods**:
- `compile_and_fill_execution_doc()`: Orchestrates compilation and filling
- `get_last_template()`: Provides access to template with metadata

**Usage Pattern**:
```python
pipeline = self.compile(dag, pipeline_name=pipeline_name, **kwargs)
if self._last_template is not None:
    filled_doc = self._last_template.fill_execution_document(execution_doc)
```

## Current Issues Identified

### 1. Tight Coupling
- Execution document logic is mixed with pipeline generation
- `get_request_dict()` and `_build_request()` are embedded in step builders
- Metadata collection is scattered across multiple layers

### 2. Dependency on Unsupported Packages
**From `com.amazon.secureaisandboxproxyservice.models`** (not supported):
- `Field`, `DataSource`, `MdsDataSourceProperties`, `EdxDataSourceProperties`
- `AndesDataSourceProperties`, `DataSourcesSpecification`, `JobSplitOptions`
- `TransformSpecification`, `OutputSpecification`, `CradleJobSpecification`
- `CreateCradleDataLoadJobRequest`

**From `secure_ai_sandbox_python_lib.utils`** (not supported):
- `coral_utils` (used for converting request objects to dictionaries)

**Supported packages**: 
- `secure_ai_sandbox_workflow_python_sdk` is supported
- `mods_workflow_core` is supported

### 3. Complex Data Flow
- Data passes through 5+ layers before reaching execution document
- Multiple transformation points create fragility
- Difficult to test execution document filling in isolation

## Refactoring Plan

### Phase 1: Create Execution Document Module

#### 1.1 Create New Module Structure
```
src/cursus/mods/exe_doc/
├── __init__.py
├── base.py                    # Base classes and interfaces
├── cradle_helper.py          # Cradle data loading helpers
├── registration_helper.py    # MIMS registration helpers
├── document_filler.py        # Main document filling orchestrator
└── utils.py                  # Utility functions
```

#### 1.2 Base Interface Design
```python
# src/cursus/mods/exe_doc/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ExecutionDocumentHelper(ABC):
    """Base class for execution document helpers."""
    
    @abstractmethod
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        """Check if this helper can handle the given step."""
        pass
    
    @abstractmethod
    def extract_step_config(self, step_name: str, config: Any, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract step configuration for execution document."""
        pass
```

#### 1.3 Cradle Helper Implementation
```python
# src/cursus/mods/exe_doc/cradle_helper.py
from .base import ExecutionDocumentHelper
from ..steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig

class CradleDataLoadingHelper(ExecutionDocumentHelper):
    """Helper for Cradle data loading execution document configuration."""
    
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        return isinstance(config, CradleDataLoadConfig)
    
    def extract_step_config(self, step_name: str, config: CradleDataLoadConfig, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract Cradle configuration without using unsupported packages."""
        # Implementation that doesn't rely on coral_utils or com.amazon packages
        return self._build_config_dict_directly(config)
    
    def _build_config_dict_directly(self, config: CradleDataLoadConfig) -> Dict[str, Any]:
        """Build configuration dictionary directly from config object."""
        # Direct dictionary construction without unsupported dependencies
        pass
```

### Phase 2: Separate Methods from Builders

#### 2.1 Extract from CradleDataLoadingStepBuilder
**Remove**:
- `_build_request()` method
- `get_request_dict()` method

**Keep**:
- `create_step()` method (core pipeline functionality)
- Configuration validation
- Step creation logic

#### 2.2 Extract from Pipeline Assembler
**Remove**:
- Cradle request collection logic from `_instantiate_step()`
- `cradle_loading_requests` class attribute

**Keep**:
- Core pipeline assembly logic
- Step instantiation
- Dependency resolution

### Phase 3: Implement New Execution Document System

#### 3.1 Document Filler Implementation
```python
# src/cursus/mods/exe_doc/document_filler.py
class ExecutionDocumentFiller:
    """Main orchestrator for filling execution documents."""
    
    def __init__(self):
        self.helpers = [
            CradleDataLoadingHelper(),
            # Add more helpers as needed
        ]
    
    def fill_document(self, execution_doc: Dict[str, Any], 
                     configs: Dict[str, Any]) -> Dict[str, Any]:
        """Fill execution document using appropriate helpers."""
        pipeline_configs = execution_doc.get("PIPELINE_STEP_CONFIGS", {})
        
        for step_name, config in configs.items():
            helper = self._find_helper(step_name, config)
            if helper:
                step_config = helper.extract_step_config(step_name, config)
                if step_name in pipeline_configs:
                    pipeline_configs[step_name]["STEP_CONFIG"] = step_config
        
        return execution_doc
```

#### 3.2 Integration Points
**Update Dynamic Template**:
```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """Fill execution document using the new modular system."""
    from ...mods.exe_doc.document_filler import ExecutionDocumentFiller
    
    filler = ExecutionDocumentFiller()
    return filler.fill_document(execution_document, self.configs)
```

### Phase 4: Remove Unsupported Dependencies

#### 4.1 Replace coral_utils Usage
- Implement direct dictionary serialization
- Remove dependency on `secure_ai_sandbox_python_lib`

#### 4.2 Replace com.amazon.secureaisandboxproxyservice Models
- Create internal data structures
- Implement direct dictionary construction
- Maintain compatibility with expected output format

## Benefits of Refactoring

### 1. Separation of Concerns
- Pipeline generation logic separated from execution document logic
- Each module has single responsibility
- Easier to test and maintain

### 2. Dependency Reduction
- Remove unsupported package dependencies
- Reduce coupling between components
- Improve reliability

### 3. Extensibility
- Easy to add new step types
- Pluggable helper system
- Consistent interface across helpers

### 4. Testability
- Execution document filling can be tested in isolation
- Mock configurations easily
- Unit test individual helpers

## Implementation Priority

1. **High Priority**: Create base module structure and interfaces
2. **High Priority**: Implement CradleDataLoadingHelper without unsupported deps
3. **Medium Priority**: Extract methods from existing builders
4. **Low Priority**: Add additional step type helpers as needed

## Migration Strategy

1. **Phase 1**: Create new module alongside existing code
2. **Phase 2**: Implement helpers and test thoroughly
3. **Phase 3**: Update templates to use new system
4. **Phase 4**: Remove old methods and dependencies
5. **Phase 5**: Clean up and optimize

This refactoring will create a cleaner, more maintainable system for execution document filling while removing dependencies on unsupported packages.
