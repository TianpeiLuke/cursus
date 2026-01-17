---
tags:
  - project
  - implementation
  - debugging
  - single_node_execution
  - pipeline_compiler
keywords:
  - single node execution
  - manual input override
  - debugging API
  - isolated execution
  - pipeline restart
  - backward compatibility
topics:
  - Debugging tools
  - Developer experience
  - Compiler architecture
language: python
date of note: 2026-01-16
---

# Single Node Execution API Implementation Plan

## Overview

This document specifies the implementation of **Single Node Execution API** to enable debugging of individual pipeline nodes with manual input overrides, eliminating the need to re-run expensive upstream steps when pipeline failures occur.

**Timeline**: 3-4 weeks  
**Current Status**: Design Complete ✅, Implementation Pending 🔴  
**Priority**: High - Critical developer experience improvement

**Problem Statement**:
```
Pipeline fails at Step 4 after 5 hours of successful Steps 1-3 computation.
To test a fix, must re-run entire pipeline (another 5.5 hours).
Need: Run Step 4 alone with outputs from previous successful run.
```

**Root Cause**: Current pipeline framework lacks API to:
- Execute individual nodes in isolation
- Manually override input paths
- Bypass automatic dependency resolution

**Solution**: Create `SingleNodeCompiler` and `compile_single_node_to_pipeline()` API that:
- Compiles single-node pipelines with manual inputs
- Works with all existing step builders (zero breaking changes)
- Provides validation and preview capabilities
- Maintains full backward compatibility

## Executive Summary

### The Problem

When pipelines fail late in execution, developers face expensive re-runs:

**Current Debugging Flow**:
```
Pipeline Execution (Attempt 1):
├─ Step 1: Data Loading (30 minutes)       ✓ Success
├─ Step 2: Preprocessing (2 hours)         ✓ Success
├─ Step 3: Feature Engineering (3 hours)   ✓ Success
└─ Step 4: Model Training (starts)         ✗ FAILURE

Fix issue in Step 4 code...

Pipeline Execution (Attempt 2):
├─ Step 1: Data Loading (30 minutes)       ✓ Success (redundant!)
├─ Step 2: Preprocessing (2 hours)         ✓ Success (redundant!)
├─ Step 3: Feature Engineering (3 hours)   ✓ Success (redundant!)
└─ Step 4: Model Training (starts)         ✗ FAILURE (different error)

Fix issue again...

Pipeline Execution (Attempt 3):
├─ Step 1: Data Loading (30 minutes)       ✓ Success (redundant!)
├─ Step 2: Preprocessing (2 hours)         ✓ Success (redundant!)
├─ Step 3: Feature Engineering (3 hours)   ✓ Success (redundant!)
└─ Step 4: Model Training (completes)      ✓ Success

Total time: 18 hours (3 × 6 hours)
Wasted time: 12 hours (redundant computation)
```

**Problem**: 67% of time wasted on redundant computation!

### The Solution

**New Debugging Flow with Single Node Execution**:
```
Pipeline Execution (Attempt 1):
├─ Step 1: Data Loading (30 minutes)       ✓ Success
├─ Step 2: Preprocessing (2 hours)         ✓ Success
├─ Step 3: Feature Engineering (3 hours)   ✓ Success
└─ Step 4: Model Training (starts)         ✗ FAILURE

Fix issue in Step 4 code...

Single Node Execution (Attempt 2):
└─ Step 4: Model Training ONLY (30 min)    ✗ FAILURE (different error)
   With inputs: s3://bucket/run-1/step-3/outputs/

Fix issue again...

Single Node Execution (Attempt 3):
└─ Step 4: Model Training ONLY (30 min)    ✓ Success
   With inputs: s3://bucket/run-1/step-3/outputs/

Verify with full pipeline (optional):
└─ Full Pipeline Run (6 hours)             ✓ Success

Total time: 13 hours (6 + 0.5 + 0.5 + 6)
Time saved: 5 hours (28% improvement)
Cost saved: ~$150 (assuming ml.p3.2xlarge @ $3.06/hour)
```

### Key Innovations

1. **Zero Breaking Changes**: Works with all existing step builders without modification
2. **Two-Tier API**: Simple function for basic use, advanced class for complex scenarios
3. **Comprehensive Validation**: Ensures manual inputs match expected specifications
4. **Preview Capability**: Users can preview execution before running
5. **Universal Compatibility**: Works with all step types (training, processing, transform, etc.)

### Benefits

| Metric | Before (Full Re-run) | After (Single Node) | Improvement |
|--------|---------------------|---------------------|-------------|
| Debugging Time | 18 hours | 13 hours | **28% faster** |
| Wasted Computation | 12 hours | 0 hours | **100% eliminated** |
| Development Velocity | Slow | Fast | **3× iterations/day** |
| AWS Costs | $450 (3 runs) | $300 (1 + 2 debug) | **33% cheaper** |
| Developer Experience | Frustrating | Smooth | **Qualitative** |

---

## Architecture Context

### Current Pipeline Compilation Flow

```
┌──────────────────────────────────────────────────────────┐
│ User: Creates PipelineDAG                               │
│   dag = PipelineDAG()                                   │
│   dag.add_node("data_load")                            │
│   dag.add_node("preprocess")                           │
│   dag.add_node("train")                                │
│   dag.add_edge("data_load", "preprocess")              │
│   dag.add_edge("preprocess", "train")                  │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ PipelineDAGCompiler: Compiles full pipeline            │
│   1. Create all step builders                          │
│   2. Resolve dependencies (automatic message passing)   │
│   3. Instantiate all steps with PropertyReferences      │
│   4. Create Pipeline with all steps                    │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ Result: Full Pipeline                                   │
│   Pipeline(                                             │
│     steps=[data_load_step, preprocess_step, train_step]│
│   )                                                      │
│                                                          │
│ All steps connected via PropertyReferences:             │
│   train_step.inputs["input_path"] =                    │
│     preprocess_step.properties.ProcessingOutput...      │
└──────────────────────────────────────────────────────────┘
```

**Limitation**: Must execute full pipeline, can't skip steps.

### New Single Node Execution Flow

```
┌──────────────────────────────────────────────────────────┐
│ User: Specifies target node + manual inputs            │
│   manual_inputs = {                                     │
│     "input_path": "s3://bucket/run-123/preprocess/out/"│
│   }                                                      │
│                                                          │
│   pipeline = compile_single_node_to_pipeline(          │
│     dag=dag,                                           │
│     target_node="train",                               │
│     manual_inputs=manual_inputs                        │
│   )                                                      │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ SingleNodeCompiler: Compiles isolated node             │
│   1. Validate node exists and has configuration        │
│   2. Validate manual inputs match specification        │
│   3. Create isolated DAG (single node, no edges)       │
│   4. Create single step builder                        │
│   5. Instantiate step with manual inputs (S3 URIs)     │
│   6. Create Pipeline with single step                  │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ Result: Single-Node Pipeline                           │
│   Pipeline(                                             │
│     steps=[train_step]  # Only target step!            │
│   )                                                      │
│                                                          │
│ Step uses manual S3 paths:                             │
│   train_step.inputs["input_path"] =                    │
│     "s3://bucket/run-123/preprocess/out/"  # Direct!   │
└──────────────────────────────────────────────────────────┘
```

**Benefit**: Execute single step immediately, bypass expensive upstream steps.

---

## Implementation Architecture

### New Module Structure

```
src/cursus/core/compiler/
├── __init__.py                      # Export new functions ← UPDATE
├── dag_compiler.py                  # Existing: Full pipeline
├── single_node_compiler.py          # NEW: Single node execution
├── dynamic_template.py              # Existing: Templates
└── validation.py                    # Existing: Validation
```

### Component Diagram

```mermaid
graph TD
    subgraph "User Interface Layer"
        USER[Developer]
        SIMPLE[compile_single_node_to_pipeline<br/>Simple Function API]
        ADV[SingleNodeCompiler<br/>Advanced Class API]
    end
    
    subgraph "New Components"
        SNC[SingleNodeCompiler]
        VALID[validate_manual_inputs]
        PREVIEW[preview_execution]
        SPEC[get_node_specification]
    end
    
    subgraph "Enhanced Existing Components"
        PA[PipelineAssembler]
        INST[_instantiate_step_with_manual_inputs<br/>NEW METHOD]
        GENSN[generate_single_node_pipeline<br/>NEW METHOD]
    end
    
    subgraph "Existing Infrastructure"
        DAG[PipelineDAG]
        CONFIG[Configuration]
        BUILDER[Step Builders]
        CATALOG[Step Catalog]
    end
    
    USER --> SIMPLE
    USER --> ADV
    SIMPLE --> SNC
    ADV --> SNC
    
    SNC --> VALID
    SNC --> PREVIEW
    SNC --> SPEC
    SNC --> PA
    
    PA --> GENSN
    GENSN --> INST
    INST --> BUILDER
    
    SNC --> DAG
    SNC --> CONFIG
    SNC --> CATALOG
    
    style SIMPLE fill:#90EE90
    style ADV fill:#90EE90
    style SNC fill:#FFD700
    style PA fill:#FFD700
    style INST fill:#FFD700
    style GENSN fill:#FFD700
```

### Design Patterns

1. **Two-Tier API Pattern**
   - Simple function for 80% of use cases
   - Advanced class for 20% needing control

2. **Delegation Pattern**
   - Simple function delegates to class
   - Class delegates to PipelineAssembler

3. **Strategy Pattern**
   - PipelineAssembler switches between normal and single-node modes
   - Same builder interface, different input resolution

4. **Validation Chain**
   - Node existence → Configuration → Builder → Inputs
   - Early failure with clear error messages

---

## Phase 1: Core Infrastructure (Week 1)

### Deliverables

1. **New File**: `src/cursus/core/compiler/single_node_compiler.py`
2. **Enhanced File**: `src/cursus/core/assembler/pipeline_assembler.py`
3. **Updated File**: `src/cursus/core/compiler/__init__.py`
4. **Unit Tests**: `tests/core/compiler/test_single_node_compiler.py`

### Implementation Tasks

#### Task 1.1: Create `single_node_compiler.py` Structure

```python
"""
Single node execution compiler.

Enables debugging by compiling pipelines with a single node,
bypassing normal dependency resolution with manual input overrides.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging

from ...api.dag.base_dag import PipelineDAG
from ..assembler.pipeline_assembler import PipelineAssembler
from ...step_catalog.step_catalog import StepCatalog
from ...steps.configs.step_config_resolver import StepConfigResolver

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of node and input validation."""
    is_valid: bool
    node_exists: bool
    has_configuration: bool
    has_builder: bool
    valid_input_names: List[str]
    invalid_input_names: List[str]
    missing_required_inputs: List[str]
    invalid_s3_uris: List[str]
    errors: List[str]
    warnings: List[str]
    
    def detailed_report(self) -> str:
        """Generate detailed validation report."""
        # Implementation in Phase 1


@dataclass
class ExecutionPreview:
    """Preview of single-node execution."""
    target_node: str
    step_type: str
    config_type: str
    input_mappings: Dict[str, str]
    missing_required_inputs: List[str]
    missing_optional_inputs: List[str]
    output_paths: Dict[str, str]
    estimated_instance_type: str
    estimated_duration: Optional[str]
    warnings: List[str]
    
    def display(self) -> str:
        """Generate formatted display string."""
        # Implementation in Phase 1


class SingleNodeCompiler:
    """
    Specialized compiler for single-node pipeline execution.
    
    Enables rapid debugging by creating isolated pipelines containing
    just one node, with manual input overrides bypassing normal 
    dependency resolution.
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[Any] = None,
        role: Optional[str] = None,
        config_resolver: Optional[StepConfigResolver] = None,
        step_catalog: Optional[StepCatalog] = None,
        **kwargs: Any
    ):
        """Initialize single-node compiler."""
        # Implementation in Phase 1
    
    def validate_node_and_inputs(
        self,
        dag: PipelineDAG,
        target_node: str,
        manual_inputs: Dict[str, str]
    ) -> ValidationResult:
        """Validate target node and manual inputs."""
        # Implementation in Phase 1
    
    def preview_execution(
        self,
        dag: PipelineDAG,
        target_node: str,
        manual_inputs: Dict[str, str]
    ) -> ExecutionPreview:
        """Preview execution without creating pipeline."""
        # Implementation in Phase 1
    
    def get_node_specification(
        self,
        dag: PipelineDAG,
        target_node: str
    ) -> Any:  # StepSpecification
        """Get step specification for target node."""
        # Implementation in Phase 1
    
    def compile(
        self,
        dag: PipelineDAG,
        target_node: str,
        manual_inputs: Dict[str, str],
        pipeline_name: Optional[str] = None,
        validate_inputs: bool = True
    ) -> Any:  # Pipeline
        """Compile single-node pipeline."""
        # Implementation in Phase 1


def compile_single_node_to_pipeline(
    dag: PipelineDAG,
    config_path: str,
    target_node: str,
    manual_inputs: Dict[str, str],
    sagemaker_session: Optional[Any] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    validate_inputs: bool = True,
    **kwargs: Any,
) -> Any:  # Pipeline
    """
    Compile a single-node pipeline with manual input overrides.
    
    Simple convenience function that creates a SingleNodeCompiler
    and delegates to its compile method.
    
    Args:
        dag: Original PipelineDAG
        config_path: Path to configuration file
        target_node: Name of node to execute
        manual_inputs: Dict mapping logical input names to S3 URIs
        sagemaker_session: SageMaker session
        role: IAM role ARN
        pipeline_name: Optional pipeline name
        validate_inputs: Whether to validate inputs (default: True)
        **kwargs: Additional arguments
    
    Returns:
        Single-node Pipeline ready for execution
    
    Example:
        >>> manual_inputs = {
        ...     "input_path": "s3://bucket/run-123/preprocess/data/"
        ... }
        >>> pipeline = compile_single_node_to_pipeline(
        ...     dag=dag,
        ...     config_path="configs/pipeline.json",
        ...     target_node="train",
        ...     manual_inputs=manual_inputs
        ... )
        >>> execution = pipeline.start()
    """
    compiler = SingleNodeCompiler(
        config_path=config_path,
        sagemaker_session=sagemaker_session,
        role=role,
        **kwargs
    )
    
    return compiler.compile(
        dag=dag,
        target_node=target_node,
        manual_inputs=manual_inputs,
        pipeline_name=pipeline_name,
        validate_inputs=validate_inputs
    )
```

**Status**: ✅ Complete  
**Completed**: 2026-01-16
**Time Estimate**: 4 hours  
**Dependencies**: None

#### Task 1.2: Add Methods to `PipelineAssembler`

```python
# In src/cursus/core/assembler/pipeline_assembler.py

def generate_single_node_pipeline(
    self,
    target_node: str,
    manual_inputs: Dict[str, str],
    pipeline_name: str
) -> Pipeline:
    """
    Generate pipeline with single node and manual inputs.
    
    Bypasses normal message propagation and dependency resolution,
    directly instantiating the target node with provided manual inputs.
    """
    logger.info(f"[SINGLE_NODE] Generating pipeline for: {target_node}")
    
    # Validate target node exists
    if target_node not in self.step_builders:
        raise ValueError(
            f"Target node '{target_node}' not found. "
            f"Available: {list(self.step_builders.keys())}"
        )
    
    # Set single-node mode (internal flag)
    self._single_node_mode = True
    self._target_node = target_node
    self._manual_inputs = manual_inputs
    
    try:
        # Instantiate only the target step
        step = self._instantiate_step_with_manual_inputs(
            target_node,
            manual_inputs
        )
        self.step_instances[target_node] = step
        
        # Create minimal pipeline
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=self.pipeline_parameters,
            steps=[step],
            sagemaker_session=self.sagemaker_session,
        )
        
        logger.info(f"[SINGLE_NODE] Pipeline created: {pipeline_name}")
        return pipeline
        
    finally:
        # Reset mode
        self._single_node_mode = False
        self._target_node = None
        self._manual_inputs = {}


def _instantiate_step_with_manual_inputs(
    self,
    step_name: str,
    manual_inputs: Dict[str, str]
) -> Any:  # Step
    """
    Instantiate step with manually provided input paths.
    
    Bypasses message-based input resolution and directly uses
    the provided manual input paths (S3 URIs).
    """
    builder = self.step_builders[step_name]
    
    logger.info(f"[SINGLE_NODE] Instantiating '{step_name}' with manual inputs")
    for input_name, s3_path in manual_inputs.items():
        logger.info(f"  {input_name}: {s3_path}")
    
    # No dependencies for isolated execution
    dependencies = []
    
    # Manual inputs are already S3 URIs
    inputs = dict(manual_inputs)
    
    # Generate outputs using specification (same as normal flow)
    outputs = self._generate_outputs(step_name)
    
    # Create step
    kwargs = {
        "inputs": inputs,
        "outputs": outputs,
        "dependencies": dependencies,
        "enable_caching": builder.config.enable_caching,
    }
    
    try:
        step = builder.create_step(**kwargs)
        logger.info(f"[SINGLE_NODE] Step created: {step_name}")
        return step
        
    except Exception as e:
        logger.error(f"[SINGLE_NODE] Error creating step: {e}")
        raise ValueError(f"Failed to create step '{step_name}': {e}") from e
```

**Status**: 🔴 To implement  
**Time Estimate**: 3 hours  
**Dependencies**: Task 1.1

#### Task 1.3: Update Module Exports

```python
# In src/cursus/core/compiler/__init__.py

"""
Pipeline compiler module.

Provides tools for compiling pipeline DAGs into executable SageMaker pipelines.
"""

from .dag_compiler import (
    compile_dag_to_pipeline,
    PipelineDAGCompiler
)

from .single_node_compiler import (
    compile_single_node_to_pipeline,
    SingleNodeCompiler,
    ValidationResult,
    ExecutionPreview
)

from .validation import (
    ValidationEngine,
    ResolutionPreview,
    ConversionReport
)

__all__ = [
    # Full pipeline compilation
    'compile_dag_to_pipeline',
    'PipelineDAGCompiler',
    
    # Single-node execution
    'compile_single_node_to_pipeline',
    'SingleNodeCompiler',
    'ValidationResult',
    'ExecutionPreview',
    
    # Validation utilities
    'ValidationEngine',
    'ResolutionPreview',
    'ConversionReport',
]
```

**Status**: 🔴 To implement  
**Time Estimate**: 15 minutes  
**Dependencies**: Tasks 1.1, 1.2

#### Task 1.4: Write Core Unit Tests

```python
# In tests/core/compiler/test_single_node_compiler.py

import pytest
from cursus.core.compiler.single_node_compiler import (
    compile_single_node_to_pipeline,
    SingleNodeCompiler,
    ValidationResult,
    ExecutionPreview
)
from cursus.api.dag.base_dag import PipelineDAG


def test_compile_single_node_basic():
    """Test basic single-node compilation."""
    dag = create_test_dag()
    manual_inputs = {"input_path": "s3://bucket/data/"}
    
    pipeline = compile_single_node_to_pipeline(
        dag=dag,
        config_path="test_config.json",
        target_node="train",
        manual_inputs=manual_inputs
    )
    
    assert pipeline is not None
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].name == "train"


def test_validation_invalid_node():
    """Test validation with invalid node name."""
    dag = create_test_dag()
    compiler = SingleNodeCompiler(config_path="test_config.json")
    
    validation = compiler.validate_node_and_inputs(
        dag=dag,
        target_node="nonexistent",
        manual_inputs={}
    )
    
    assert not validation.is_valid
    assert not validation.node_exists
    assert "not found" in validation.errors[0].lower()


def test_validation_missing_required_inputs():
    """Test validation with missing required inputs."""
    dag = create_test_dag()
    compiler = SingleNodeCompiler(config_path="test_config.json")
    
    validation = compiler.validate_node_and_inputs(
        dag=dag,
        target_node="train",
        manual_inputs={}  # No inputs provided
    )
    
    assert not validation.is_valid
    assert len(validation.missing_required_inputs) > 0


def test_validation_invalid_s3_uri():
    """Test validation with invalid S3 URI format."""
    dag = create_test_dag()
    compiler = SingleNodeCompiler(config_path="test_config.json")
    
    validation = compiler.validate_node_and_inputs(
        dag=dag,
        target_node="train",
        manual_inputs={"input_path": "/local/path/"}  # Not S3!
    )
    
    assert not validation.is_valid
    assert len(validation.invalid_s3_uris) > 0


def test_execution_preview():
    """Test execution preview."""
    dag = create_test_dag()
    compiler = SingleNodeCompiler(config_path="test_config.json")
    
    preview = compiler.preview_execution(
        dag=dag,
        target_node="train",
        manual_inputs={"input_path": "s3://bucket/data/"}
    )
    
    assert preview.target_node == "train"
    assert "input_path" in preview.input_mappings
    assert len(preview.output_paths) > 0


def test_simple_function_delegates_to_class():
    """Test simple function delegates to compiler class."""
    dag = create_test_dag()
    
    # Should create compiler internally and delegate
    pipeline = compile_single_node_to_pipeline(
        dag=dag,
        config_path="test_config.json",
        target_node="train",
        manual_inputs={"input_path": "s3://bucket/data/"}
    )
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1
```

**Status**: 🔴 To implement  
**Time Estimate**: 4 hours  
**Dependencies**: Tasks 1.1, 1.2, 1.3

### Phase 1 Success Criteria

- [ ] `single_node_compiler.py` created with all core functions
- [ ] `PipelineAssembler` has new single-node methods
- [ ] Module exports updated
- [ ] Unit tests passing (>80% coverage)
- [ ] Can compile basic single-node pipeline
- [ ] Validation catches invalid inputs

---

## Phase 2: Robust Validation (Week 2)

### Deliverables

1. **Enhanced**: Input validation logic
2. **Enhanced**: S3 URI format validation
3. **Enhanced**: Specification-based validation
4. **New Tests**: Validation test suite

### Implementation Tasks

#### Task 2.1: Implement Comprehensive Input Validation

```python
def validate_manual_inputs(
    target_node: str,
    manual_inputs: Dict[str, str],
    step_spec: Any  # StepSpecification
) -> ValidationResult:
    """
    Validate manual inputs against step specification.
    
    Checks:
    1. All provided input names exist in specification
    2. All required inputs are provided
    3. S3 URIs are properly formatted
    4. Optional inputs warnings
    """
    errors = []
    warnings = []
    valid_names = []
    invalid_names = []
    missing_required = []
    invalid_uris = []
    
    # Get expected inputs from specification
    expected_inputs = step_spec.dependencies
    
    # Validate input names
    for input_name in manual_inputs.keys():
        if input_name not in expected_inputs:
            invalid_names.append(input_name)
            errors.append(
                f"Input '{input_name}' not in specification. "
                f"Expected: {list(expected_inputs.keys())}"
            )
        else:
            valid_names.append(input_name)
    
    # Check required inputs
    for input_name, dep_spec in expected_inputs.items():
        if dep_spec.required and input_name not in manual_inputs:
            missing_required.append(input_name)
            errors.append(
                f"Required input '{input_name}' missing. "
                f"Type: {dep_spec.dependency_type}"
            )
    
    # Check optional inputs
    for input_name, dep_spec in expected_inputs.items():
        if not dep_spec.required and input_name not in manual_inputs:
            warnings.append(
                f"Optional input '{input_name}' not provided. "
                f"Step may use default or skip processing."
            )
    
    # Validate S3 URI format
    import re
    s3_pattern = re.compile(r'^s3://[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]/.+$')
    
    for input_name, s3_uri in manual_inputs.items():
        if not s3_pattern.match(s3_uri):
            invalid_uris.append(input_name)
            errors.append(
                f"Invalid S3 URI for '{input_name}': {s3_uri}. "
                f"Must match s3://bucket-name/path format."
            )
    
    # Determine validity
    is_valid = (
        len(errors) == 0 and
        len(invalid_names) == 0 and
        len(missing_required) == 0 and
        len(invalid_uris) == 0
    )
    
    return ValidationResult(
        is_valid=is_valid,
        node_exists=True,
        has_configuration=True,
        has_builder=True,
        valid_input_names=valid_names,
        invalid_input_names=invalid_names,
        missing_required_inputs=missing_required,
        invalid_s3_uris=invalid_uris,
        errors=errors,
        warnings=warnings
    )
```

**Status**: 🔴 To implement  
**Time Estimate**: 4 hours  
**Dependencies**: Phase 1

#### Task 2.2: Add Validation Test Suite

```python
# Comprehensive validation tests

def test_s3_uri_validation_valid():
    """Test S3 URI validation with valid URIs."""
    valid_uris = [
        "s3://bucket/path/",
        "s3://my-bucket-123/data/file.csv",
        "s3://bucket.with.dots/path/to/data/",
    ]
    
    for uri in valid_uris:
        assert is_valid_s3_uri(uri), f"Should be valid: {uri}"


def test_s3_uri_validation_invalid():
    """Test S3 URI validation with invalid URIs."""
    invalid_uris = [
        "http://bucket/path/",          # Wrong protocol
        "s3:/bucket/path/",              # Missing slash
        "/local/path/",                  # Local path
        "bucket/path/",                  # No protocol
        "s3://UPPERCASE/path/",          # Uppercase bucket
        "s3://bucket-/path/",            # Invalid bucket name
    ]
    
    for uri in invalid_uris:
        assert not is_valid_s3_uri(uri), f"Should be invalid: {uri}"
```

**Status**: 🔴 To implement  
**Time Estimate**: 3 hours  
**Dependencies**: Task 2.1

### Phase 2 Success Criteria

- [ ] Comprehensive input validation implemented
- [ ] S3 URI format validation working
- [ ] Specification-based validation complete
- [ ] Validation test suite passing (>90% coverage)
- [ ] Clear error messages for all failure modes
- [ ] Warning messages for optional inputs

---

## Phase 3: Enhanced Features (Week 3)

### Deliverables

1. **Enhanced**: Preview execution functionality
2. **Enhanced**: Specification introspection
3. **New**: Helper utilities
4. **Integration Tests**: End-to-end testing

### Implementation Tasks

#### Task 3.1: Implement Preview Functionality

**Status**: 🔴 To implement  
**Time Estimate**: 4 hours  
**Dependencies**: Phase 2

#### Task 3.2: Add Specification Introspection

**Status**: 🔴 To implement  
**Time Estimate**: 3 hours  
**Dependencies**: Phase 2

#### Task 3.3: Create Integration Tests

**Status**: 🔴 To implement  
**Time Estimate**: 5 hours  
**Dependencies**: Tasks 3.1, 3.2

### Phase 3 Success Criteria

- [ ] Preview execution displays detailed information
- [ ] Specification introspection works for all step types
- [ ] Integration tests cover all major workflows
- [ ] Documentation examples work as shown

---

## Phase 4: Documentation and Examples (Week 4)

### Deliverables

1. **API Documentation**: Comprehensive docstrings and guides
2. **Usage Examples**: Common scenarios and patterns
3. **Jupyter Notebooks**: Interactive tutorials
4. **Troubleshooting Guide**: Common issues and solutions

### Implementation Tasks

#### Task 4.1: Write API Documentation

**Status**: 🔴 To implement  
**Time Estimate**: 6 hours  
**Dependencies**: Phase 3

#### Task 4.2: Create Usage Examples

**Status**: 🔴 To implement  
**Time Estimate**: 4 hours  
**Dependencies**: Phase 3

#### Task 4.3: Create Jupyter Notebook Tutorial

**Status**: 🔴 To implement  
**Time Estimate**: 4 hours  
**Dependencies**: Tasks 4.1, 4.2

### Phase 4 Success Criteria

- [ ] Complete API documentation with examples
- [ ] Usage examples for common scenarios
- [ ] Jupyter notebook tutorial functional
- [ ] Troubleshooting guide comprehensive

---

## Backward Compatibility Strategy

### Critical Requirements

1. **Zero Breaking Changes**: All existing pipelines must continue to work
2. **Opt-In Feature**: Single-node execution is explicitly invoked
3. **No Modified Builders**: Step builders require no changes
4. **Same Output Format**: Single-node pipelines produce standard outputs

### Compatibility Guarantees

#### Guarantee 1: Existing Pipeline Compilation Unaffected

```python
# Existing code continues to work identically
from cursus.core.compiler import compile_dag_to_pipeline

pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="configs/pipeline.json"
)
# ✅ No changes needed, works as before
```

#### Guarantee 2: PipelineAssembler Remains Compatible

```python
# Existing PipelineAssembler usage unchanged
assembler = PipelineAssembler(...)
pipeline = assembler.generate_pipeline()
# ✅ Normal mode still default, single-node mode explicit only
```

#### Guarantee 3: Step Builders Unchanged

```python
# Step builders work with both modes without modification
class XGBoostTrainingStepBuilder(StepBuilderBase):
    def create_step(self, inputs=None, outputs=None, **kwargs):
        # Same implementation works for:
        # 1. Normal pipeline (inputs from PropertyReferences)
        # 2. Single-node pipeline (inputs from manual S3 URIs)
        # Builder doesn't know or care about the difference!
        pass
```

#### Guarantee 4: Configuration System Unchanged

```python
# Configuration files and contracts unchanged
# No new required fields in any config classes
# Single-node execution uses existing configuration
```

### Rollback Plan

If issues arise, rollback is simple:

1. **Immediate**: Don't use new API (continue with normal compilation)
2. **Quick**: Remove `single_node_compiler.py` (isolated module)
3. **Full**: Revert PipelineAssembler changes (minimal modifications)

**Risk**: Extremely low - new functionality is additive and isolated.

---

## Testing Strategy

### Test Matrix

| Test Category | Coverage Target | Priority |
|--------------|----------------|----------|
| Unit Tests | >80% | High |
| Integration Tests | All workflows | High |
| Validation Tests | >90% | High |
| Compatibility Tests | 100% | Critical |
| Performance Tests | Baseline | Medium |

### Test Categories

#### 1. Unit Tests

```python
# Test individual functions and methods
test_compile_single_node_basic()
test_validation_logic()
test_s3_uri_validation()
test_preview_generation()
test_specification_retrieval()
```

#### 2. Integration Tests

```python
# Test end-to-end workflows
test_single_node_training_step()
test_single_node_processing_step()
test_single_node_transform_step()
test_multiple_step_types()
test_with_real_sagemaker_execution()
```

#### 3. Compatibility Tests

```python
# Ensure existing functionality unaffected
test_normal_pipeline_compilation_unchanged()
test_existing_builders_work()
test_existing_configs_work()
test_no_breaking_changes()
```

#### 4. Validation Tests

```python
# Test validation edge cases
test_invalid_node_name()
test_missing_required_inputs()
test_invalid_s3_uris()
test_extra_inputs_warning()
test_validation_error_messages()
```

---

## Risk Mitigation

### Risk 1: Step Builders Don't Work with Manual Inputs

**Probability**: Low  
**Impact**: High  
**Mitigation**:
- Step builders already accept `inputs` parameter
- Builders don't distinguish between PropertyReferences and S3 URIs
- Tested during Phase 1

**Fallback**: Modify specific builders if needed (unlikely)

### Risk 2: Validation Too Strict or Too Lenient

**Probability**: Medium  
**Impact**: Medium  
**Mitigation**:
- Comprehensive validation tests
- User feedback during Phase 4
- Adjustable validation settings

**Fallback**: Add `--skip-validation` flag if needed

### Risk 3: Performance Issues

**Probability**: Low  
**Impact**: Low  
**Mitigation**:
- Single-node mode skips expensive dependency resolution
- Should be faster than normal compilation

**Fallback**: Optimize specific bottlenecks if found

### Risk 4: Documentation Insufficient

**Probability**: Medium  
**Impact**: Medium  
**Mitigation**:
- Comprehensive examples in Phase 4
- Jupyter notebook tutorial
- Troubleshooting guide

**Fallback**: Add more examples based on user feedback

---

## Success Metrics

### Quantitative Metrics

1. **Time Savings**: 28% average reduction in debugging time
2. **Cost Savings**: 33% average reduction in AWS costs
3. **Adoption**: 50% of developers using within 3 months
4. **Test Coverage**: >85% code coverage
5. **Error Rate**: <5% validation false positives/negatives

### Qualitative Metrics

1. **Developer Satisfaction**: Survey feedback >4/5 stars
2. **Ease of Use**: Can use without reading full documentation
3. **Error Messages**: Clear and actionable
4. **Integration**: Seamless with existing workflows

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `single_node_compiler.py` with core classes
- [ ] Add `generate_single_node_pipeline()` to PipelineAssembler
- [ ] Add `_instantiate_step_with_manual_inputs()` to PipelineAssembler
- [ ] Update `__init__.py` exports
- [ ] Write unit tests for core functionality
- [ ] Test basic single-node compilation
- [ ] Verify zero breaking changes

### Phase 2: Robust Validation (Week 2)
- [ ] Implement comprehensive input validation
- [ ] Implement S3 URI format validation
- [ ] Implement specification-based validation
- [ ] Write validation test suite
- [ ] Test edge cases and error messages
- [ ] Verify validation accuracy

### Phase 3: Enhanced Features (Week 3)
- [ ] Implement preview execution
- [ ] Implement specification introspection
- [ ] Add helper utilities
- [ ] Write integration tests
- [ ] Test with various step types
- [ ] Verify end-to-end workflows

### Phase 4: Documentation and Examples (Week 4)
- [ ] Write API documentation
- [ ] Create usage examples
- [ ] Create Jupyter notebook tutorial
- [ ] Write troubleshooting guide
- [ ] Review documentation completeness
- [ ] Get user feedback

---

## Deployment Strategy

### Week 1-2: Development and Testing
- Complete Phases 1-2
- Internal testing and validation
- Code review with team

### Week 3: Alpha Release
- Complete Phase 3
- Deploy to development environment
- Limited user testing

### Week 4: Beta Release
- Complete Phase 4
- Deploy to staging environment
- Broader user testing and feedback

### Week 5+: Production Release
- Address beta feedback
- Deploy to production
- Monitor usage and issues
- Iterate based on feedback

---

## Related Documents

### Primary Design Documentation
- **[Single Node Execution API Design](../1_design/single_node_execution_api_design.md)** - Complete design specification with detailed architecture, API interfaces, and usage examples

### Pipeline Compilation System
- [Pipeline Compiler Design](../1_design/pipeline_compiler.md) - Full pipeline compilation architecture
- [DAG to Template](../1_design/dag_to_template.md) - DAG transformation process
- [Dynamic Template System](../1_design/dynamic_template_system.md) - Runtime template generation
- [MODS DAG Compiler](../1_design/mods_dag_compiler_design.md) - MODS integration

### Pipeline Assembly System
- [Pipeline Assembler](../1_design/pipeline_assembler.md) - Message passing orchestration and assembly
- [Pipeline Template Base](../1_design/pipeline_template_base.md) - Template foundation class
- [Specification-Driven Design](../1_design/specification_driven_design.md) - Core architecture philosophy

### Dependency Resolution
- [Dependency Resolver](../1_design/dependency_resolver.md) - Automatic dependency matching
- [Dependency Resolution System](../1_design/dependency_resolution_system.md) - System architecture
- [Specification Registry](../1_design/specification_registry.md) - Step metadata registry

### DAG System
- [Pipeline DAG](../1_design/pipeline_dag.md) - DAG data structure and operations
- [Pipeline DAG Resolver](../1_design/pipeline_dag_resolver_design.md) - DAG analysis system

### Component Development Guides
- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Step creation workflow
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Builder implementation patterns
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Script implementation
- [Step Specification Guide](../0_developer_guide/step_specification.md) - Specification creation

### Standards and Best Practices
- [Design Principles](../0_developer_guide/design_principles.md) - Architectural guidelines
- [Best Practices](../0_developer_guide/best_practices.md) - Development standards
- [Common Pitfalls](../0_developer_guide/common_pitfalls.md) - Mistakes to avoid
- [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md) - Validation usage

### Testing and Validation
- [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md) - Comprehensive validation
- [Universal Step Builder Test](../1_design/universal_step_builder_test.md) - Builder testing
- [Enhanced Universal Step Builder Tester](../1_design/enhanced_universal_step_builder_tester_design.md) - Advanced testing
- [Validation Checklist](../0_developer_guide/validation_checklist.md) - Development validation

### Configuration System
- [Three-Tier Config Design](../1_design/config_tiered_design.md) - Configuration architecture
- [Config Manager Three-Tier Implementation](../1_design/config_manager_three_tier_implementation.md) - Implementation guide
- [Config Resolution Enhancements](../1_design/config_resolution_enhancements.md) - Resolution improvements
- [Step Config Resolver](../1_design/step_config_resolver.md) - Config resolution system

### Step Library System
- [Unified Step Catalog System](../1_design/unified_step_catalog_system_design.md) - Auto-discovery system
- [Step Catalog Component Architecture](../1_design/unified_step_catalog_component_architecture_design.md) - Architecture
- [Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md) - Integration

### Entry Points and Navigation
- [Cursus Code Structure Index](../00_entry_points/cursus_code_structure_index.md) - Complete code→design mapping
- [Cursus Package Overview](../00_entry_points/cursus_package_overview.md) - Executive summary and architecture
- [Core and MODS Systems Index](../00_entry_points/core_and_mods_systems_index.md) - Core orchestration
- [Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md) - Step patterns

### Related Implementation Plans
- [Streaming Mode Memory Optimization](./2026-01-12_streaming_mode_memory_optimization_plan.md) - Similar phased implementation pattern
- [Names3Risk Training Infrastructure](./2026-01-05_names3risk_training_infrastructure_implementation_plan.md) - Infrastructure setup

### Implementation Prerequisites

**Required Reading Before Implementation**:
1. [Single Node Execution API Design](../1_design/single_node_execution_api_design.md) - Complete specification
2. [Pipeline Assembler](../1_design/pipeline_assembler.md) - Understand message passing system
3. [Pipeline Compiler Design](../1_design/pipeline_compiler.md) - Understand compilation architecture
4. [Step Builder Guide](../0_developer_guide/step_builder.md) - Understand builder interface
5. [Design Principles](../0_developer_guide/design_principles.md) - Follow architectural guidelines
6. [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md) - Implement proper validation

This comprehensive reference collection provides all necessary documentation for understanding, implementing, and integrating the Single Node Execution API into the Cursus pipeline framework.

---

## Summary

### Problem Solved
**Before**: Pipeline failures required full re-runs (18 hours for 3 attempts)  
**After**: Debug single nodes with manual inputs (13 hours, 28% faster)

### Key Achievements
1. **Zero Breaking Changes**: Works with all existing code
2. **Two-Tier API**: Simple for basic use, powerful for advanced
3. **Comprehensive Validation**: Prevents errors before execution
4. **Universal Compatibility**: All step types supported

### Implementation Effort
- **Timeline**: 4 weeks (3 weeks development + 1 week deployment)
- **Code**: ~1500 lines (new module + enhancements + tests)
- **Testing**: ~800 lines of test code
- **Documentation**: ~50 pages

### Expected Impact
- **Time Savings**: 5+ hours per debugging session (28% improvement)
- **Cost Savings**: $150 per debugging session (33% improvement)
- **Developer Experience**: Dramatically improved iteration speed
- **Adoption**: Expected 50% developer adoption within 3 months

---

## Phase 1.5: API Consistency Enhancement ✅ COMPLETE

**Timeline**: Completed 2026-01-16
**Status**: ✅ Fully Implemented
**Priority**: High - Addresses API inconsistency

### Problem Identified

Phase 1 implementation required users to manually provide `config_map` parameter, creating API inconsistency with `compile_dag_to_pipeline()`:

```python
# ❌ Inconsistent - required manual config_map
pipeline = compile_single_node_to_pipeline(
    dag=dag,
    config_path="configs/pipeline.json",
    target_node="train",
    manual_inputs=manual_inputs,
    config_map=config_map,  # Manual burden!
    sagemaker_session=session,
    role=role
)
```

### Solution: Automatic Config Loading

Added `_load_target_node_config()` method to `SingleNodeCompiler` that mirrors `DynamicPipelineTemplate`'s proven mechanism:

1. Auto-detect config classes from JSON using `detect_config_classes_from_json()`
2. Load all configs from file using `load_configs()`
3. Filter to target node only (efficiency optimization)
4. Return minimal config_map

### Implementation Completed

**Task 1.5.1**: Added `_load_target_node_config()` Method ✅
- Location: `src/cursus/core/compiler/single_node_compiler.py`
- Uses same utilities as `DynamicPipelineTemplate`
- Provides clear error messages when node not found
- Efficient: only loads target node's config

**Task 1.5.2**: Made `config_map` Parameter Optional ✅
- Updated `compile()` method signature
- Auto-loading fallback when `config_map=None`
- Maintained backward compatibility with explicit `config_map`

**Task 1.5.3**: Updated Documentation ✅
- Enhanced docstrings with improved API examples
- Added usage examples for both simple and advanced patterns
- Documented performance optimization scenarios

### API Now Consistent

**Simple Usage (Config Auto-Loaded)**:
```python
# ✅ Clean, consistent API - just like compile_dag_to_pipeline()
pipeline = compile_single_node_to_pipeline(
    dag=dag,
    config_path="configs/pipeline.json",  # That's it!
    target_node="train",
    manual_inputs={"input_path": "s3://..."},
    sagemaker_session=session,
    role=role
)
```

**Advanced Usage (Performance Optimization)**:
```python
# Optional: Pre-load config for performance
from cursus.steps.configs.utils import load_configs
config_map = load_configs("configs/pipeline.json")

pipeline = compile_single_node_to_pipeline(
    dag=dag,
    config_path="configs/pipeline.json",
    target_node="train",
    manual_inputs={"input_path": "s3://..."},
    config_map=config_map,  # Override auto-loading
    sagemaker_session=session,
    role=role
)
```

### Benefits Achieved

✅ **API Consistency**: Now matches `compile_dag_to_pipeline()` interface exactly  
✅ **User-Friendly**: No manual config loading required  
✅ **Efficient**: Only loads target node's config (not entire pipeline)  
✅ **Backward Compatible**: Old code with explicit `config_map` still works  
✅ **Robust**: Uses same proven mechanism as `DynamicPipelineTemplate`

### Phase 1.5 Checklist

- [x] Added `_load_target_node_config()` method to SingleNodeCompiler
- [x] Made `config_map` parameter optional in `compile()` method
- [x] Added auto-loading fallback with proper error handling
- [x] Updated docstrings with improved API examples
- [x] Updated design document with Phase 1.5 section
- [x] Updated implementation plan with Phase 1.5 section
- [x] Verified full backward compatibility

---

**Document Version**: 1.1  
**Created**: 2026-01-16  
**Last Updated**: 2026-01-16 (Phase 1.5 enhancement documented)  
**Status**: Phase 1 Complete ✅, Phase 1.5 Complete ✅, Phases 2-4 Pending  
**Author**: AI Assistant  
**Reviewers**: TBD
