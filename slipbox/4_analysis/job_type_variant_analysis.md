---
tags:
  - analysis
  - job_type
  - step_catalog
  - component_mapping
keywords:
  - job_type variants
  - component sharing
  - step specifications
  - unified catalog
topics:
  - job type analysis
  - component mapping patterns
  - step catalog integration
language: python
date of note: 2025-09-10
---

# Job Type Variant Analysis for Unified Step Catalog System

## Overview

This analysis examines how job_type variants work in the cursus system and their implications for the unified step catalog system design. The key finding is that **multiple step specifications with different job_types share the same underlying components** (scripts, contracts, configs, builders).

## Job Type Pattern Discovery

### Core Pattern: One-to-Many Component Mapping

**Key Insight**: One set of core components (script, contract, config, builder) maps to multiple step specifications differentiated by job_type.

```
Base Step: "CradleDataLoading"
├── Script: cradle_data_loading.py (shared)
├── Contract: cradle_data_loading_contract.py (shared)  
├── Config: config_cradle_data_loading_step.py (shared)
├── Builder: CradleDataLoadingStepBuilder (shared)
└── Specs: (job_type variants)
    ├── cradle_data_loading_spec.py (base/default)
    ├── cradle_data_loading_training_spec.py (job_type="training")
    ├── cradle_data_loading_validation_spec.py (job_type="validation")
    ├── cradle_data_loading_testing_spec.py (job_type="testing")
    └── cradle_data_loading_calibration_spec.py (job_type="calibration")
```

### Evidence from File Structure Analysis

#### Specs Directory Pattern
```
src/cursus/steps/specs/
├── cradle_data_loading_spec.py                    # Base spec
├── cradle_data_loading_training_spec.py           # job_type="training"
├── cradle_data_loading_validation_spec.py         # job_type="validation"
├── cradle_data_loading_testing_spec.py            # job_type="testing"
├── cradle_data_loading_calibration_spec.py        # job_type="calibration"
├── tabular_preprocessing_spec.py                  # Base spec
├── tabular_preprocessing_training_spec.py         # job_type="training"
├── tabular_preprocessing_validation_spec.py       # job_type="validation"
├── tabular_preprocessing_testing_spec.py          # job_type="testing"
├── tabular_preprocessing_calibration_spec.py      # job_type="calibration"
└── ... (similar patterns for other steps)
```

#### Shared Components Pattern
```
src/cursus/steps/
├── scripts/
│   ├── tabular_preprocessing.py                   # Shared across all job_types
│   └── ... (one script per base step type)
├── contracts/
│   ├── cradle_data_loading_contract.py            # Shared across all job_types
│   ├── tabular_preprocess_contract.py             # Shared across all job_types
│   └── ... (one contract per base step type)
├── configs/
│   ├── config_cradle_data_loading_step.py         # Shared across all job_types
│   ├── config_tabular_preprocessing_step.py       # Shared across all job_types
│   └── ... (one config per base step type)
└── builders/
    └── ... (one builder per base step type)
```

### Job Type Implementation Details

#### 1. Step Type Generation with Job Type
From `src/cursus/registry/step_names.py`:

```python
def get_spec_step_type_with_job_type(step_name: str, job_type: str = None, workspace_id: str = None) -> str:
    """Get step_type with optional job_type suffix, workspace-aware."""
    base_type = get_spec_step_type(step_name, workspace_id)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type
```

**Pattern**: `{BaseStepType}_{JobType}` (e.g., `CradleDataLoading_Training`)

#### 2. Spec Implementation Example
From `cradle_data_loading_training_spec.py`:

```python
DATA_LOADING_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("CradleDataLoading", "training"),
    node_type=NodeType.SOURCE,
    # Same dependencies and outputs as base spec, but with job_type-specific semantics
    outputs=[
        OutputSpec(
            logical_name="DATA",
            # ... same property_path as base spec
            description="Training data output from Cradle data loading",
            semantic_keywords=["training", "train", "data", ...]  # Job-type specific keywords
        )
    ]
)
```

#### 3. Config Job Type Support
From `config_cradle_data_loading_step.py`:

```python
class CradleDataLoadConfig(BasePipelineConfig):
    job_type: str = Field(
        description="One of ['training','validation','testing','calibration']"
    )
    
    # Same config structure regardless of job_type
    # job_type affects runtime behavior, not config structure
```

## Implications for Unified Step Catalog System

### 1. Component Indexing Strategy

**Challenge**: How to index components that are shared across multiple job_type variants?

**Solution**: Hierarchical indexing with base step types and job_type variants:

```python
class StepInfo(BaseModel):
    # Base step information
    base_step_name: str  # e.g., "CradleDataLoading"
    step_type: str       # e.g., "CradleDataLoading" (base) or "CradleDataLoading_Training" (variant)
    job_type: Optional[str] = None  # e.g., "training", "validation", etc.
    
    # Component paths (shared across job_types for same base step)
    script_path: Optional[Path] = None
    contract_path: Optional[Path] = None
    config_path: Optional[Path] = None
    builder_path: Optional[Path] = None
    
    # Spec path (unique per job_type variant)
    spec_path: Optional[Path] = None
    
    @property
    def is_job_type_variant(self) -> bool:
        """Check if this is a job_type variant of a base step."""
        return self.job_type is not None
    
    @property
    def base_step_key(self) -> str:
        """Get the base step key for component sharing."""
        return self.base_step_name
    
    @property
    def variant_key(self) -> str:
        """Get the unique variant key."""
        if self.job_type:
            return f"{self.base_step_name}_{self.job_type.capitalize()}"
        return self.base_step_name
```

### 2. Discovery Algorithm Enhancement

**Enhanced Discovery Pattern**:

```python
class UnifiedStepCatalog:
    def discover_components(self) -> Dict[str, StepInfo]:
        """Discover all components with job_type variant support."""
        
        # Step 1: Discover base components (scripts, contracts, configs, builders)
        base_components = self._discover_base_components()
        
        # Step 2: Discover all spec variants
        spec_variants = self._discover_spec_variants()
        
        # Step 3: Map specs to base components
        step_catalog = {}
        
        for spec_info in spec_variants:
            base_step_name = self._extract_base_step_name(spec_info.name)
            job_type = self._extract_job_type(spec_info.name)
            
            # Get shared components for this base step
            base_components_for_step = base_components.get(base_step_name, {})
            
            # Create StepInfo with shared components + unique spec
            step_info = StepInfo(
                base_step_name=base_step_name,
                step_type=spec_info.step_type,
                job_type=job_type,
                script_path=base_components_for_step.get('script'),
                contract_path=base_components_for_step.get('contract'),
                config_path=base_components_for_step.get('config'),
                builder_path=base_components_for_step.get('builder'),
                spec_path=spec_info.path
            )
            
            step_catalog[step_info.variant_key] = step_info
        
        return step_catalog
```

### 3. Completeness Validation Update

**Enhanced is_complete Logic**:

```python
def is_complete(self) -> bool:
    """Check if step has all required components based on SageMaker step type and job_type pattern."""
    
    # Get base step requirements (same for all job_type variants)
    base_requirements = self._get_base_step_requirements()
    
    # Check base components (shared across job_types)
    for component in base_requirements:
        if component == 'script' and not self.script_path:
            return False
        elif component == 'contract' and not self.contract_path:
            return False
        elif component == 'config' and not self.config_path:
            return False
        elif component == 'builder' and not self.builder_path:
            return False
    
    # Spec is always required (unique per job_type variant)
    if not self.spec_path:
        return False
    
    return True
```

### 4. Index Structure Optimization

**Two-Level Indexing**:

```python
class StepCatalogIndex:
    def __init__(self):
        # Base step index (for shared components)
        self.base_steps: Dict[str, BaseStepComponents] = {}
        
        # Variant index (for job_type-specific specs)
        self.step_variants: Dict[str, StepInfo] = {}
        
        # Reverse lookup indexes
        self.file_to_step: Dict[Path, List[str]] = {}  # File -> list of step variants using it
        self.job_type_groups: Dict[str, List[str]] = {}  # job_type -> list of step variants
```

## PipelineDAG Node Naming Pattern

### Node Names with Job Type Suffixes

From the PipelineDAG analysis in `src/cursus/pipeline_catalog/shared_dags/xgboost/simple_dag.py`:

```python
def create_xgboost_simple_dag() -> PipelineDAG:
    """Create a simple XGBoost training pipeline DAG."""
    dag = PipelineDAG()
    
    # Add nodes with job_type suffixes in node names
    dag.add_node("CradleDataLoading_training")       # job_type="training"
    dag.add_node("TabularPreprocessing_training")    # job_type="training"
    dag.add_node("XGBoostTraining")                  # No job_type (base step)
    dag.add_node("CradleDataLoading_calibration")    # job_type="calibration"
    dag.add_node("TabularPreprocessing_calibration") # job_type="calibration"
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")
    
    # Calibration flow (independent of training)
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
```

**Key Pattern**: Node names follow `{BaseStepName}_{job_type}` format, enabling:
- Clear separation of data flows by job_type
- Independent processing paths for different purposes
- Semantic matching between steps with same job_type

## Integration with Existing Systems

### 1. Builder Registry Integration

**Enhanced Builder Resolution**:

```python
class UnifiedStepCatalog(StepBuilderRegistry):
    def get_builder_for_config_with_job_type(self, config, job_type: str = None):
        """Get builder considering job_type variants."""
        
        # Extract base step name from config
        base_step_name = self._get_base_step_name_from_config(config)
        
        # All job_type variants of the same base step use the same builder
        return self.get_builder_for_step_type(base_step_name)
```

### 2. Config Resolver Integration

**Enhanced Resolution with Job Type Awareness**:

```python
class EnhancedConfigResolver(StepConfigResolver):
    def resolve_with_job_type_support(self, dag_nodes, available_configs):
        """Resolve configs with job_type variant support."""
        
        # Enhanced node name parsing to extract job_type
        for node_name in dag_nodes:
            base_name, job_type = self._parse_node_name_with_job_type(node_name)
            
            # Find matching config with same job_type
            matching_configs = [
                config for config in available_configs.values()
                if (hasattr(config, 'job_type') and config.job_type == job_type) or
                   (job_type is None and not hasattr(config, 'job_type'))
            ]
            
            # Apply existing resolution strategies to filtered configs
            resolved_config = self._resolve_single_config(base_name, matching_configs)
            
        return resolved_map
    
    def _parse_node_name_with_job_type(self, node_name: str) -> Tuple[str, Optional[str]]:
        """Parse node name to extract base name and job_type."""
        job_type_suffixes = ['_training', '_calibration', '_validation', '_testing']
        
        for suffix in job_type_suffixes:
            if node_name.endswith(suffix):
                base_name = node_name[:-len(suffix)]
                job_type = suffix[1:]  # Remove leading underscore
                return base_name, job_type
        
        return node_name, None
```

## Related Documentation

This analysis builds upon and references the following existing documentation:

### 1. Job Type Variant Solution (2025-07-04)
**Reference**: [Job Type Variant Solution](../2_project_planning/2025-07-04_job_type_variant_solution.md)

**Key Contributions**:
- Complete implementation of job_type variant specifications
- Dynamic specification selection in step builders
- Environment variable contract enforcement
- Pipeline template integration with job_type awareness

### 2. Job Type Variant Handling Design
**Reference**: [Job Type Variant Handling](../1_design/job_type_variant_handling.md)

**Key Contributions**:
- Comprehensive design for job_type variant handling
- Step name generation with job_type suffixes
- Semantic keyword differentiation by job_type
- Dependency resolution with job_type normalization
- Configuration system integration patterns

### 3. Implementation Evidence
**Reference**: `src/cursus/pipeline_catalog/shared_dags/xgboost/simple_dag.py`

**Key Evidence**:
- Actual PipelineDAG node naming with job_type suffixes
- Separation of training and calibration data flows
- Independent processing paths for different job_types

## Implications for Unified Step Catalog System

### 1. Enhanced Component Indexing Strategy

**Updated StepInfo Model**:

```python
class StepInfo(BaseModel):
    # Base step information
    base_step_name: str  # e.g., "CradleDataLoading"
    step_type: str       # e.g., "CradleDataLoading" (base) or "CradleDataLoading_Training" (variant)
    job_type: Optional[str] = None  # e.g., "training", "validation", etc.
    
    # Component paths (shared across job_types for same base step)
    script_path: Optional[Path] = None
    contract_path: Optional[Path] = None
    config_path: Optional[Path] = None
    builder_path: Optional[Path] = None
    
    # Spec path (unique per job_type variant)
    spec_path: Optional[Path] = None
    
    @property
    def is_job_type_variant(self) -> bool:
        """Check if this is a job_type variant of a base step."""
        return self.job_type is not None
    
    @property
    def base_step_key(self) -> str:
        """Get the base step key for component sharing."""
        return self.base_step_name
    
    @property
    def variant_key(self) -> str:
        """Get the unique variant key matching PipelineDAG node names."""
        if self.job_type:
            return f"{self.base_step_name}_{self.job_type}"
        return self.base_step_name
    
    @property
    def pipeline_node_name(self) -> str:
        """Get the node name as used in PipelineDAG."""
        return self.variant_key  # Same as variant_key for consistency
```

### 2. Discovery Algorithm with Job Type Support

**Enhanced Discovery Pattern**:

```python
class UnifiedStepCatalog:
    def discover_components_with_job_types(self) -> Dict[str, StepInfo]:
        """Discover all components with comprehensive job_type variant support."""
        
        # Step 1: Discover base components (scripts, contracts, configs, builders)
        base_components = self._discover_base_components()
        
        # Step 2: Discover all spec variants (including job_type variants)
        spec_variants = self._discover_spec_variants_with_job_types()
        
        # Step 3: Map specs to base components with job_type awareness
        step_catalog = {}
        
        for spec_info in spec_variants:
            base_step_name = self._extract_base_step_name(spec_info.name)
            job_type = self._extract_job_type_from_spec(spec_info.name)
            
            # Get shared components for this base step
            base_components_for_step = base_components.get(base_step_name, {})
            
            # Create StepInfo with shared components + unique spec
            step_info = StepInfo(
                base_step_name=base_step_name,
                step_type=spec_info.step_type,
                job_type=job_type,
                script_path=base_components_for_step.get('script'),
                contract_path=base_components_for_step.get('contract'),
                config_path=base_components_for_step.get('config'),
                builder_path=base_components_for_step.get('builder'),
                spec_path=spec_info.path
            )
            
            # Use pipeline node name as key for consistency with DAG
            step_catalog[step_info.pipeline_node_name] = step_info
        
        return step_catalog
    
    def _extract_job_type_from_spec(self, spec_name: str) -> Optional[str]:
        """Extract job_type from spec file name."""
        # Pattern: {base_name}_{job_type}_spec.py
        job_type_patterns = ['_training_spec', '_calibration_spec', '_validation_spec', '_testing_spec']
        
        for pattern in job_type_patterns:
            if spec_name.endswith(pattern):
                job_type = pattern.split('_')[1]  # Extract job_type part
                return job_type
        
        return None  # Base spec without job_type
```

### 3. Index Structure with Job Type Optimization

**Three-Level Indexing**:

```python
class StepCatalogIndex:
    def __init__(self):
        # Base step index (for shared components)
        self.base_steps: Dict[str, BaseStepComponents] = {}
        
        # Variant index (for job_type-specific specs)
        self.step_variants: Dict[str, StepInfo] = {}
        
        # Job type grouping (for pipeline variant creation)
        self.job_type_groups: Dict[str, List[str]] = {}
        
        # Pipeline node name index (for DAG integration)
        self.pipeline_nodes: Dict[str, StepInfo] = {}
        
        # Reverse lookup indexes
        self.file_to_steps: Dict[Path, List[str]] = {}  # File -> list of step variants using it
        self.base_to_variants: Dict[str, List[str]] = {}  # Base step -> list of variants
```

### 4. Pipeline Integration Support

**DAG Node Resolution**:

```python
def resolve_pipeline_dag_nodes(self, dag: PipelineDAG) -> Dict[str, StepInfo]:
    """Resolve PipelineDAG nodes to StepInfo objects."""
    resolved_nodes = {}
    
    for node_name in dag.nodes:
        # Direct lookup using pipeline node name
        if node_name in self.pipeline_nodes:
            resolved_nodes[node_name] = self.pipeline_nodes[node_name]
        else:
            # Fallback: parse node name and resolve
            base_name, job_type = self._parse_pipeline_node_name(node_name)
            step_info = self._find_step_by_base_and_job_type(base_name, job_type)
            if step_info:
                resolved_nodes[node_name] = step_info
    
    return resolved_nodes
```

## Summary

The job_type variant analysis reveals a sophisticated system where:

1. **One-to-Many Mapping**: Base components (script, contract, config, builder) map to multiple job_type-specific specifications
2. **PipelineDAG Integration**: Node names follow `{BaseStepName}_{job_type}` pattern for clear data flow separation
3. **Existing Implementation**: Comprehensive solution already exists with dynamic specification selection and semantic matching
4. **Unified Catalog Opportunity**: The step catalog system can leverage this pattern for efficient component indexing and discovery

The unified step catalog system should embrace and extend this job_type variant pattern rather than replace it, providing enhanced indexing and discovery capabilities while maintaining full compatibility with existing PipelineDAG structures and component sharing patterns.
