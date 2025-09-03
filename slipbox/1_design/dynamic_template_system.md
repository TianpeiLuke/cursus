---
tags:
  - design
  - implementation
  - pipeline
  - architecture
  - dynamic_template
  - config_resolution
keywords:
  - dynamic template
  - pipeline generation
  - specification-driven
  - DAG
  - config resolution
  - validation engine
  - builder registry
  - intelligent matching
  - automatic resolution
topics:
  - pipeline template
  - configuration management
  - pipeline generation
  - automation
  - dynamic resolution
language: python
date of note: 2025-08-12
---

# Dynamic Template System

## Overview

The Dynamic Template System is a comprehensive solution for creating flexible, configuration-driven pipelines without requiring custom template classes. Based on the implementation in `src/cursus/core/compiler`, this system automatically resolves DAG nodes to configurations and step builders, enabling universal pipeline creation from any DAG structure.

## Core Architecture

The system consists of several interconnected components:

```
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ Dynamic         │    │ Step Config         │    │ Validation       │
│ Pipeline        │◄───┤ Resolver            ├───►│ Engine           │
│ Template        │    └─────────────────────┘    └──────────────────┘
└─────────────────┘              │
         │                       ▼
         ▼                ┌─────────────────────┐
┌─────────────────┐       │ Step Builder        │
│ Config Class    │       │ Registry            │
│ Detection       │       └─────────────────────┘
└─────────────────┘
```

### Key Components

1. **DynamicPipelineTemplate** (`dynamic_template.py`) - Main orchestrator extending PipelineTemplateBase
2. **StepConfigResolver** (`config_resolver.py`) - Intelligent DAG node to configuration mapping
3. **ValidationEngine** (`validation.py`) - Comprehensive pipeline validation
4. **Config Class Detection** - Automatic detection of required configuration classes
5. **Step Builder Registry Integration** - Mapping configurations to step builders

## Dynamic Pipeline Template

The `DynamicPipelineTemplate` class provides a universal implementation of `PipelineTemplateBase`:

### Core Implementation

```python
class DynamicPipelineTemplate(PipelineTemplateBase):
    """
    Dynamic pipeline template that works with any PipelineDAG.
    
    This template automatically implements the abstract methods of
    PipelineTemplateBase by using intelligent resolution mechanisms
    to map DAG nodes to configurations and step builders.
    """
    
    def __init__(
        self,
        dag: PipelineDAG,
        config_path: str,
        config_resolver: Optional[StepConfigResolver] = None,
        builder_registry: Optional[StepBuilderRegistry] = None,
        skip_validation: bool = False,
        **kwargs
    ):
        """Initialize dynamic template with intelligent resolution."""
        self._dag = dag
        self._config_resolver = config_resolver or StepConfigResolver()
        self._builder_registry = builder_registry or StepBuilderRegistry()
        self._validation_engine = ValidationEngine()
        
        # Auto-detect required config classes
        cls = self.__class__
        if not cls.CONFIG_CLASSES:
            cls.CONFIG_CLASSES = self._detect_config_classes()
        
        super().__init__(config_path=config_path, **kwargs)
```

### Automatic Method Implementation

The template automatically implements all required abstract methods:

1. **`_create_pipeline_dag()`** - Returns the provided DAG
2. **`_create_config_map()`** - Maps DAG nodes to configurations using intelligent resolution
3. **`_create_step_builder_map()`** - Maps configuration types to step builder classes
4. **`_validate_configuration()`** - Validates the complete pipeline configuration

### Config Class Detection

```python
def _detect_config_classes(self) -> Dict[str, Type[BasePipelineConfig]]:
    """
    Automatically detect required config classes from configuration file.
    
    Analyzes the configuration file to determine which configuration classes
    are needed based on:
    1. Config type metadata in the configuration file
    2. Model type information in configuration entries
    3. Essential base classes needed for all pipelines
    """
    from ...steps.configs.utils import detect_config_classes_from_json
    
    detected_classes = detect_config_classes_from_json(self.config_path)
    self.logger.debug(f"Detected {len(detected_classes)} required config classes")
    
    return detected_classes
```

## Step Config Resolver

The `StepConfigResolver` implements intelligent matching between DAG nodes and configurations using multiple strategies:

### Resolution Strategies (Priority Order)

1. **Direct Name Matching** (Confidence: 1.0)
   - Exact match between node name and config key
   - Case-insensitive matching as fallback
   - Metadata mapping from `config_types`

2. **Enhanced Job Type Matching** (Confidence: 0.8-0.9)
   - Parses node names for job type information
   - Matches against configuration `job_type` attributes
   - Considers config type similarity

3. **Semantic Similarity Matching** (Confidence: 0.5-0.8)
   - Uses predefined semantic mappings
   - Calculates string similarity scores
   - Handles synonyms and related terms

4. **Pattern-Based Matching** (Confidence: 0.6-0.9)
   - Uses regex patterns to identify step types
   - Maps patterns to configuration types
   - Includes job type boost calculations

### Enhanced Resolution Algorithm

```python
def resolve_config_map(
    self, 
    dag_nodes: List[str], 
    available_configs: Dict[str, BasePipelineConfig],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, BasePipelineConfig]:
    """
    Resolve DAG nodes to configurations with enhanced metadata handling.
    
    Resolution strategies (in order of preference):
    1. Direct name matching with exact match
    2. Metadata mapping from config_types
    3. Job type + config type matching with pattern recognition
    4. Semantic similarity matching
    5. Pattern-based matching
    """
    # Extract metadata.config_types mapping if available
    self._metadata_mapping = {}
    if metadata and "config_types" in metadata:
        self._metadata_mapping = metadata["config_types"]
        
    # Resolve each node using tiered approach
    resolved_configs = {}
    for node_name in dag_nodes:
        config, confidence, method = self._resolve_single_node(node_name, available_configs)
        resolved_configs[node_name] = config
        
    return resolved_configs
```

### Node Name Parsing

The resolver includes sophisticated node name parsing:

```python
def _parse_node_name(self, node_name: str) -> Dict[str, str]:
    """
    Parse node name to extract config type and job type information.
    
    Supports patterns like:
    - ConfigType_JobType (e.g., CradleDataLoading_training)
    - JobType_Task (e.g., training_data_load)
    """
    patterns = [
        (r'^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$', 'config_first'),
        (r'^([a-z]+)_([A-Za-z_]+)$', 'job_first'),
    ]
    
    for pattern, pattern_type in patterns:
        match = re.match(pattern, node_name)
        if match:
            # Extract and return parsed information
            # Implementation details...
```

### Pattern Matching System

```python
STEP_TYPE_PATTERNS = {
    r'.*data_load.*': ['CradleDataLoading'],
    r'.*preprocess.*': ['TabularPreprocessing'],
    r'.*train.*': ['XGBoostTraining', 'PyTorchTraining', 'DummyTraining'],
    r'.*eval.*': ['XGBoostModelEval'],
    r'.*model.*': ['XGBoostModel', 'PyTorchModel'],
    r'.*calibrat.*': ['ModelCalibration'],
    r'.*packag.*': ['MIMSPackaging'],
    r'.*payload.*': ['MIMSPayload'],
    r'.*regist.*': ['ModelRegistration'],
    r'.*transform.*': ['BatchTransform'],
    # Additional patterns...
}
```

### Semantic Matching

```python
def _calculate_semantic_similarity(self, node_name: str, config: BasePipelineConfig) -> float:
    """Calculate semantic similarity between node name and config."""
    semantic_mappings = {
        'data': ['cradle', 'load', 'loading'],
        'preprocess': ['preprocessing', 'process', 'clean'],
        'train': ['training', 'fit', 'learn', 'model_fit'],
        'eval': ['evaluation', 'evaluate', 'test', 'assess', 'model_test'],
        'model': ['model', 'create', 'build'],
        'calibrat': ['calibration', 'calibrate', 'adjust'],
        'packag': ['packaging', 'package', 'bundle'],
        'regist': ['registration', 'register', 'deploy'],
    }
    
    # Calculate similarity using semantic mappings and string matching
    # Implementation details...
```

## Validation Engine

The `ValidationEngine` provides comprehensive validation of pipeline configurations:

### Validation Checks

1. **DAG Node Coverage** - All nodes have corresponding configurations
2. **Builder Availability** - All configurations have matching step builders
3. **Configuration Validity** - Configuration-specific validation rules
4. **Dependency Resolution** - Step dependencies can be resolved
5. **Metadata Consistency** - Metadata mappings are valid

### Validation Implementation

```python
def validate_dag_compatibility(
    self,
    dag_nodes: List[str],
    available_configs: Dict[str, BasePipelineConfig],
    config_map: Dict[str, BasePipelineConfig],
    builder_registry: Dict[str, Type[StepBuilderBase]]
) -> ValidationResult:
    """
    Perform comprehensive validation of DAG compatibility.
    
    Returns:
        ValidationResult with detailed validation information
    """
    # Implementation includes all validation checks
    # Returns structured validation results
```

## Advanced Features

### Resolution Preview

```python
def get_resolution_preview(self) -> Dict[str, Any]:
    """
    Get a preview of how DAG nodes will be resolved.
    
    Returns:
        Dictionary with resolution preview information including:
        - Node count and resolution details
        - Confidence scores and methods
        - Alternative candidates
        - Diagnostic recommendations
    """
    preview_data = self._config_resolver.preview_resolution(
        dag_nodes=list(self._dag.nodes),
        available_configs=self.configs,
        metadata=self._loaded_metadata
    )
    
    # Convert to display format with confidence scores
    # Implementation details...
```

### Pipeline Metadata Management

The template includes sophisticated metadata handling:

```python
def _store_pipeline_metadata(self, assembler: "PipelineAssembler") -> None:
    """
    Store pipeline metadata from template.
    
    Dynamically discovers and stores metadata from various step types:
    - Cradle data loading requests
    - Registration step configurations
    - Model names and execution parameters
    """
    # Store Cradle data loading requests
    if hasattr(assembler, 'cradle_loading_requests'):
        self.pipeline_metadata['cradle_loading_requests'] = assembler.cradle_loading_requests
    
    # Find and store registration steps
    registration_steps = self._find_registration_steps(assembler)
    if registration_steps:
        self._process_registration_metadata(registration_steps)
```

### Execution Document Filling

```python
def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill execution document with pipeline metadata.
    
    Populates the execution document with:
    1. Cradle data loading requests (if present)
    2. Registration configurations (if present)
    3. Model names and execution parameters
    """
    pipeline_configs = execution_document.get("PIPELINE_STEP_CONFIGS", {})
    
    # Handle Cradle configurations
    self._fill_cradle_configurations(pipeline_configs)
    
    # Handle Registration configurations
    self._fill_registration_configurations(pipeline_configs)
    
    return execution_document
```

## Error Handling and Diagnostics

### Exception Types

```python
class ConfigurationError(Exception):
    """Raised when configuration resolution fails."""
    def __init__(self, message: str, missing_configs: List[str] = None, 
                 available_configs: List[str] = None):
        super().__init__(message)
        self.missing_configs = missing_configs or []
        self.available_configs = available_configs or []

class AmbiguityError(Exception):
    """Raised when multiple configurations match with similar confidence."""
    def __init__(self, message: str, node_name: str = None, candidates: List = None):
        super().__init__(message)
        self.node_name = node_name
        self.candidates = candidates or []

class ResolutionError(Exception):
    """Raised when a single node cannot be resolved."""
    pass
```

### Comprehensive Logging

The system provides extensive logging for debugging:

```python
# Resolution logging
self.logger.info(f"Resolved node '{node_name}' to {type(config).__name__} "
                f"(job_type='{getattr(config, 'job_type', 'N/A')}') "
                f"with confidence {confidence:.2f} using {method} matching")

# Validation logging
self.logger.info(f"Configuration validation passed successfully")
if validation_result.warnings:
    for warning in validation_result.warnings:
        self.logger.warning(warning)
```

## Integration Examples

### Basic Usage

```python
from src.cursus.core.compiler.dynamic_template import DynamicPipelineTemplate
from src.cursus.api.dag.base_dag import PipelineDAG

# Create a DAG defining pipeline structure
dag = PipelineDAG()
dag.add_node("data_loading", node_type="CradleDataLoading")
dag.add_node("preprocessing", node_type="TabularPreprocessing")
dag.add_node("training", node_type="XGBoostTraining")
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")

# Create dynamic template
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="path/to/config.json"
)

# Build the pipeline
pipeline = template.build_pipeline()
```

### Advanced Usage with Custom Resolvers

```python
# Create custom resolver with specific confidence threshold
custom_resolver = StepConfigResolver(confidence_threshold=0.8)

# Create template with custom components
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="path/to/config.json",
    config_resolver=custom_resolver,
    skip_validation=False  # Enable full validation
)

# Get resolution preview before building
preview = template.get_resolution_preview()
print(f"Resolving {preview['nodes']} nodes...")

for node, info in preview['resolutions'].items():
    print(f"{node} -> {info['config_type']} (confidence: {info['confidence']:.2f})")

# Build pipeline if preview looks good
pipeline = template.build_pipeline()
```

### Integration with DAG Compiler

```python
from src.cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# Convert JSON DAG to PipelineDAG object
dag_json = {
    "nodes": [
        {"name": "data_loading", "type": "CradleDataLoading"},
        {"name": "preprocessing", "type": "TabularPreprocessing"},
        {"name": "training", "type": "XGBoostTraining"}
    ],
    "edges": [
        {"from": "data_loading", "to": "preprocessing"},
        {"from": "preprocessing", "to": "training"}
    ]
}

dag = PipelineDAGCompiler.convert_from_json(dag_json)

# Create dynamic template with converted DAG
template = DynamicPipelineTemplate(dag=dag, config_path="config.json")
pipeline = template.build_pipeline()
```

## Performance Considerations

1. **Config Class Detection** - Only loads required configuration classes
2. **Resolution Caching** - Caches parsed node names and resolution results
3. **Lazy Evaluation** - Defers expensive operations until needed
4. **Efficient Validation** - Optimized validation checks with early termination

## Benefits

1. **Universal Compatibility** - Works with any DAG structure without modification
2. **Intelligent Resolution** - Multiple strategies ensure successful matching
3. **Reduced Code Duplication** - Single template implementation for all pipelines
4. **Rapid Prototyping** - Quick pipeline creation and iteration
5. **Comprehensive Validation** - Thorough validation before execution
6. **Enhanced Debugging** - Detailed logging and error reporting
7. **Metadata Integration** - Automatic handling of pipeline metadata

## Future Enhancements

1. **Machine Learning Integration** - Learn from successful resolutions to improve matching
2. **Custom Resolution Strategies** - Plugin system for domain-specific resolution logic
3. **Visual Resolution Tools** - Graphical tools for debugging resolution issues
4. **Performance Optimization** - Further caching and optimization improvements
5. **Template Customization** - More options for customizing template behavior

## References

- Implementation: `src/cursus/core/compiler/`
- Base Classes: `src/cursus/core/base.py`
- Step Builder Registry: `src/cursus/registry/`
- Configuration Classes: `src/cursus/steps/configs/`
- Pipeline Template Base: `src/cursus/core/assembler/pipeline_template_base.py`

## Conclusion

The Dynamic Template System represents a significant advancement in pipeline architecture, providing a universal, intelligent solution for creating pipelines from any DAG structure. Through sophisticated resolution algorithms, comprehensive validation, and extensive metadata handling, it eliminates the need for custom template classes while maintaining flexibility and reliability. The system's implementation in `cursus/core/compiler` demonstrates a mature, production-ready approach to dynamic pipeline generation.
