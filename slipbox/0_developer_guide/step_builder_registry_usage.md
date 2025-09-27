# Step Catalog System Usage Guide

**Version**: 3.0  
**Date**: September 27, 2025  
**Author**: Development Team

## Overview

This guide provides practical examples of using the **StepCatalog System** for step builder discovery and usage in pipeline development. The examples demonstrate modern patterns with automatic builder discovery and config-to-builder resolution.

## Basic Usage Patterns

### Getting StepCatalog Instance

```python
from cursus.step_catalog import StepCatalog

# Get catalog instance - automatically discovers all builders
catalog = StepCatalog()
```

### Checking Builder Availability

```python
# Check if a step builder exists
if catalog.is_step_type_supported("XGBoostTraining"):
    print("XGBoostTraining builder is available")

# List all available step types
step_types = catalog.list_supported_step_types()
print(f"Available step types: {step_types}")

# Get complete builder map
builder_map = catalog.get_builder_map()
print(f"Available builders: {list(builder_map.keys())}")
```

### Getting and Using Builders

```python
# Get builder class by step type
builder_class = catalog.get_builder_for_step_type("XGBoostTraining")

# Create builder instance with proper XGBoost configuration
from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingStepConfig

# Create config
config = XGBoostTrainingStepConfig(
    job_type="training",
    region="us-west-2",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    # ... other config parameters
)

# Create builder instance
builder = builder_class(config=config, role=role, sagemaker_session=session)

# Create step
step = builder.create_step()
```

## Config-to-Builder Resolution

### Direct Config Resolution

The StepCatalog can directly resolve builder classes from configuration instances:

```python
from cursus.step_catalog import StepCatalog
from cursus.steps.configs.config_batch_transform_step import BatchTransformStepConfig

# Initialize catalog
catalog = StepCatalog()

# Create configuration
config = BatchTransformStepConfig(
    job_type="training",
    region="us-west-2",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    model_name="my-model",
    instance_type="ml.m5.xlarge"
)

# Get builder class directly from config
builder_class = catalog.get_builder_for_config(config)
if builder_class:
    # Create builder instance
    builder = builder_class(config=config, role=role, sagemaker_session=session)
    print(f"Created builder: {builder.__class__.__name__}")
    
    # Create step
    step = builder.create_step()
else:
    print("No builder found for config")
```

### Multiple Config Types

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Different config types
configs = [
    XGBoostTrainingStepConfig(job_type="training", region="us-west-2"),
    BatchTransformStepConfig(job_type="inference", region="us-west-2"),
    PackageStepConfig(job_type="packaging", region="us-west-2")
]

# Resolve builders for each config
builders = []
for config in configs:
    builder_class = catalog.get_builder_for_config(config)
    if builder_class:
        builder = builder_class(config=config, role=role, sagemaker_session=session)
        builders.append(builder)
        print(f"Created {builder.__class__.__name__} for {config.__class__.__name__}")

# Create steps
steps = [builder.create_step() for builder in builders]
```

## Step Type Resolution

### Basic Step Type Resolution

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Get builder class by step type
builder_class = catalog.get_builder_for_step_type("XGBoostTraining")
if builder_class:
    print(f"Builder class: {builder_class.__name__}")
    
    # Create instance with config
    config = XGBoostTrainingStepConfig(
        job_type="training",
        region="us-west-2",
        pipeline_s3_loc="s3://my-bucket/pipeline"
    )
    builder = builder_class(config=config, role=role, sagemaker_session=session)
```

### Job Type Variant Resolution

The StepCatalog automatically handles job type variants:

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Handles variants like "CradleDataLoading_training", "CradleDataLoading_calibration"
training_builder = catalog.get_builder_for_step_type("CradleDataLoading_training")
calibration_builder = catalog.get_builder_for_step_type("CradleDataLoading_calibration")

# Both resolve to the same builder class but with different specifications
print(f"Training builder: {training_builder.__name__}")
print(f"Calibration builder: {calibration_builder.__name__}")
print(f"Same class: {training_builder == calibration_builder}")  # True
```

### Legacy Alias Support

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Legacy aliases are automatically resolved
legacy_builder = catalog.get_builder_for_step_type("PytorchTraining")  # Legacy name
canonical_builder = catalog.get_builder_for_step_type("PyTorchTraining")  # Canonical name

# Both return the same builder class
print(f"Legacy builder: {legacy_builder.__name__}")
print(f"Canonical builder: {canonical_builder.__name__}")
print(f"Same class: {legacy_builder == canonical_builder}")  # True
```

## Pipeline Construction Examples

### Complete Pipeline Assembly

```python
from cursus.step_catalog import StepCatalog

def create_complete_pipeline():
    """Create a complete pipeline using StepCatalog."""
    
    catalog = StepCatalog()
    
    # Define pipeline steps
    pipeline_steps = [
        ("CradleDataLoading", "training"),
        ("TabularPreprocessing", "training"),
        ("XGBoostTraining", None),
        ("BatchTransform", None),
        ("Package", None)
    ]
    
    # Create builders for each step
    builders = []
    for step_type, job_type in pipeline_steps:
        # Handle job type variants
        if job_type:
            step_name = f"{step_type}_{job_type}"
        else:
            step_name = step_type
            
        builder_class = catalog.get_builder_for_step_type(step_name)
        if builder_class:
            # Create appropriate config (simplified for example)
            config = create_config_for_step(step_type, job_type)
            builder = builder_class(config=config, role=role, sagemaker_session=session)
            builders.append((step_name, builder))
            print(f"Created builder for {step_name}")
        else:
            print(f"No builder found for {step_name}")
    
    return builders

def create_config_for_step(step_type, job_type):
    """Create appropriate config for step type."""
    base_config = {
        "region": "us-west-2",
        "pipeline_s3_loc": "s3://my-bucket/pipeline"
    }
    
    if job_type:
        base_config["job_type"] = job_type
    
    # Return appropriate config class instance
    if step_type == "CradleDataLoading":
        from cursus.steps.configs.config_cradle_data_load import CradleDataLoadConfig
        return CradleDataLoadConfig(**base_config)
    elif step_type == "TabularPreprocessing":
        from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
        return TabularPreprocessingConfig(**base_config)
    elif step_type == "XGBoostTraining":
        from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingStepConfig
        return XGBoostTrainingStepConfig(**base_config)
    # ... add other step types as needed
    
    return None

# Usage
pipeline_builders = create_complete_pipeline()
```

### Builder Availability Validation

```python
from cursus.step_catalog import StepCatalog

def validate_pipeline_builders(required_steps):
    """Validate that all required builders are available."""
    
    catalog = StepCatalog()
    
    # Validate builder availability for multiple step types
    availability = catalog.validate_builder_availability(required_steps)
    
    print("Builder Availability Report:")
    print("=" * 40)
    
    all_available = True
    for step_type, available in availability.items():
        status = "✅" if available else "❌"
        print(f"{status} {step_type}: {'Available' if available else 'Not Available'}")
        if not available:
            all_available = False
    
    if all_available:
        print("\n🎉 All required builders are available!")
    else:
        print("\n⚠️ Some builders are missing. Check step names and registry.")
        
        # Show available alternatives
        available_steps = catalog.list_supported_step_types()
        print(f"\nAvailable step types: {available_steps}")
    
    return all_available

# Usage
required_steps = ["XGBoostTraining", "BatchTransform", "Package", "Registration"]
validate_pipeline_builders(required_steps)
```

## Advanced Usage Patterns

### Dynamic Builder Discovery

```python
from cursus.step_catalog import StepCatalog

def discover_processing_steps():
    """Discover all available processing steps."""
    
    catalog = StepCatalog()
    
    # Get all step types
    all_steps = catalog.list_supported_step_types()
    
    # Filter for processing steps (example heuristic)
    processing_steps = []
    for step_type in all_steps:
        try:
            builder_class = catalog.get_builder_for_step_type(step_type)
            # Check if it creates ProcessingStep (simplified check)
            if hasattr(builder_class, 'create_step'):
                processing_steps.append(step_type)
        except Exception:
            continue
    
    print(f"Discovered {len(processing_steps)} processing steps:")
    for step in processing_steps:
        print(f"  - {step}")
    
    return processing_steps

# Usage
processing_steps = discover_processing_steps()
```

### Config Type Discovery

```python
from cursus.step_catalog import StepCatalog

def discover_config_types():
    """Discover config types for each step."""
    
    catalog = StepCatalog()
    
    step_types = catalog.list_supported_step_types()
    
    print("Step Type → Config Type Mapping:")
    print("=" * 50)
    
    for step_type in step_types:
        try:
            config_types = catalog.get_config_types_for_step_type(step_type)
            if config_types:
                print(f"{step_type}: {config_types}")
            else:
                print(f"{step_type}: No config types found")
        except Exception as e:
            print(f"{step_type}: Error - {e}")

# Usage
discover_config_types()
```

### Builder Factory Pattern

```python
from cursus.step_catalog import StepCatalog

class StepBuilderFactory:
    """Factory for creating step builders using StepCatalog."""
    
    def __init__(self):
        self.catalog = StepCatalog()
    
    def create_builder(self, step_type, config, **kwargs):
        """Create a step builder instance."""
        if not self.catalog.is_step_type_supported(step_type):
            available = self.catalog.list_supported_step_types()
            raise ValueError(f"Step type '{step_type}' not supported. Available: {available}")
        
        builder_class = self.catalog.get_builder_for_step_type(step_type)
        return builder_class(config=config, **kwargs)
    
    def create_builder_from_config(self, config, **kwargs):
        """Create builder directly from config."""
        builder_class = self.catalog.get_builder_for_config(config)
        if not builder_class:
            raise ValueError(f"No builder found for config type: {type(config).__name__}")
        
        return builder_class(config=config, **kwargs)
    
    def list_available_steps(self):
        """List all available step types."""
        return self.catalog.list_supported_step_types()
    
    def validate_step_availability(self, step_types):
        """Validate availability of multiple step types."""
        return self.catalog.validate_builder_availability(step_types)

# Usage
factory = StepBuilderFactory()

# Create builder by step type
config = XGBoostTrainingStepConfig(job_type="training", region="us-west-2")
builder = factory.create_builder("XGBoostTraining", config, role=role, sagemaker_session=session)

# Create builder from config
builder = factory.create_builder_from_config(config, role=role, sagemaker_session=session)

# List available steps
available_steps = factory.list_available_steps()
print(f"Available steps: {available_steps}")
```

## Integration with Pipeline Systems

### DAG Compiler Integration

```python
from cursus.step_catalog import StepCatalog
from cursus.core.dag.dag_compiler import PipelineDAGCompiler

def create_dag_compiler_with_catalog():
    """Create DAG compiler with StepCatalog integration."""
    
    # Create catalog
    catalog = StepCatalog()
    
    # Create DAG compiler with catalog
    compiler = PipelineDAGCompiler(step_catalog=catalog)
    
    # Get supported step types from catalog
    supported_types = compiler.get_supported_step_types()
    print(f"Compiler supports {len(supported_types)} step types")
    
    return compiler, catalog

# Usage
compiler, catalog = create_dag_compiler_with_catalog()

# Create pipeline template
template = compiler.create_template(
    dag=pipeline_dag,
    config_path="config.json",
    step_catalog=catalog
)
```

### Pipeline Assembler Integration

```python
from cursus.step_catalog import StepCatalog
from cursus.core.assembly.pipeline_assembler import PipelineAssembler

def create_pipeline_with_catalog(dag, config_map):
    """Create pipeline using StepCatalog for builder resolution."""
    
    # Create catalog
    catalog = StepCatalog()
    
    # Create assembler with catalog
    assembler = PipelineAssembler(
        dag=dag,
        config_map=config_map,
        step_catalog=catalog  # Direct config-to-builder resolution
    )
    
    # Assembler automatically uses catalog for builder resolution
    pipeline = assembler.assemble()
    
    return pipeline

# Usage
pipeline = create_pipeline_with_catalog(my_dag, my_config_map)
```

## Performance Optimization

### Catalog Performance Monitoring

```python
from cursus.step_catalog import StepCatalog
import time

def monitor_catalog_performance():
    """Monitor StepCatalog performance."""
    
    # Measure initialization time
    start_time = time.time()
    catalog = StepCatalog()
    init_time = time.time() - start_time
    print(f"Catalog initialized in {init_time:.3f}s")
    
    # Measure discovery time
    start_time = time.time()
    step_types = catalog.list_supported_step_types()
    discovery_time = time.time() - start_time
    print(f"Discovered {len(step_types)} step types in {discovery_time:.3f}s")
    
    # Measure resolution time
    test_steps = ["XGBoostTraining", "BatchTransform", "Package"]
    start_time = time.time()
    for step_type in test_steps:
        builder_class = catalog.get_builder_for_step_type(step_type)
    resolution_time = time.time() - start_time
    print(f"Resolved {len(test_steps)} builders in {resolution_time:.3f}s")
    
    return {
        "init_time": init_time,
        "discovery_time": discovery_time,
        "resolution_time": resolution_time,
        "step_count": len(step_types)
    }

# Usage
performance_stats = monitor_catalog_performance()
```

### Caching Optimization

```python
from cursus.step_catalog import StepCatalog

def optimize_catalog_usage():
    """Demonstrate optimal catalog usage patterns."""
    
    # Create single catalog instance (recommended)
    catalog = StepCatalog()
    
    # Cache frequently used builders
    common_builders = {}
    common_steps = ["XGBoostTraining", "BatchTransform", "Package"]
    
    for step_type in common_steps:
        builder_class = catalog.get_builder_for_step_type(step_type)
        common_builders[step_type] = builder_class
        print(f"Cached builder for {step_type}")
    
    # Use cached builders
    def create_cached_builder(step_type, config, **kwargs):
        if step_type in common_builders:
            builder_class = common_builders[step_type]
            return builder_class(config=config, **kwargs)
        else:
            # Fall back to catalog lookup
            builder_class = catalog.get_builder_for_step_type(step_type)
            return builder_class(config=config, **kwargs)
    
    return create_cached_builder

# Usage
create_builder = optimize_catalog_usage()
builder = create_builder("XGBoostTraining", config, role=role, sagemaker_session=session)
```

## Error Handling and Debugging

### Graceful Error Handling

```python
from cursus.step_catalog import StepCatalog

def get_builder_safely(step_type):
    """Get builder with comprehensive error handling."""
    
    catalog = StepCatalog()
    
    try:
        # Check if step type is supported
        if not catalog.is_step_type_supported(step_type):
            available_steps = catalog.list_supported_step_types()
            raise ValueError(
                f"Step type '{step_type}' not supported. "
                f"Available step types: {available_steps}"
            )
        
        # Get builder class
        builder_class = catalog.get_builder_for_step_type(step_type)
        return builder_class
        
    except Exception as e:
        print(f"Error getting builder for '{step_type}': {e}")
        
        # Provide helpful debugging information
        print("\nDebugging Information:")
        print(f"- Supported step types: {catalog.list_supported_step_types()}")
        
        # Check for similar step names
        available_steps = catalog.list_supported_step_types()
        similar_steps = [s for s in available_steps if step_type.lower() in s.lower()]
        if similar_steps:
            print(f"- Similar step types: {similar_steps}")
        
        return None

# Usage
builder_class = get_builder_safely("XGBoostTraining")
if builder_class:
    builder = builder_class(config=config, role=role, sagemaker_session=session)
```

### Debugging Catalog State

```python
from cursus.step_catalog import StepCatalog

def debug_catalog_state():
    """Debug catalog state and discovery."""
    
    catalog = StepCatalog()
    
    print("StepCatalog Debug Information:")
    print("=" * 50)
    
    # List all discovered step types
    step_types = catalog.list_supported_step_types()
    print(f"Total discovered step types: {len(step_types)}")
    
    # Show step types by category (heuristic)
    processing_steps = [s for s in step_types if any(keyword in s.lower() 
                       for keyword in ['processing', 'preprocess', 'transform', 'package'])]
    training_steps = [s for s in step_types if 'training' in s.lower()]
    model_steps = [s for s in step_types if 'model' in s.lower()]
    
    print(f"\nProcessing steps ({len(processing_steps)}): {processing_steps}")
    print(f"Training steps ({len(training_steps)}): {training_steps}")
    print(f"Model steps ({len(model_steps)}): {model_steps}")
    
    # Test builder resolution for each step type
    print(f"\nBuilder Resolution Test:")
    failed_resolutions = []
    for step_type in step_types:
        try:
            builder_class = catalog.get_builder_for_step_type(step_type)
            print(f"✅ {step_type} → {builder_class.__name__}")
        except Exception as e:
            print(f"❌ {step_type} → Error: {e}")
            failed_resolutions.append(step_type)
    
    if failed_resolutions:
        print(f"\nFailed resolutions: {failed_resolutions}")
    else:
        print(f"\n🎉 All step types resolved successfully!")

# Usage
debug_catalog_state()
```

## Best Practices

### 1. Use Single Catalog Instance

```python
# Good: Single instance
catalog = StepCatalog()
builder1 = catalog.get_builder_for_step_type("Step1")
builder2 = catalog.get_builder_for_step_type("Step2")

# Avoid: Multiple instances
catalog1 = StepCatalog()
builder1 = catalog1.get_builder_for_step_type("Step1")
catalog2 = StepCatalog()  # Unnecessary
builder2 = catalog2.get_builder_for_step_type("Step2")
```

### 2. Handle Missing Builders Gracefully

```python
# Good: Check availability first
if catalog.is_step_type_supported("MyStep"):
    builder_class = catalog.get_builder_for_step_type("MyStep")
else:
    print("MyStep not supported")

# Better: Use try-catch with helpful error messages
try:
    builder_class = catalog.get_builder_for_step_type("MyStep")
except Exception as e:
    available = catalog.list_supported_step_types()
    print(f"Error: {e}. Available steps: {available}")
```

### 3. Leverage Config-to-Builder Resolution

```python
# Good: Direct config resolution
builder_class = catalog.get_builder_for_config(config)
if builder_class:
    builder = builder_class(config=config, **kwargs)

# Avoid: Manual step type extraction
step_type = extract_step_type_from_config(config)  # Manual work
builder_class = catalog.get_builder_for_step_type(step_type)
```

### 4. Validate Builder Availability Early

```python
# Good: Validate all required builders upfront
required_steps = ["XGBoostTraining", "BatchTransform", "Package"]
availability = catalog.validate_builder_availability(required_steps)

if not all(availability.values()):
    missing = [step for step, available in availability.items() if not available]
    raise ValueError(f"Missing required builders: {missing}")

# Proceed with pipeline creation
```

### 5. Use Performance Monitoring

```python
# Good: Monitor performance in production
import time

start_time = time.time()
catalog = StepCatalog()
init_time = time.time() - start_time

if init_time > 1.0:  # Threshold
    print(f"Warning: Catalog initialization took {init_time:.3f}s")

# Log performance metrics
logger.info(f"StepCatalog initialized in {init_time:.3f}s")
```

## Migration from Legacy Systems

### Old Pattern (Removed)

```python
# Old way - don't use (this has been removed)
from cursus.registry.builder_registry import StepBuilderRegistry

registry = StepBuilderRegistry()
builder = registry.get_builder_for_config(config)
```

### New Pattern (Current)

```python
# New way - use this
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()
builder = catalog.get_builder_for_config(config)
```

## Related Resources

- [Step Catalog Integration Guide](step_catalog_integration_guide.md) - Comprehensive system documentation
- [Adding a New Pipeline Step](adding_new_pipeline_step.md) - Complete guide to adding new steps
- [Step Builder Implementation](step_builder.md) - Detailed step builder patterns
- [Creation Process](creation_process.md) - Step-by-step creation process

This guide demonstrates practical usage patterns for the modern StepCatalog system and should be used for all pipeline development.
