# Step Builder Registry Usage Guide

This guide provides examples of how to use the enhanced step builder registry system, particularly focused on adding new step builders with minimal effort thanks to the auto-discovery and registration mechanisms.

## Adding a New Step Builder

### Method 1: Auto-Registration via Decorator

The simplest way to add a new step builder is to use the `@register_builder` decorator:

```python
from src.pipeline_registry.builder_registry import register_builder
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.config_base import BasePipelineConfig

# Define your config class
class CustomProcessingConfig(BasePipelineConfig):
    """Custom processing step configuration."""
    
    def __init__(self, 
                 input_path: str,
                 output_path: str,
                 processing_mode: str = "standard",
                 **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.output_path = output_path
        self.processing_mode = processing_mode

# Use the decorator to auto-register your step builder
@register_builder("CustomProcessing")  # Explicitly provide the step type
class CustomProcessingStepBuilder(StepBuilderBase):
    """Builder for custom processing step."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config: CustomProcessingConfig = config
    
    # Implement required methods
    def create_step(self, **kwargs):
        # Implementation here
        pass
```

The `@register_builder` decorator automatically registers your step builder with the registry using the specified step type ("CustomProcessing" in this example). If no step type is provided, it will be derived from the class name by removing the "StepBuilder" suffix.

### Method 2: Auto-Discovery

If you prefer not to use the decorator, you can still benefit from auto-discovery:

```python
# src/pipeline_steps/builder_custom_validation_step.py
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.config_base import BasePipelineConfig

class CustomValidationConfig(BasePipelineConfig):
    """Custom validation step configuration."""
    
    def __init__(self, 
                 model_path: str,
                 validation_dataset: str,
                 metrics: List[str],
                 **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.validation_dataset = validation_dataset
        self.metrics = metrics

class CustomValidationStepBuilder(StepBuilderBase):
    """Builder for custom validation step."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config: CustomValidationConfig = config
    
    # Implement required methods
    def create_step(self, **kwargs):
        # Implementation here
        pass
```

With this approach, simply save the file in the `src/pipeline_steps/` directory with a name starting with `builder_` and the step builder registry will automatically discover and register it when initialized.

### Method 3: Manual Registration

For special cases where you need more control, you can manually register a step builder:

```python
from src.pipeline_registry.builder_registry import register_global_builder
from your_module import YourCustomStepBuilder

# Register your builder with the global registry
register_global_builder("CustomStep", YourCustomStepBuilder)
```

## Integrating with the Step Names Registry

For full integration with the pipeline system, update the step names registry as well:

```python
# src/pipeline_registry/step_names.py

# In the STEP_NAMES dictionary, add your new step
STEP_NAMES = {
    # ... existing entries ...
    
    "CustomProcessing": {
        "config_class": "CustomProcessingConfig",
        "builder_step_name": "CustomProcessingStep", 
        "spec_type": "CustomProcessing",
        "description": "Custom data processing step"
    },
}
```

## Verifying Registration

To verify that your step builder was properly registered:

```python
from src.pipeline_registry.builder_registry import get_global_registry

# Get the global registry
registry = get_global_registry()

# Check if your step type is supported
is_supported = registry.is_step_type_supported("CustomProcessing")
print(f"CustomProcessing supported: {is_supported}")

# List all supported step types
all_steps = registry.list_supported_step_types()
print(f"All supported steps: {all_steps}")

# Validate the registry for consistency
validation = registry.validate_registry()
for category, items in validation.items():
    if items:
        print(f"{category.capitalize()}:")
        for item in items:
            print(f"  - {item}")
```

## Best Practices

1. **Follow Naming Conventions**: 
   - Name your step builder class `YourNameStepBuilder` for consistency
   - Place your builder in a file named `builder_your_name_step.py` for auto-discovery

2. **Update Step Names Registry**:
   - Always update the `STEP_NAMES` registry with your new step to maintain a single source of truth
   - Make sure the canonical names are consistent across the system

3. **Use Validation**:
   - Run `registry.validate_registry()` to check for inconsistencies
   - Address any issues found in the 'invalid' or 'missing' categories

4. **Consider Job Types**:
   - If your step supports multiple job types (e.g., training, validation), support this in your configuration

5. **Documentation**:
   - Add clear docstrings to your config and builder classes
   - Document the purpose and usage of your step

## Troubleshooting

### Step Not Found in Registry

If your step isn't being registered automatically:

1. Check the file name - it should start with `builder_`
2. Verify the class inherits from `StepBuilderBase`
3. Ensure the module is importable (no import errors)
4. Look for logs with "Error discovering builders" message

### Inconsistent Step Names

If you're seeing warnings about inconsistent step names:

1. Check that the step name in `STEP_NAMES` registry matches what you use in `@register_builder`
2. Verify that the `builder_step_name` in `STEP_NAMES` matches your class name without the "Builder" suffix

### Multiple Entries for the Same Step

If multiple builders are being registered for the same step:

1. Check for duplicate step names in manual registrations
2. Verify you're not registering the same step through both auto-discovery and the decorator
3. Inspect the result of `registry.get_registry_stats()` to see counts of different builder types
