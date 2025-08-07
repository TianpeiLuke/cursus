# Step Creation Process

This document outlines the step-by-step process for adding a new step to the pipeline. Follow these steps in order to ensure proper integration with the existing architecture.

## Overview of the Process

Adding a new step to the pipeline involves creating several components that work together:

1. Register the new step name in the step registry
2. Create the step configuration class
3. Develop the script contract
4. Create the step specification
5. Implement the step builder
6. Update required registry files
7. Create unit tests
8. Integrate with pipeline templates

## Detailed Steps

### 1. Register the New Step Name

First, register your step in the central step registry:

**File to Update**: `src/cursus/steps/registry/step_names.py`

```python
STEP_NAMES = {
    # ... existing steps ...
    
    "YourNewStep": {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder",
        "spec_type": "YourNewStep",
        "sagemaker_step_type": "Processing",  # Based on create_step() return type
        "description": "Description of your new step"
    },
}
```

**Important**: The `sagemaker_step_type` field must match the actual SageMaker step type returned by your step builder's `create_step()` method:

- **"Processing"** - for steps that return `ProcessingStep`
- **"Training"** - for steps that return `TrainingStep`  
- **"Transform"** - for steps that return `TransformStep`
- **"CreateModel"** - for steps that return `CreateModelStep`
- **"RegisterModel"** - for steps that return custom registration steps (like `MimsModelRegistrationProcessingStep`)
- **"Lambda"** - for steps that return `LambdaStep`
- **"Base"** - for base/utility steps

This registration connects your step's components together and makes them discoverable by the pipeline system.

### 2. Create the Step Configuration

Create a configuration class using the three-tier field classification design:

**Create New File**: `src/pipeline_steps/config_your_new_step.py`

```python
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .config_base import BasePipelineConfig

class YourNewStepConfig(BasePipelineConfig):
    """
    Configuration for YourNewStep using three-tier field classification.
    
    Tier 1: Essential fields (required user inputs)
    Tier 2: System fields (with defaults, can be overridden)
    Tier 3: Derived fields (private with property access)
    """
    
    # Tier 1: Essential user inputs (required, no defaults)
    region: str = Field(..., description="AWS region code (NA, EU, FE)")
    pipeline_s3_loc: str = Field(..., description="S3 location for pipeline artifacts")
    param1: str = Field(..., description="Essential step parameter")
    
    # Tier 2: System inputs with defaults (can be overridden)
    instance_type: str = Field(default="ml.m5.xlarge", description="SageMaker instance type")
    instance_count: int = Field(default=1, description="Number of instances")
    volume_size_gb: int = Field(default=30, description="EBS volume size in GB")
    max_runtime_seconds: int = Field(default=3600, description="Maximum runtime in seconds")
    param2: int = Field(default=0, description="Optional step parameter")
    
    # Tier 3: Derived fields (private with property access)
    _output_path: Optional[str] = Field(default=None, exclude=True)
    _job_name_prefix: Optional[str] = Field(default=None, exclude=True)
    
    # Internal cache for computed values
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Public properties for derived fields
    @property
    def output_path(self) -> str:
        """Get derived output path."""
        if self._output_path is None:
            self._output_path = f"{self.pipeline_s3_loc}/your_new_step/{self.region}"
        return self._output_path
    
    @property
    def job_name_prefix(self) -> str:
        """Get derived job name prefix."""
        if self._job_name_prefix is None:
            self._job_name_prefix = f"your-new-step-{self.region}"
        return self._job_name_prefix
    
    # Include derived fields in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        data["output_path"] = self.output_path
        data["job_name_prefix"] = self.job_name_prefix
        return data
    
    # Get the script contract
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..pipeline_script_contracts.your_new_step_contract import YOUR_NEW_STEP_CONTRACT
        return YOUR_NEW_STEP_CONTRACT
```

The configuration class follows the three-tier field classification:

1. **Tier 1 (Essential Fields)**: Required inputs from users (no defaults)
2. **Tier 2 (System Fields)**: Default values that can be overridden by users
3. **Tier 3 (Derived Fields)**: Private fields with public property access, computed from other fields

For more details on the three-tier design, see [Three-Tier Config Design](three_tier_config_design.md).

### 3. Create the Script Contract

Define the contract between your script and the SageMaker environment:

**Create New File**: `src/pipeline_script_contracts/your_new_step_contract.py`

```python
from pydantic import BaseModel
from typing import Dict, List, Optional

from .base_script_contract import ScriptContract

YOUR_NEW_STEP_CONTRACT = ScriptContract(
    entry_point="your_script.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "input_metadata": "/opt/ml/processing/input/metadata"
        # Add all required input paths with logical names matching step specification
    },
    expected_output_paths={
        "output_data": "/opt/ml/processing/output/data",
        "output_metadata": "/opt/ml/processing/output/metadata"
        # Add all expected output paths with logical names matching step specification
    },
    required_env_vars=[
        "REQUIRED_PARAM_1",
        "REQUIRED_PARAM_2"
        # List all required environment variables
    ],
    optional_env_vars={
        "OPTIONAL_PARAM_1": "default_value",
        # Optional environment variables with default values
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.20.0",
        # Add all framework requirements with version constraints
    },
    description="Your new step's script contract description"
)
```

The script contract defines:
- The entry point script name
- All input and output paths used by the script
- Required and optional environment variables
- Framework dependencies with version constraints
- A description of the script's purpose

### 4. Create the Step Specification

Define how your step connects with others in the pipeline:

**Create New File**: `src/pipeline_step_specs/your_new_step_spec.py`

```python
from typing import Dict, List, Optional

from ..pipeline_deps.base_specifications import StepSpecification, NodeType, DependencySpec, OutputSpec, DependencyType
from ..pipeline_script_contracts.your_new_step_contract import YOUR_NEW_STEP_CONTRACT
from ..pipeline_registry.step_names import get_spec_step_type

def _get_your_new_step_contract():
    """Get the script contract for this step."""
    from ..pipeline_script_contracts.your_new_step_contract import YOUR_NEW_STEP_CONTRACT
    return YOUR_NEW_STEP_CONTRACT

YOUR_NEW_STEP_SPEC = StepSpecification(
    step_type=get_spec_step_type("YourNewStep"),
    node_type=NodeType.INTERNAL,  # Or SOURCE, SINK, or other appropriate type
    script_contract=_get_your_new_step_contract(),
    dependencies={
        "input_data": DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["PreviousStep", "AnotherStep"],
            semantic_keywords=["data", "input", "features"],
            data_type="S3Uri",
            description="Input data for processing"
        ),
        "input_metadata": DependencySpec(
            logical_name="input_metadata",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,  # Optional dependency
            compatible_sources=["PreviousStep", "AnotherStep"],
            semantic_keywords=["metadata", "schema", "information"],
            data_type="S3Uri",
            description="Input metadata for processing"
        )
    },
    outputs={
        "output_data": OutputSpec(
            logical_name="output_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['output_data'].S3Output.S3Uri",
            aliases=["processed_data", "transformed_data"],  # Optional aliases for backward compatibility
            data_type="S3Uri",
            description="Processed output data"
        ),
        "output_metadata": OutputSpec(
            logical_name="output_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['output_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Output metadata"
        )
    }
)

# If you need job type variants:
def get_your_new_step_spec(job_type: str = None):
    """Get the appropriate specification based on job type."""
    if job_type and job_type.lower() == "calibration":
        return YOUR_NEW_STEP_CALIBRATION_SPEC
    elif job_type and job_type.lower() == "validation":
        return YOUR_NEW_STEP_VALIDATION_SPEC
    else:
        return YOUR_NEW_STEP_SPEC  # Default to training
```

The step specification defines:
- The step type and node type
- The connection to the script contract
- Dependencies on other steps' outputs
- Outputs for use by downstream steps
- Job type variants if needed

### 5. Create the Step Builder

Implement the builder that creates the SageMaker step, using the `@register_builder` decorator:

**Create New File**: `src/pipeline_steps/builder_your_new_step.py`

```python
from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ..pipeline_registry.builder_registry import register_builder
from ..pipeline_deps.base_specifications import StepSpecification
from ..pipeline_script_contracts.base_script_contract import ScriptContract
from .builder_step_base import StepBuilderBase
from .config_your_new_step import YourNewStepConfig
from ..pipeline_step_specs.your_new_step_spec import YOUR_NEW_STEP_SPEC

@register_builder("YourNewStep")
class YourNewStepBuilder(StepBuilderBase):
    """Builder for YourNewStep processing step."""
    
    def __init__(
        self, 
        config, 
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        # Get job type if available
        job_type = getattr(config, 'job_type', None)
        
        # Get the appropriate specification based on job type
        if job_type and hasattr(self, '_get_spec_for_job_type'):
            spec = self._get_spec_for_job_type(job_type)
        else:
            spec = YOUR_NEW_STEP_SPEC
        
        # Get the script contract from the specification
        contract = spec.script_contract if spec else None
        
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: YourNewStepConfig = config
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract."""
        # Use the specification-driven approach to generate inputs
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract."""
        # Use the specification-driven approach to generate outputs
        return self._get_spec_driven_processor_outputs(outputs)
    
    def _get_processor_env_vars(self) -> Dict[str, str]:
        """Get environment variables for the processor."""
        env_vars = {
            "REQUIRED_PARAM_1": self.config.param1,
            "REQUIRED_PARAM_2": str(self.config.param2)
            # Add any other environment variables needed by your script
        }
        return env_vars
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the processing step.
        
        Args:
            **kwargs: Additional keyword arguments for step creation.
                     Should include 'dependencies' list if step has dependencies.
        """
        # Extract inputs from dependencies using the resolver
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Get processor inputs and outputs
        inputs = self._get_inputs(extracted_inputs)
        outputs = self._get_outputs({})
        
        # Create processor
        processor = self._get_processor()
        
        # Set environment variables
        env_vars = self._get_processor_env_vars()
        
        # Create and return the step
        step_name = kwargs.get('step_name', 'YourNewStep')
        step = processor.run(
            inputs=inputs,
            outputs=outputs,
            container_arguments=[],
            container_entrypoint=["python", self.config.get_script_path()],
            job_name=self._generate_job_name(step_name),
            wait=False,
            environment=env_vars
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

The step builder:
- Gets the appropriate specification based on job type
- Creates a SageMaker processor
- Sets up inputs and outputs based on the specification
- Configures environment variables from the config
- Creates and returns the SageMaker step

### 6. Update Step Names Registry

Add your new step to the central step names registry:

**File to Update**: `src/cursus/steps/registry/step_names.py`

```python
# Add to existing STEP_NAMES dictionary
STEP_NAMES = {
    # ... existing steps ...
    
    "YourNewStep": {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder",
        "spec_type": "YourNewStep",
        "sagemaker_step_type": "Processing",  # Based on create_step() return type
        "description": "Description of your new step"
    },
}
```

**Important**: Ensure the `sagemaker_step_type` field matches the actual SageMaker step type returned by your step builder's `create_step()` method. This field is used by the Universal Builder Test framework for step-type-specific validation and testing.

Note: With the auto-discovery system, you don't need to manually update `__init__.py` files anymore. The `@register_builder` decorator automatically handles registration, and step builder files are discovered based on their naming pattern.

### 7. Create Unit Tests

Implement tests to verify your components work correctly:

**Create New File**: `test/pipeline_steps/test_builder_your_new_step.py`

```python
import unittest
from unittest.mock import MagicMock, patch

from src.pipeline_steps.builder_your_new_step import YourNewStepBuilder
from src.pipeline_steps.config_your_new_step import YourNewStepConfig
from src.pipeline_step_specs.your_new_step_spec import YOUR_NEW_STEP_SPEC
from src.pipeline_deps.base_specifications import NodeType, DependencyType

class TestYourNewStepBuilder(unittest.TestCase):
    def setUp(self):
        self.config = YourNewStepConfig(
            region="us-west-2",
            pipeline_s3_loc="s3://bucket/prefix",
            param1="value1",
            param2=42
        )
        self.builder = YourNewStepBuilder(self.config)
    
    def test_initialization(self):
        """Test that the builder initializes correctly with specification."""
        self.assertIsNotNone(self.builder.spec)
        self.assertEqual(self.builder.spec.step_type, YOUR_NEW_STEP_SPEC.step_type)
        self.assertEqual(self.builder.spec.node_type, NodeType.INTERNAL)  # Adjust as needed
    
    def test_get_inputs(self):
        """Test that inputs are correctly derived from dependencies."""
        # Mock input data
        inputs = {
            "input_data": "s3://bucket/input/data",
            "input_metadata": "s3://bucket/input/metadata"
        }
        
        # Get processing inputs
        processing_inputs = self.builder._get_inputs(inputs)
        
        # Verify inputs
        self.assertEqual(len(processing_inputs), 2)
        self.assertEqual(processing_inputs[0].input_name, "input_data")
        self.assertEqual(processing_inputs[0].source, "s3://bucket/input/data")
        self.assertEqual(processing_inputs[1].input_name, "input_metadata")
    
    def test_get_outputs(self):
        """Test that outputs are correctly configured."""
        # Get processing outputs
        processing_outputs = self.builder._get_outputs({})
        
        # Verify outputs
        self.assertEqual(len(processing_outputs), 2)
        self.assertEqual(processing_outputs[0].output_name, "output_data")
        self.assertEqual(processing_outputs[1].output_name, "output_metadata")
    
    def test_get_processor_env_vars(self):
        """Test that environment variables are correctly set."""
        env_vars = self.builder._get_processor_env_vars()
        
        self.assertEqual(env_vars["REQUIRED_PARAM_1"], "value1")
        self.assertEqual(env_vars["REQUIRED_PARAM_2"], "42")
    
    @patch('src.pipeline_steps.builder_your_new_step.YourNewStepBuilder._get_processor')
    def test_create_step(self, mock_get_processor):
        """Test step creation with dependencies."""
        # Mock dependencies
        dependencies = [MagicMock()]
        
        # Mock processor
        mock_processor = MagicMock()
        mock_get_processor.return_value = mock_processor
        
        # Create step
        self.builder.create_step(dependencies=dependencies, step_name="TestStep")
        
        # Verify processor was called
        mock_processor.run.assert_called_once()
```

**Create New File**: `test/pipeline_step_specs/test_your_new_step_spec.py`

```python
import unittest

from src.pipeline_step_specs.your_new_step_spec import YOUR_NEW_STEP_SPEC
from src.pipeline_deps.base_specifications import ValidationResult

class TestYourNewStepSpec(unittest.TestCase):
    def test_contract_alignment(self):
        """Test that spec and contract are properly aligned."""
        result = YOUR_NEW_STEP_SPEC.validate_contract_alignment()
        self.assertTrue(result.is_valid, f"Contract alignment validation failed: {result.errors}")
    
    def test_property_path_consistency(self):
        """Test property path consistency in outputs."""
        for output in YOUR_NEW_STEP_SPEC.outputs.values():
            expected = f"properties.ProcessingOutputConfig.Outputs['{output.logical_name}'].S3Output.S3Uri"
            self.assertEqual(output.property_path, expected,
                           f"Property path inconsistency in {output.logical_name}")
    
    def test_dependency_specifications(self):
        """Test dependency specifications."""
        # Test required dependency
        input_data_dep = YOUR_NEW_STEP_SPEC.dependencies.get("input_data")
        self.assertIsNotNone(input_data_dep)
        self.assertTrue(input_data_dep.required)
        self.assertEqual(input_data_dep.dependency_type, DependencyType.PROCESSING_OUTPUT)
        
        # Test optional dependency
        input_metadata_dep = YOUR_NEW_STEP_SPEC.dependencies.get("input_metadata")
        self.assertIsNotNone(input_metadata_dep)
        self.assertFalse(input_metadata_dep.required)
```

### 8. Integrate With Pipeline Templates

Finally, make your step usable in pipeline templates:

**File to Update**: `src/pipeline_builder/template_pipeline_your_template.py`

```python
# Add your step to the template's DAG creation
def _create_pipeline_dag(self) -> PipelineDAG:
    dag = PipelineDAG()
    
    # Add your node
    dag.add_node("your_new_step")
    
    # Add connections
    dag.add_edge("previous_step", "your_new_step")
    dag.add_edge("your_new_step", "next_step")
    
    return dag

# Add your configuration to the template's config map
def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
    config_map = {}
    
    # Add your config
    your_new_step_config = self._get_config_by_type(YourNewStepConfig)
    if your_new_step_config:
        config_map["your_new_step"] = your_new_step_config
    
    # Other configs...
    return config_map

# Add your builder to the template's builder map
def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
    return {
        # Existing mappings...
        "your_new_step": YourNewStepBuilder
    }
```

## Alignment and Validation

Throughout this process, it's crucial to ensure alignment between components:

1. **Script to Contract Alignment**: Ensure your script uses the paths defined in the contract
2. **Contract to Specification Alignment**: Ensure logical names match between contract and specification
3. **Specification to Dependencies Alignment**: Ensure your dependencies match the upstream step outputs
4. **Property Path Consistency**: Ensure property paths follow the standard format

Use the [validation checklist](validation_checklist.md) to verify your implementation before integration.
