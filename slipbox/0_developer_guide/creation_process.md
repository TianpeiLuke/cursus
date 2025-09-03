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
7. **Run validation framework tests**
8. Create unit tests
9. Integrate with pipeline templates

## Detailed Steps

### 1. Register the New Step Name

First, register your step in the central step registry:

**File to Update**: `src/cursus/registry/step_names.py`

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

**File to Update**: `src/cursus/registry/step_names.py`

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

### 7. Run Validation Framework Tests

Before proceeding with unit tests, run the comprehensive validation framework to ensure your step implementation is correct.

**For complete usage instructions, see the [Validation Framework Guide](validation_framework_guide.md).**

#### 7.1 Unified Alignment Tester

Execute the **Unified Alignment Tester** located in `cursus/validation/alignment` to perform 4-tier validation:

**Option A: Using CLI Commands (Recommended)**
```bash
# Validate a specific script with detailed output
python -m cursus.cli.alignment_cli validate your_new_step --verbose --show-scoring

# Validate a specific alignment level
python -m cursus.cli.alignment_cli validate-level your_new_step 1 --verbose

# Generate visualization and scoring reports
python -m cursus.cli.alignment_cli visualize your_new_step --output-dir ./validation_reports --verbose

# Run validation for all scripts
python -m cursus.cli.alignment_cli validate-all --output-dir ./reports --format both --verbose
```

**Option B: Using Test Scripts**
```bash
# Run individual validation script (create based on existing patterns)
python test/steps/scripts/alignment_validation/validate_your_new_step.py

# Run comprehensive alignment validation for all scripts
python test/steps/scripts/alignment_validation/run_alignment_validation.py
```

**Option C: Direct Python Usage**
```python
#!/usr/bin/env python3
"""
Alignment validation for your new step.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

def main():
    """Run alignment validation for your new step."""
    print("🔍 Your New Step Alignment Validation")
    print("=" * 60)
    
    # Initialize the tester with directory paths
    tester = UnifiedAlignmentTester(
        scripts_dir=str(project_root / "src" / "cursus" / "steps" / "scripts"),
        contracts_dir=str(project_root / "src" / "cursus" / "steps" / "contracts"),
        specs_dir=str(project_root / "src" / "cursus" / "steps" / "specs"),
        builders_dir=str(project_root / "src" / "cursus" / "steps" / "builders"),
        configs_dir=str(project_root / "src" / "cursus" / "steps" / "configs")
    )
    
    # Run validation for your specific script
    script_name = "your_new_step"  # Replace with your actual script name
    
    try:
        results = tester.validate_specific_script(script_name)
        
        # Print results
        status = results.get('overall_status', 'UNKNOWN')
        status_emoji = '✅' if status == 'PASSING' else '❌'
        print(f"{status_emoji} Overall Status: {status}")
        
        # Print level-by-level results
        for level_num, level_name in enumerate([
            "Script ↔ Contract",
            "Contract ↔ Specification", 
            "Specification ↔ Dependencies",
            "Builder ↔ Configuration"
        ], 1):
            level_key = f"level{level_num}"
            level_result = results.get(level_key, {})
            level_passed = level_result.get('passed', False)
            level_issues = level_result.get('issues', [])
            
            status_emoji = '✅' if level_passed else '❌'
            print(f"\n{status_emoji} Level {level_num}: {level_name}")
            print(f"   Status: {'PASS' if level_passed else 'FAIL'}")
            print(f"   Issues: {len(level_issues)}")
            
            # Print issues with details
            for issue in level_issues:
                severity = issue.get('severity', 'ERROR')
                message = issue.get('message', 'No message')
                recommendation = issue.get('recommendation', '')
                
                print(f"   • {severity}: {message}")
                if recommendation:
                    print(f"     💡 Recommendation: {recommendation}")
        
        return 0 if status == 'PASSING' else 1
        
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

The 4-tier validation includes:
- **Level 1**: Script-Contract Alignment (script paths match contract definitions)
- **Level 2**: Contract-Specification Alignment (logical names consistency)
- **Level 3**: Specification-Dependencies Alignment (dependency compatibility)
- **Level 4**: Builder-Configuration Alignment (builder config integration)

#### 7.2 Universal Step Builder Test

Execute the **Universal Step Builder Test** located in `cursus/validation/builders` for comprehensive builder testing:

**Option A: Using CLI Commands (Recommended)**
```bash
# Run all tests for your builder with scoring
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_new_step.YourNewStepBuilder --scoring --verbose

# Run specific level tests
python -m cursus.cli.builder_test_cli level 1 src.cursus.steps.builders.builder_your_new_step.YourNewStepBuilder --verbose

# Test all builders of your step type (e.g., Processing)
python -m cursus.cli.builder_test_cli test-by-type Processing --verbose --scoring

# Export results to JSON and generate charts
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_new_step.YourNewStepBuilder --export-json ./reports/builder_test_results.json --export-chart --output-dir ./reports
```

**Option B: Using Test Scripts (Pattern from existing tests)**
```bash
# Run Processing-specific tests (if your step is a Processing step)
python test/steps/builders/run_processing_tests.py

# Run Training-specific tests (if your step is a Training step)
python test/steps/builders/run_training_tests.py

# Run Transform-specific tests (if your step is a Transform step)
python test/steps/builders/run_transform_tests.py

# Run CreateModel-specific tests (if your step is a CreateModel step)
python test/steps/builders/run_createmodel_tests.py
```

**Option C: Direct Python Usage (Following existing patterns)**
```python
#!/usr/bin/env python3
"""
Builder validation for your new step.
Based on pattern from test_processing_step_builders.py
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cursus.validation.builders.universal_test import UniversalStepBuilderTest

def main():
    """Run builder validation for your new step."""
    print("🔧 Your New Step Builder Validation")
    print("=" * 60)
    
    # Import your builder class
    from cursus.steps.builders.builder_your_new_step import YourNewStepBuilder
    
    try:
        # Initialize the tester with enhanced features
        tester = UniversalStepBuilderTest(
            YourNewStepBuilder, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run all tests
        results = tester.run_all_tests()
        
        # Extract test results from enhanced format
        test_results = results.get('test_results', results) if isinstance(results, dict) and 'test_results' in results else results
        
        # Print results
        passed_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get("passed", False))
        total_tests = len([r for r in test_results.values() if isinstance(r, dict)])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 Builder Test Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = {k: v for k, v in test_results.items() 
                       if isinstance(v, dict) and not v.get("passed", True)}
        
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test_name, result in failed_tests.items():
                print(f"  • {test_name}: {result.get('error', 'Unknown error')}")
        else:
            print("\n✅ All builder tests passed!")
        
        # Print scoring information if available
        scoring = results.get('scoring', {})
        if scoring:
            print(f"\n📈 Scoring Information:")
            for metric, value in scoring.items():
                print(f"  • {metric}: {value}")
        
        return 0 if pass_rate == 100 else 1
        
    except Exception as e:
        print(f"❌ ERROR during builder validation: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

The 4-level testing includes:
- **Level 1**: Interface Testing (builder interface compliance)
- **Level 2**: Specification Testing (spec-driven functionality)
- **Level 3**: Path Mapping Testing (input/output path correctness)
- **Level 4**: Integration Testing (end-to-end step creation)

#### 7.3 Step Type-Specific Validation

The validation framework automatically applies step type-specific validation variants based on your `sagemaker_step_type`:

- **Processing Steps**: Standard processing validation patterns
- **Training Steps**: Training-specific validation with hyperparameter checks
- **Transform Steps**: Transform-specific validation patterns
- **CreateModel Steps**: Model creation validation patterns
- **RegisterModel Steps**: Model registration validation patterns

#### 7.4 Running the Validation Tests

You can run these validation tests in several ways:

**CLI Commands (Recommended)**
```bash
# Alignment validation using CLI
python -m cursus.cli.alignment_cli validate your_new_step --verbose --show-scoring
python -m cursus.cli.alignment_cli visualize your_new_step --output-dir ./validation_reports

# Builder validation using CLI
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_your_new_step.YourNewStepBuilder --scoring --verbose
python -m cursus.cli.builder_test_cli test-by-type Processing --verbose --scoring
```

**Test Scripts (Following existing patterns)**
```bash
# Run alignment validation scripts
python test/steps/scripts/alignment_validation/run_alignment_validation.py
python test/steps/scripts/alignment_validation/validate_your_new_step.py

# Run builder test scripts by step type
python test/steps/builders/run_processing_tests.py  # For Processing steps
python test/steps/builders/run_training_tests.py    # For Training steps
python test/steps/builders/run_transform_tests.py   # For Transform steps
python test/steps/builders/run_createmodel_tests.py # For CreateModel steps
```

**Important**: Both validation frameworks must pass before proceeding to unit tests and integration.

### 8. Create Unit Tests
### 7. Run Validation Framework Tests

Before proceeding with unit tests, run the comprehensive validation framework to ensure your step implementation is correct:

#### 7.1 Unified Alignment Tester

Execute the **Unified Alignment Tester** located in `cursus/validation/alignment` to perform 4-tier validation:

**Option A: Using CLI Commands (Recommended)**
```bash
# Validate a specific script with detailed output and scoring
python -m cursus.cli.alignment_cli validate your_new_step --verbose --show-scoring

# Validate a specific alignment level only
python -m cursus.cli.alignment_cli validate-level your_new_step 1 --verbose

# Generate comprehensive visualization and scoring reports
python -m cursus.cli.alignment_cli visualize your_new_step --output-dir ./validation_reports --verbose

# Run validation for all scripts with reports
python -m cursus.cli.alignment_cli validate-all --output-dir ./reports --format both --verbose
```

**Option B: Using Test Scripts (Pattern from existing tests)**
```bash
# Create individual validation script following the pattern from validate_tabular_preprocessing.py
python test/steps/scripts/alignment_validation/validate_your_new_step.py

# Run comprehensive alignment validation for all scripts
python test/steps/scripts/alignment_validation/run_alignment_validation.py
```

**Option C: Direct Python Usage (Following existing patterns)**
```python
#!/usr/bin/env python3
"""
Alignment validation for your new step.
Based on pattern from validate_tabular_preprocessing.py
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

def main():
    """Run alignment validation for your new step."""
    print("🔍 Your New Step Alignment Validation")
    print("=" * 60)
    
    # Initialize the tester with directory paths
    tester = UnifiedAlignmentTester(
        scripts_dir=str(project_root / "src" / "cursus" / "steps" / "scripts"),
        contracts_dir=str(project_root / "src" / "cursus" / "steps" / "contracts"),
        specs_dir=str(project_root / "src" / "cursus" / "steps" / "specs"),
        builders_dir=str(project_root / "src" / "cursus" / "steps" / "builders"),
        configs_dir=str(project_root / "src" / "cursus" / "steps" / "configs")
    )
    
    # Run validation for your specific script
    script_name = "your_new_step"  # Replace with your actual script name
    
    try:
        results = tester.validate_specific_script(script_name)
        
        # Print results
        status = results.get('overall_status', 'UNKNOWN')
        status_emoji = '✅' if status == 'PASSING' else '❌'
        print(f"{status_emoji} Overall Status: {status}")
        
        # Print level-by-level results
        for level_num, level_name in enumerate([
            "Script ↔ Contract",
            "Contract ↔ Specification", 
            "Specification ↔ Dependencies",
            "Builder ↔ Configuration"
        ], 1):
            level_key = f"level{level_num}"
            level_result = results.get(level_key, {})
            level_passed = level_result.get('passed', False)
            level_issues = level_result.get('issues', [])
            
            status_emoji = '✅' if level_passed else '❌'
            print(f"\n{status_emoji} Level {level_num}: {level_name}")
            print(f"   Status: {'PASS' if level_passed else 'FAIL'}")
            print(f"   Issues: {len(level_issues)}")
            
            # Print issues with details
            for issue in level_issues:
                severity = issue.get('severity', 'ERROR')
                message = issue.get('message', 'No message')
                recommendation = issue.get('recommendation', '')
                
                print(f"   • {severity}: {message}")
                if recommendation:
                    print(f"     💡 Recommendation: {recommendation}")
        
        return 0 if status == 'PASSING' else 1
        
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
```

The 4-tier validation includes:
- **Level 1**: Script-Contract Alignment (script paths match contract definitions)
- **Level 2**: Contract-Specification Alignment (logical names consistency)
- **Level 3**: Specification-Dependencies Alignment (dependency compatibility)
- **Level 4**: Builder-Configuration Alignment (builder config integration)

### 8. Create Unit Tests

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
