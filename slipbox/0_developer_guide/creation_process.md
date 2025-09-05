# Step Creation Process

This document outlines the step-by-step process for adding a new step to the pipeline. Follow these steps in order to ensure proper integration with the existing architecture.

## Table of Contents

1. [Set Up Workspace Context](#1-set-up-workspace-context)
2. [Create the Step Configuration](#2-create-the-step-configuration)
3. [Create the Script Contract](#3-create-the-script-contract)
4. [Create the Step Specification](#4-create-the-step-specification)
5. [Create the Step Builder](#5-create-the-step-builder)
6. [Register Step with Hybrid Registry System](#6-register-step-with-hybrid-registry-system)
7. [Run Validation Framework Tests](#7-run-validation-framework-tests)
8. [Create Unit Tests](#8-create-unit-tests)
9. [Integrate With Pipeline Templates](#9-integrate-with-pipeline-templates)

## Overview of the Process

Adding a new step to the pipeline involves creating several components that work together:

1. Set up workspace context
2. Create the step configuration class
3. Create the script contract
4. Create the step specification
5. Implement the step builder (with automatic registry integration)
6. Register step with hybrid registry system
7. **Run validation framework tests**
8. Create unit tests
9. Integrate with pipeline templates

## Detailed Steps

### 1. Set Up Workspace Context

First, determine your development approach and set up the appropriate workspace context:

#### For Main Workspace Development (`src/cursus/`)

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Set main workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("main")
```

#### For Isolated Project Development (`development/projects/*/`)

```bash
# Initialize or activate project workspace
cursus init-workspace --project your_project --type isolated
cd development/projects/your_project
cursus activate-workspace your_project
```

### 2. Create the Step Configuration

Create a configuration class using the three-tier field classification design:

**Create New File**: `src/cursus/steps/configs/config_your_new_step.py`

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
        from ..contracts.your_new_step_contract import YOUR_NEW_STEP_CONTRACT
        return YOUR_NEW_STEP_CONTRACT
```

The configuration class follows the three-tier field classification:

1. **Tier 1 (Essential Fields)**: Required inputs from users (no defaults)
2. **Tier 2 (System Fields)**: Default values that can be overridden by users
3. **Tier 3 (Derived Fields)**: Private fields with public property access, computed from other fields

For more details on the three-tier design, see [Three-Tier Config Design](three_tier_config_design.md).

### 3. Create the Script Contract

Define the contract between your script and the SageMaker environment:

**Create New File**: `src/cursus/steps/contracts/your_new_step_contract.py`

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

**Create New File**: `src/cursus/steps/specs/your_new_step_spec.py`

```python
from typing import Dict, List, Optional

from ...core.base.specification_base import StepSpecification, NodeType, DependencySpec, OutputSpec, DependencyType
from ...registry.step_names import get_spec_step_type

def _get_your_new_step_contract():
    """Get the script contract for this step."""
    from ..contracts.your_new_step_contract import YOUR_NEW_STEP_CONTRACT
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

Implement the builder that creates the SageMaker step using the modern hybrid registry system:

**Create New File**: `src/cursus/steps/builders/builder_your_new_step.py`

```python
from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ...registry.hybrid.manager import UnifiedRegistryManager
from ...core.base.specification_base import StepSpecification
from ...core.base.builder_base import StepBuilderBase
from ..configs.config_your_new_step import YourNewStepConfig
from ..specs.your_new_step_spec import YOUR_NEW_STEP_SPEC

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
        
        # Register with UnifiedRegistryManager (automatic discovery handles this)
        if registry_manager is None:
            registry_manager = UnifiedRegistryManager()
        # Registration is handled automatically by the hybrid registry system
        # based on naming conventions and file location
    
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
        
        Returns:
            ProcessingStep: The SageMaker processing step
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
- Integrates with the hybrid registry system through automatic discovery
- Gets the appropriate specification based on job type
- Creates a SageMaker processor (Processing step in this example)
- Sets up inputs and outputs based on the specification
- Configures environment variables from the config
- Creates and returns the SageMaker step

**Important**: The `create_step()` method returns a `ProcessingStep` in this example. Your step's `sagemaker_step_type` must match the actual SageMaker step type returned by this method. Available

### 6. Register Step with Hybrid Registry System

With the modern hybrid registry system, step registration is handled automatically through the UnifiedRegistryManager. However, you need to ensure your step is properly registered:

#### Option A: Automatic Registration (Recommended)

The UnifiedRegistryManager automatically discovers and registers your step if you follow the naming conventions:

1. **File Naming**: Your builder file should follow the pattern `builder_your_new_step.py`
2. **Class Naming**: Your builder class should be named `YourNewStepBuilder`
3. **Location**: Place your builder in `src/cursus/steps/builders/`

The registry will automatically:
- Discover your step builder
- Extract step metadata from your configuration and specification
- Register the step with the appropriate workspace context

#### Option B: Explicit Registration (For Custom Cases)

If you need explicit control over registration, use the registry's validation-enabled registration:

```python
from cursus.registry.step_names import add_new_step_with_validation

# Register your step with validation
warnings = add_new_step_with_validation(
    step_name="YourNewStep",
    config_class="YourNewStepConfig", 
    builder_name="YourNewStepBuilder",
    sagemaker_type="Processing",  # Based on create_step() return type
    description="Description of your new step",
    validation_mode="warn",  # Options: "warn", "strict", "auto_correct"
    workspace_id=None  # Use current workspace context
)

# Check for any validation warnings
if warnings:
    for warning in warnings:
        print(f"⚠️ {warning}")
```

#### Option C: Workspace-Specific Registration

For isolated workspace development:

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry manager with workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

# Register step in specific workspace
registry.register_step_definition(
    "YourNewStep",
    {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder", 
        "spec_type": "YourNewStep",
        "sagemaker_step_type": "Processing",
        "description": "Description of your new step"
    }
)
```

#### Verification

Verify your step registration:

```bash
# List all registered steps
cursus list-steps

# List steps in specific workspace
cursus list-steps --workspace your_project

# Check step details
cursus describe-step YourNewStep
```

**Important**: The `sagemaker_step_type` field must match the actual SageMaker step type returned by your step builder's `create_step()` method. This field is used by the Universal Builder Test framework for step-type-specific validation and testing.

**Available SageMaker Step Types:**
- **Processing**: For data processing, feature engineering, and data transformation steps
- **Training**: For model training steps using SageMaker training jobs
- **Transform**: For batch transform jobs and model inference steps
- **CreateModel**: For creating SageMaker model artifacts and endpoints
- **RegisterModel**: For registering models in the SageMaker Model Registry
- **Lambda**: For AWS Lambda function execution steps
- **Base**: For custom step implementations that don't fit standard categories

Choose the appropriate `sagemaker_step_type` based on what your `create_step()` method returns:
- Return `ProcessingStep` → use `sagemaker_step_type="Processing"`
- Return `TrainingStep` → use `sagemaker_step_type="Training"`
- Return `TransformStep` → use `sagemaker_step_type="Transform"`
- Return `CreateModelStep` → use `sagemaker_step_type="CreateModel"`
- Return `RegisterModelStep` → use `sagemaker_step_type="RegisterModel"`
- Return `LambdaStep` → use `sagemaker_step_type="Lambda"`
- Return custom step → use `sagemaker_step_type="Base"`

**Note**: The hybrid registry system maintains backward compatibility while providing workspace isolation and automatic discovery. Manual updates to `step_names_original.py` are only needed for legacy compatibility or when working with the fallback registry system.

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

Implement tests to verify your components work correctly:

**Create New File**: `test/steps/builders/test_builder_your_new_step.py`

```python
import unittest
from unittest.mock import MagicMock, patch

from cursus.steps.builders.builder_your_new_step import YourNewStepBuilder
from cursus.steps.configs.config_your_new_step import YourNewStepConfig
from cursus.steps.specs.your_new_step_spec import YOUR_NEW_STEP_SPEC
from cursus.core.base.specification_base import NodeType, DependencyType

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
    
    @patch('cursus.steps.builders.builder_your_new_step.YourNewStepBuilder._get_processor')
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

**Create New File**: `test/steps/specs/test_your_new_step_spec.py`

```python
import unittest

from cursus.steps.specs.your_new_step_spec import YOUR_NEW_STEP_SPEC
from cursus.steps.specs.base_specifications import ValidationResult

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

### 9. Integrate With Pipeline Catalog

Once your step is created and validated, it becomes available for use in the Pipeline Catalog system. The modern pipeline catalog uses a Zettelkasten-based approach with connection-based discovery rather than traditional templates.

#### Automatic Step Discovery

Your step is automatically available to all pipelines once registered with the hybrid registry system:

```bash
# Verify your step is available
cursus list-steps

# Check if your step appears in pipeline discovery
cursus catalog find --tags your_step_tags
```

#### Using Your Step in Existing Pipelines

Your step can be used in any pipeline that matches its dependency requirements:

```python
# Example: Using your step in a pipeline
from cursus.pipeline_catalog.pipelines.your_pipeline import create_pipeline
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Your step is automatically available through the registry
registry = UnifiedRegistryManager()
available_steps = registry.list_steps()

# Pipeline builders can now discover and use your step
pipeline, report, dag_compiler, template = create_pipeline(
    config_path="config.json",
    session=pipeline_session,
    role=role
)
```

#### Creating New Pipelines with Your Step

To create a new pipeline that uses your step, follow the Pipeline Catalog patterns:

**Create New File**: `src/cursus/pipeline_catalog/pipelines/pipeline_with_your_step.py`

```python
"""
Pipeline using YourNewStep - A pipeline demonstrating your new step.

This pipeline showcases the usage of YourNewStep in a complete workflow.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from cursus.core.dag.dag_compiler import DAGCompiler
from cursus.core.dag.pipeline_dag import PipelineDAG
from cursus.pipeline_catalog.utils.zettelkasten_metadata import ZettelkastenMetadata
from cursus.steps.builders.builder_your_new_step import YourNewStepBuilder
from cursus.steps.configs.config_your_new_step import YourNewStepConfig

# Zettelkasten metadata for pipeline discovery
PIPELINE_METADATA = ZettelkastenMetadata(
    id="pipeline_with_your_step",
    title="Pipeline with YourNewStep",
    description="A complete pipeline demonstrating YourNewStep usage",
    tags=["your_new_step", "processing", "example"],
    framework_tags=["sagemaker"],
    task_tags=["processing", "transformation"],
    complexity_tags=["standard"],
    connections={
        "alternatives": ["other_similar_pipeline"],
        "extensions": ["enhanced_pipeline_with_your_step"],
        "components": ["your_new_step"],
        "progressions": ["basic_pipeline", "advanced_pipeline"]
    },
    use_cases=[
        "Demonstrate YourNewStep functionality",
        "Process data using your new step",
        "Example workflow with custom step"
    ]
)

def create_dag() -> PipelineDAG:
    """Create the pipeline DAG with YourNewStep."""
    dag = PipelineDAG()
    
    # Add your step to the DAG
    dag.add_node("your_new_step")
    
    # Add other steps as needed
    dag.add_node("preprocessing_step")
    dag.add_node("postprocessing_step")
    
    # Define step connections
    dag.add_edge("preprocessing_step", "your_new_step")
    dag.add_edge("your_new_step", "postprocessing_step")
    
    return dag

def create_pipeline(
    config_path: str,
    session,
    role: str,
    enable_mods: bool = False
) -> Tuple[Any, Dict[str, Any], DAGCompiler, Optional[Any]]:
    """
    Create pipeline with YourNewStep.
    
    Args:
        config_path: Path to configuration file
        session: SageMaker session
        role: IAM role ARN
        enable_mods: Enable MODS features if available
    
    Returns:
        Tuple of (pipeline, report, dag_compiler, template)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create step configurations
    your_step_config = YourNewStepConfig(**config.get("your_new_step", {}))
    
    # Create step builders
    step_builders = {
        "your_new_step": YourNewStepBuilder(
            config=your_step_config,
            sagemaker_session=session,
            role=role
        )
    }
    
    # Create DAG
    dag = create_dag()
    
    # Create DAG compiler
    dag_compiler = DAGCompiler(
        dag=dag,
        step_builders=step_builders,
        session=session,
        role=role
    )
    
    # Compile pipeline
    pipeline = dag_compiler.compile()
    
    # Generate report
    report = {
        "pipeline_name": "pipeline_with_your_step",
        "steps": list(step_builders.keys()),
        "dag_structure": dag.to_dict(),
        "metadata": PIPELINE_METADATA.to_dict()
    }
    
    # MODS integration (if available)
    template = None
    if enable_mods:
        try:
            from cursus.mods.template_registration import register_template
            template = register_template(pipeline, PIPELINE_METADATA)
        except ImportError:
            # Graceful fallback if MODS not available
            pass
    
    # Sync to registry
    sync_to_registry(PIPELINE_METADATA)
    
    return pipeline, report, dag_compiler, template

def get_enhanced_dag_metadata() -> Dict[str, Any]:
    """Get enhanced metadata for the DAG."""
    return {
        "zettelkasten_metadata": PIPELINE_METADATA.to_dict(),
        "step_metadata": {
            "your_new_step": {
                "builder_class": "YourNewStepBuilder",
                "config_class": "YourNewStepConfig",
                "sagemaker_step_type": "Processing"
            }
        }
    }

def sync_to_registry(metadata: ZettelkastenMetadata) -> None:
    """Sync pipeline metadata to the catalog registry."""
    from cursus.pipeline_catalog.utils.registry_sync import sync_pipeline_metadata
    sync_pipeline_metadata(metadata)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)
```

#### Pipeline Discovery and Usage

Your step becomes discoverable through the catalog system:

```bash
# Find pipelines that use your step
cursus catalog find --tags your_new_step

# Get recommendations for pipelines with your step
cursus catalog recommend --use-case "pipeline using YourNewStep"

# Show connections to pipelines using your step
cursus catalog connections --pipeline pipeline_with_your_step
```

#### Best Practices for Pipeline Integration

1. **Use Descriptive Metadata**: Include comprehensive ZettelkastenMetadata with relevant tags and connections
2. **Follow Naming Conventions**: Use semantic naming like `{use_case}_{complexity}` (e.g., `data_processing_standard`)
3. **Define Clear Connections**: Specify alternatives, extensions, and progressions to help users discover related pipelines
4. **Include Use Cases**: Provide clear use case descriptions for better discoverability
5. **Enable MODS Integration**: Support MODS features with graceful fallback for enhanced operational capabilities

#### Validation

Validate your pipeline integration:

```bash
# Validate pipeline registry
cursus catalog registry validate

# Test pipeline discovery
cursus catalog find --pipeline pipeline_with_your_step

# Check pipeline metadata
cursus catalog registry export --pipelines pipeline_with_your_step
```

Your step is now fully integrated into the Pipeline Catalog ecosystem and can be discovered, used, and connected with other pipelines through the modern Zettelkasten-based approach.

## Alignment and Validation

Throughout this process, it's crucial to ensure alignment between components:

1. **Script to Contract Alignment**: Ensure your script uses the paths defined in the contract
2. **Contract to Specification Alignment**: Ensure logical names match between contract and specification
3. **Specification to Dependencies Alignment**: Ensure your dependencies match the upstream step outputs
4. **Property Path Consistency**: Ensure property paths follow the standard format

Use the [validation checklist](validation_checklist.md) to verify your implementation before integration.

## Related Documentation

- [Pipeline Catalog Integration Guide](pipeline_catalog_integration_guide.md) - How to integrate your pipeline steps with the Zettelkasten-inspired catalog system
- [Adding New Pipeline Step](adding_new_pipeline_step.md) - Main developer guide with overview and quick start
- [Prerequisites](prerequisites.md) - Required information before starting step creation
- [Design Principles](design_principles.md) - Core architectural principles to follow
- [Best Practices](best_practices.md) - Recommended practices for step development
- [Standardization Rules](standardization_rules.md) - Coding and naming conventions
- [Alignment Rules](alignment_rules.md) - Component alignment requirements
- [Validation Framework Guide](validation_framework_guide.md) - Comprehensive validation usage instructions
- [Validation Checklist](validation_checklist.md) - Pre-integration validation checklist
- [Step Builder Guide](step_builder.md) - Detailed step builder implementation patterns
- [Script Contract Development](script_contract.md) - Script contract creation guide
- [Step Specification Development](step_specification.md) - Step specification creation guide
- [Three-Tier Config Design](three_tier_config_design.md) - Configuration design patterns
- [Step Builder Registry Guide](step_builder_registry_guide.md) - Comprehensive guide to the UnifiedRegistryManager and hybrid registry system
- [Step Builder Registry Usage](step_builder_registry_usage.md) - Practical examples and usage patterns for registry operations
- [Common Pitfalls](common_pitfalls.md) - Common mistakes to avoid
- [Example](example.md) - Complete step implementation example
