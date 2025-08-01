# Implementation Plan for DummyTraining Step - Version 4

## Document History
- **Version 1**: Initial implementation plan
- **Version 2**: Revised based on validator feedback to address identified issues
- **Version 3**: Updated for compatibility with packaging step dependencies
- **Version 4**: Enhanced based on validation report to address critical implementation issues

## 1. Step Overview
- **Purpose**: Create a training step that bypasses actual training by taking a pretrained model.tar.gz file from a local folder and making it available for downstream packaging and registration steps
- **Inputs**: Path to a pretrained model.tar.gz file
- **Outputs**: S3 path to the uploaded model artifacts
- **Position in pipeline**: Between data preprocessing and model packaging/registration steps
- **Architectural considerations**: 
  - Using a Processing step instead of a Training step since we're not actually training, just moving a file
  - Must maintain compatibility with existing step interfaces for seamless integration
  - Output must be compatible with the packaging step's expected inputs
- **Alignment with design principles**: Follows the specification-driven architecture with clear separation of concerns across the four layers

## 2. Components to Create

### Script Contract: src/pipeline_script_contracts/dummy_training_contract.py
- **Input paths**:
  - `pretrained_model_path`: "/opt/ml/processing/input/model/model.tar.gz" (Path to pretrained model file)
- **Output paths**:
  - `model_input`: "/opt/ml/processing/output/model" (Where the model will be copied to)
- **Environment variables**: None required
- **Framework requirements**: 
  - `boto3`: ">=1.26.0"
  - `pathlib`: ">=1.0.0"
  - `tarfile`: (standard library)

### Step Specification: src/pipeline_step_specs/dummy_training_spec.py
- **Dependencies**:
  - `pretrained_model_path`: Path to pretrained model file (DependencyType.PROCESSING_INPUT)
    - Compatible sources: ["ProcessingStep", "XGBoostTraining", "PytorchTraining", "TabularPreprocessing"]
    - Semantic keywords: ["model", "pretrained", "artifact", "weights", "training_output", "model_data"]
- **Outputs**:
  - `model_input`: S3 path to the model artifacts (DependencyType.MODEL_ARTIFACTS) *(Updated for Package step compatibility)*
    - Property path: "properties.ProcessingOutputConfig.Outputs['model_input'].S3Output.S3Uri"
    - Aliases: ["ModelOutputPath", "ModelArtifacts", "model_data", "output_path"]

### Configuration: src/pipeline_steps/config_dummy_training.py
- **Step-specific parameters**:
  - `pretrained_model_path`: Local path to the pretrained model.tar.gz file
- **SageMaker parameters**:
  - instance_type: Small instance type, default "ml.m5.large"
  - instance_count: Default 1
  - volume_size_gb: Default 30
  - max_runtime_seconds: Default 1200 (20 minutes)

### Step Builder: src/pipeline_steps/builder_dummy_training.py
- **Special handling**:
  - Upload the local pretrained model to S3
  - Create a minimal processing script to copy the model from input to output
  - Set up the processing step to execute this script
  - Configure caching for the step
- **Required Helper Methods** (Added based on validation report):
  - `_normalize_s3_uri`: Handle S3 paths with proper PipelineVariable support
  - `_validate_s3_uri`: Validate S3 URIs including PipelineVariable handling
  - `_get_s3_directory_path`: Get directory path from S3 URI

### Processing Script: src/pipeline_scripts/dummy_training.py
- **Algorithm**: 
  - Model validation and file copy from input to output location
- **Main functions**:
  - `validate_model`: Verify the model file format and structure
  - `copy_model`: Copy the validated model.tar.gz from input to output directory
  - `main`: Parse arguments, validate paths, and execute the copy

## 3. Files to Update
- src/pipeline_registry/step_names.py
  - Add "DummyTraining" to STEP_NAMES dictionary with correct builder name
```python
STEP_NAMES = {
    # ... existing steps ...
    
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "description": "Training step that uses a pretrained model"
    },
}
```
- src/pipeline_steps/__init__.py
  - Import and expose DummyTrainingStepBuilder
- src/pipeline_step_specs/__init__.py
  - Import and expose DUMMY_TRAINING_SPEC
- src/pipeline_script_contracts/__init__.py
  - Import and expose DUMMY_TRAINING_CONTRACT
- src/pipeline_step_specs/packaging_spec.py (Optional enhancement for future robustness)
  - Add "DummyTraining" to the compatible_sources list in the model_input dependency

## 4. Integration Strategy
- **Upstream steps**: XGBoostTraining, PytorchTraining, TabularPreprocessing, or any ProcessingStep
- **Downstream steps**: ModelPackagingStep, MIMSRegistrationStep
- **DAG updates**: Add DummyTraining node between preprocessing and packaging steps
- **Compatibility enhancements**: 
  - Using MODEL_ARTIFACTS output type to match packaging step dependency type
  - Using matching logical name (model_input) with the packaging step dependency

## 5. Contract-Specification Alignment
- **Input alignment**: 
  - Contract input path `pretrained_model_path` maps to specification dependency `pretrained_model_path`
- **Output alignment**: 
  - Contract output path `model_input` maps to specification output `model_input`
- **Validation strategy**:
  - Use existing validation utilities to ensure input/output paths match between contract and specification
  - Verify `validate_contract_alignment()` method returns successful result
- **Spec/Contract Validation** (Added based on validation report):
  - Add explicit validation in _get_inputs and _get_outputs to ensure spec and contract are available

## 6. Error Handling Strategy
- **Input validation**: 
  - Verify the pretrained model.tar.gz file exists
  - Validate file has .tar.gz extension
  - Verify it's a valid tar archive using tarfile.is_tarfile()
- **Script robustness**: 
  - Handle missing input files gracefully
  - Provide meaningful error messages for common failure modes
  - Use try/except blocks with specific exception handling
- **Logging strategy**: 
  - Log file sizes and paths for debugging
  - Log validation success/failure
  - Clearly indicate success or failure of the copy operation
  - Include detailed error information in exception messages
- **Error reporting**: 
  - Raise specific exceptions with descriptive messages
  - Include traceback information for debugging
  - Return appropriate error codes from script

## 7. Testing and Validation Plan
- **Unit tests**: test/pipeline_steps/test_dummy_training.py
  - Test DummyTrainingStepBuilder initialization
  - Test configuration validation
  - Test input/output handling
  - Test model file validation logic
  - Test error handling scenarios
  - **Test PipelineVariable handling** (Added based on validation report)
  - **Test S3 path handling helper methods** (Added based on validation report)
- **Integration tests**: test/integration/test_dummy_training_integration.py
  - Test end-to-end flow from preprocessing to packaging using the dummy training step
  - Test with various input model formats
  - Test error recovery scenarios
- **Specification validation tests**:
  - Test contract-specification alignment
  - Verify property paths are correct
- **Validation criteria**:
  - Successful validation and copy of model.tar.gz to the output location
  - Correct property paths in specification outputs
  - Compatible integration with downstream model packaging steps
  - Successful dependency resolution with the packaging step

## Implementation Details

### 1. Step Registry Addition

```python
# In src/pipeline_registry/step_names.py
STEP_NAMES = {
    # ... existing steps ...
    
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "description": "Training step that uses a pretrained model"
    },
}
```

### 2. Script Contract Implementation

```python
# src/pipeline_script_contracts/dummy_training_contract.py
from .base_script_contract import ScriptContract

DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        "pretrained_model_path": "/opt/ml/processing/input/model/model.tar.gz"
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"  # Matches specification logical name
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "pathlib": ">=1.0.0"
    },
    description="Contract for dummy training step that copies a pretrained model.tar.gz to output location"
)
```

### 3. Step Specification Implementation

```python
# src/pipeline_step_specs/dummy_training_spec.py
from ..pipeline_deps.base_specifications import StepSpecification, NodeType, DependencySpec, OutputSpec, DependencyType
from ..pipeline_registry.step_names import get_spec_step_type

def _get_dummy_training_contract():
    from ..pipeline_script_contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT
    return DUMMY_TRAINING_CONTRACT

DUMMY_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DummyTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_dummy_training_contract(),
    dependencies=[
        DependencySpec(
            logical_name="pretrained_model_path",
            dependency_type=DependencyType.PROCESSING_INPUT,
            required=True,
            compatible_sources=["ProcessingStep", "XGBoostTraining", "PytorchTraining", "TabularPreprocessing"],
            semantic_keywords=["model", "pretrained", "artifact", "weights", "training_output", "model_data"],
            data_type="S3Uri",
            description="Path to pretrained model.tar.gz file"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_input",  # Matches contract output path name
            output_type=DependencyType.MODEL_ARTIFACTS,  # For packaging compatibility
            property_path="properties.ProcessingOutputConfig.Outputs['model_input'].S3Output.S3Uri",
            data_type="S3Uri",
            description="S3 path to model artifacts",
            aliases=["ModelOutputPath", "ModelArtifacts", "model_data", "output_path"]
        )
    ]
)
```

### 4. Configuration Class Implementation

```python
# src/pipeline_steps/config_dummy_training.py
from .config_processing_step_base import ProcessingStepConfigBase

class DummyTrainingConfig(ProcessingStepConfigBase):
    """Configuration for DummyTraining."""
    
    def __init__(
        self,
        region: str,
        pipeline_s3_loc: str,
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
        volume_size_gb: int = 30,
        max_runtime_seconds: int = 1200,
        pretrained_model_path: str = "",
    ):
        """Initialize DummyTraining configuration.
        
        Args:
            region: AWS region
            pipeline_s3_loc: S3 location for pipeline artifacts
            instance_type: SageMaker instance type
            instance_count: Number of instances
            volume_size_gb: EBS volume size in GB
            max_runtime_seconds: Maximum runtime in seconds
            pretrained_model_path: Local path to pretrained model.tar.gz
        """
        super().__init__(region, pipeline_s3_loc)
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.volume_size_gb = volume_size_gb
        self.max_runtime_seconds = max_runtime_seconds
        self.pretrained_model_path = pretrained_model_path
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..pipeline_script_contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT
        return DUMMY_TRAINING_CONTRACT
        
    def get_script_path(self):
        """Return the path to the processing script."""
        return "src/pipeline_scripts/dummy_training.py"
```

### 5. Step Builder Implementation (Enhanced with S3 path handling)

```python
# src/pipeline_steps/builder_dummy_training.py
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List

from sagemaker.processing import Processor, ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.functions import Join
from sagemaker.s3 import S3Uploader

from .config_dummy_training_step import DummyTrainingConfig
from .builder_step_base import StepBuilderBase
from ..pipeline_step_specs.dummy_training_spec import DUMMY_TRAINING_SPEC

logger = logging.getLogger(__name__)

class DummyTrainingStepBuilder(StepBuilderBase):
    """Builder for DummyTraining processing step."""
    
    def __init__(
        self, 
        config: DummyTrainingConfig,
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        """Initialize the DummyTraining step builder.
        
        Args:
            config: Configuration for the DummyTraining step
            sagemaker_session: SageMaker session to use
            role: IAM role for SageMaker execution
            notebook_root: Root directory for notebook execution
            registry_manager: Registry manager for dependency injection
            dependency_resolver: Dependency resolver for dependency injection
        """
        if not isinstance(config, DummyTrainingConfig):
            raise ValueError("DummyTrainingStepBuilder requires a DummyTrainingConfig instance.")
        
        super().__init__(
            config=config,
            spec=DUMMY_TRAINING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: DummyTrainingConfig = config
    
    def validate_configuration(self):
        """
        Validate the provided configuration.
        
        Raises:
            ValueError: If pretrained_model_path is not provided
            FileNotFoundError: If the pretrained model file doesn't exist
        """
        self.log_info("Validating DummyTrainingConfig...")
        
        # Check for required local file
        if not self.config.pretrained_model_path:
            raise ValueError("pretrained_model_path is required in DummyTrainingConfig")
        
        # Check if file exists (if path is concrete and not a variable)
        if not hasattr(self.config.pretrained_model_path, 'expr'):
            model_path = Path(self.config.pretrained_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Pretrained model not found at {model_path}")
            
            # Additional validation: check file extension
            if not model_path.suffix == '.tar.gz' and not str(model_path).endswith('.tar.gz'):
                self.log_warning(f"Model file {model_path} does not have .tar.gz extension")
        
        self.log_info("DummyTrainingConfig validation succeeded.")
    
    def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
        """
        Normalizes an S3 URI to ensure it has no trailing slashes and is properly formatted.
        
        Args:
            uri: The S3 URI to normalize
            description: Description for logging purposes
            
        Returns:
            Normalized S3 URI
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
            self.log_info(f"Normalizing PipelineVariable URI: {uri}")
        
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference: %s", uri)
            return uri
        
        if not isinstance(uri, str):
            self.log_warning("Invalid %s URI type: %s", description, type(uri).__name__)
            return str(uri)
            
        if not uri.startswith('s3://'):
            self.log_warning("URI does not start with s3://: %s", uri)
        
        # Remove trailing slashes
        while uri.endswith('/'):
            uri = uri[:-1]
        
        return uri
    
    def _validate_s3_uri(self, uri: str, description: str = "S3 URI") -> bool:
        """
        Validates that a string is a properly formatted S3 URI.
        
        Args:
            uri: The URI to validate
            description: Description for error messages
            
        Returns:
            True if valid, False otherwise
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            self.log_info(f"Validating PipelineVariable URI: {uri.expr}")
            return True
        
        # Handle Pipeline step references
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info(f"Validating Pipeline reference URI: {uri}")
            return True
        
        if not isinstance(uri, str):
            self.log_warning("Invalid %s URI: type %s", description, type(uri).__name__)
            return False
        
        if not uri.startswith('s3://'):
            self.log_warning("Invalid %s URI: does not start with s3://", description)
            return False
        
        return True
    
    def _get_s3_directory_path(self, uri: str, filename: str = None) -> str:
        """
        Gets the directory part of an S3 URI, handling special cases correctly.
        
        Args:
            uri: The S3 URI which may or may not contain a filename
            filename: Optional filename to check for at the end of the URI
            
        Returns:
            The directory part of the URI without trailing slash
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
            self.log_info(f"Getting directory path for PipelineVariable URI: {uri}")
        
        # Handle Pipeline step references
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info(f"Cannot extract directory from Pipeline reference URI: {uri}")
            return uri
        
        # Remove trailing slashes
        uri = self._normalize_s3_uri(uri)
        
        # If a filename is provided and the URI ends with it, remove it
        if filename and uri.endswith(f"/{filename}"):
            uri = uri[:-len(filename)-1]
        
        return uri
    
    def _upload_model_to_s3(self) -> str:
        """
        Upload the pretrained model to S3.
        
        Returns:
            S3 URI where the model was uploaded
            
        Raises:
            Exception: If upload fails
        """
        self.log_info(f"Uploading pretrained model from {self.config.pretrained_model_path}")
        
        # Construct target S3 URI
        target_s3_uri = f"{self.config.pipeline_s3_loc}/dummy_training/input/model.tar.gz"
        target_s3_uri = self._normalize_s3_uri(target_s3_uri)
        
        try:
            # Upload the file
            S3Uploader.upload(
                self.config.pretrained_model_path,
                target_s3_uri,
                sagemaker_session=self.session
            )
            
            self.log_info(f"Uploaded model to {target_s3_uri}")
            return target_s3_uri
        except Exception as e:
            self.log_error(f"Failed to upload model to S3: {e}")
            raise
    
    def _get_processor(self):
        """
        Get the processor for the step.
        
        Returns:
            ScriptProcessor: Configured processor for running the step
        """
        return ScriptProcessor(
            image_uri="137112412989.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            command=["python3"],
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            volume_size_in_gb=self.config.volume_size_gb,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            role=self.role,
            sagemaker_session=self.session,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('DummyTraining')}"
            )
        )
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the processor using the specification and contract.
        
        Args:
            inputs: Dictionary of input sources keyed by logical name
            
        Returns:
            List of ProcessingInput objects for the processor
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
        
        processing_inputs = []
        
        # Use either the uploaded model or one provided through dependencies
        model_s3_uri = inputs.get("pretrained_model_path")
        if not model_s3_uri:
            # Upload the local model file if no S3 path is provided
            model_s3_uri = self._upload_model_to_s3()
        
        # Handle PipelineVariable objects
        if hasattr(model_s3_uri, 'expr'):
            self.log_info(f"Processing PipelineVariable for model_s3_uri: {model_s3_uri.expr}")
        
        # Get container path from contract
        container_path = self.contract.expected_input_paths.get("pretrained_model_path",
                          "/opt/ml/processing/input/model/model.tar.gz")
        
        # Add model input
        processing_inputs.append(
            ProcessingInput(
                source=model_s3_uri,
                destination=container_path,
                input_name="model"
            )
        )
        
        return processing_inputs
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the processor using the specification and contract.
        
        Args:
            outputs: Dictionary of output destinations keyed by logical name
            
        Returns:
            List of ProcessingOutput objects for the processor
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
        
        # Use the pipeline S3 location to construct output path
        default_output_path = f"{self.config.pipeline_s3_loc}/dummy_training/output"
        output_path = outputs.get("model_input", default_output_path)
        
        # Handle PipelineVariable objects in output_path
        if hasattr(output_path, 'expr'):
            self.log_info(f"Processing PipelineVariable for output_path: {output_path.expr}")
        
        # Get source path from contract
        source_path = self.contract.expected_output_paths.get("model_input",
                       "/opt/ml/processing/output/model")
        
        return [
            ProcessingOutput(
                output_name="model_input",  # Using consistent name matching specification
                source=source_path,
                destination=output_path
            )
        ]
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Create the processing step.
        
        Args:
            **kwargs: Additional keyword arguments for step creation.
                     Should include 'dependencies' list if step has dependencies.
                     
        Returns:
            ProcessingStep: The configured processing step
            
        Raises:
            ValueError: If inputs cannot be extracted
            Exception: If step creation fails
        """
        try:
            # Extract inputs from dependencies using the resolver
            dependencies = kwargs.get('dependencies', [])
            inputs = {}
            if dependencies:
                inputs = self.extract_inputs_from_dependencies(dependencies)
            
            # Create processor
            processor = self._get_processor()
            
            # Get processor inputs and outputs
            processing_inputs = self._get_inputs(inputs)
            processing_outputs = self._get_outputs({})
            
            # Create the step
            step_name = kwargs.get('step_name', 'DummyTraining')
            
            # Generate script arguments
            script_args = [
                "--pretrained-model-path", "/opt/ml/processing/input/model/model.tar.gz",
                "--output-dir", "/opt/ml/processing/output/model"
            ]
            
            # Get cache configuration
            cache_config = self._get_cache_config(kwargs.get('enable_caching', True))
            
            # Create the step
            step = processor.run(
                code=self.config.get_script_path(),
                inputs=processing_inputs,
                outputs=processing_outputs,
                arguments=script_args,
                job_name=self._generate_job_name(step_name),
                wait=False,
                cache_config=cache_config
            )
            
            # Store specification in step for future reference
            setattr(step, '_spec', self.spec)
            
            return step
        
        except Exception as e:
            self.log_error(f"Error creating DummyTraining step: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            raise ValueError(f"Failed to create DummyTraining step: {str(e)}") from e
```

### 6. Processing Script Implementation (Enhanced error messages)

```python
# src/pipeline_scripts/dummy_training.py
#!/usr/bin/env python
"""
DummyTraining Processing Script

This script validates and copies a pretrained model.tar.gz file from the input location
to the output location. It serves as a dummy training step that skips actual training 
and simply passes a pretrained model to downstream steps.
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model(input_path: Path) -> bool:
    """
    Validate the model file format and structure.
    
    Args:
        input_path: Path to the input model.tar.gz file
        
    Returns:
        True if validation passes, False otherwise
        
    Raises:
        ValueError: If the file format is incorrect
        Exception: For other validation errors
    """
    logger.info(f"Validating model file: {input_path}")
    
    # Check file extension
    if not input_path.suffix == '.tar.gz' and not str(input_path).endswith('.tar.gz'):
        raise ValueError(f"Expected a .tar.gz file, but got: {input_path} (ERROR_CODE: INVALID_FORMAT)")
    
    # Check if it's a valid tar archive
    if not tarfile.is_tarfile(input_path):
        raise ValueError(f"File is not a valid tar archive: {input_path} (ERROR_CODE: INVALID_ARCHIVE)")
    
    # Additional validation could be performed here:
    # - Check for required files within the archive
    # - Verify file sizes and structures
    # - Validate model format-specific details
    
    logger.info("Model validation successful")
    return True

def copy_model(input_path: str, output_dir: str) -> str:
    """
    Copy the pretrained model.tar.gz from input to output location.
    
    Args:
        input_path: Path to the input model.tar.gz file
        output_dir: Directory to copy the model to
        
    Returns:
        Path to the copied model file
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the copy operation fails
    """
    logger.info(f"Input model path: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Pretrained model file not found: {input_path} (ERROR_CODE: FILE_NOT_FOUND). Please check that the file exists at the specified location.")
    
    # Validate the model file
    validate_model(input_path)
    
    # Copy the file
    output_path = output_dir / "model.tar.gz"
    logger.info(f"Copying {input_path} to {output_path}")
    
    try:
        shutil.copy2(input_path, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to copy model file: {e} (ERROR_CODE: COPY_FAILED). Please check file permissions and disk space.")
    
    # Verify copy was successful
    if output_path.exists() and output_path.stat().st_size == input_path.stat().st_size:
        logger.info(f"Successfully copied model file ({output_path.stat().st_size} bytes)")
    else:
        raise RuntimeError(f"Model file copy failed or size mismatch (ERROR_CODE: SIZE_MISMATCH). Expected size: {input_path.stat().st_size}, actual size: {output_path.stat().st_size if output_path.exists() else 0}")
    
    return str(output_path)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dummy Training Processing Script")
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        required=True,
        help="Path to the pretrained model.tar.gz file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the copied model"
    )
    return parser.parse_args()

def main():
