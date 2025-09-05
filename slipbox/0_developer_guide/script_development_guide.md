# Script Development Guide

This guide provides comprehensive instructions for developing SageMaker-compatible scripts that integrate seamlessly with the Cursus pipeline system. Scripts must follow standardized patterns for testability, SageMaker container compatibility, and alignment with script contracts.

## Overview

Pipeline scripts are the core processing logic that runs within SageMaker containers. They must be:

1. **Testable**: Follow the unified main function interface for local testing
2. **SageMaker Compatible**: Work with SageMaker's prebuilt containers and path conventions
3. **Contract Aligned**: Match the expectations defined in script contracts
4. **Workspace Aware**: Support both shared and isolated development workspaces

## Script Architecture Requirements

### 1. Unified Main Function Interface

All scripts must implement a standardized main function signature that separates environment concerns from business logic:

```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger=None
) -> Any:
    """
    Main processing function with standardized interface.
    
    Args:
        input_paths: Dictionary mapping logical names to physical input paths
        output_paths: Dictionary mapping logical names to physical output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments parsed by argparse
        logger: Optional logger function (defaults to print if None)
    
    Returns:
        Processing results (optional, for testing purposes)
    """
    # Your processing logic here
    pass
```

### 2. SageMaker Container Compatibility

Scripts must work with SageMaker's standard container paths and conventions:

#### Standard Container Paths

```python
# Container path constants - define these in your script
CONTAINER_PATHS = {
    # Processing containers
    "PROCESSING_INPUT_BASE": "/opt/ml/processing/input",
    "PROCESSING_OUTPUT_BASE": "/opt/ml/processing/output",
    
    # Training containers
    "TRAINING_INPUT_BASE": "/opt/ml/input/data",
    "TRAINING_MODEL_DIR": "/opt/ml/model",
    "TRAINING_OUTPUT_BASE": "/opt/ml/output/data",
    
    # Transform containers
    "TRANSFORM_INPUT_BASE": "/opt/ml/input/data",
    "TRANSFORM_OUTPUT_BASE": "/opt/ml/output",
    
    # Common paths
    "CONFIG_DIR": "/opt/ml/input/data/config",
    "HYPERPARAMETERS_FILE": "/opt/ml/input/data/config/hyperparameters.json"
}
```

#### Container Type Detection

```python
def detect_container_type() -> str:
    """Detect the type of SageMaker container we're running in."""
    if os.path.exists("/opt/ml/processing"):
        return "processing"
    elif os.path.exists("/opt/ml/model") and os.path.exists("/opt/ml/input/data"):
        return "training"
    elif os.path.exists("/opt/ml/input/data") and not os.path.exists("/opt/ml/model"):
        return "transform"
    else:
        return "unknown"
```

### 3. Script Contract Alignment

Scripts must align with their corresponding script contracts. The contract defines:

- **Entry Point**: Script filename
- **Expected Input Paths**: Logical names and container paths
- **Expected Output Paths**: Logical names and container paths
- **Arguments**: CLI arguments with hyphen-style names
- **Environment Variables**: Required and optional environment variables

#### Argument Naming Convention

**Contract uses CLI-style hyphens, scripts use Python-style underscores (standard argparse behavior):**

```python
# Contract Declaration (CLI convention)
"arguments": {
    "job-type": {"required": true},
    "model-dir": {"required": true},
    "output-eval-dir": {"required": true}
}

# Script Implementation (Python convention)
parser.add_argument("--job-type", required=True)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--output-eval-dir", required=True)

# Script Usage (automatic argparse conversion)
args.job_type  # argparse converts job-type → job_type
args.model_dir  # argparse converts model-dir → model_dir
args.output_eval_dir  # argparse converts output-eval-dir → output_eval_dir
```

## Script Templates

### Processing Script Template

```python
#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, Any

# Container path constants
CONTAINER_PATHS = {
    "PROCESSING_INPUT_BASE": "/opt/ml/processing/input",
    "PROCESSING_OUTPUT_BASE": "/opt/ml/processing/output",
    "DATA_INPUT": "/opt/ml/processing/input/data",
    "DATA_OUTPUT": "/opt/ml/processing/output"
}

def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    sys.stdout.flush()
    return logger

def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger=None
) -> Dict[str, Any]:
    """
    Main processing logic.
    
    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger function
    
    Returns:
        Processing results
    """
    log = logger or print
    
    # Extract parameters
    job_type = job_args.job_type
    input_data_dir = input_paths.get("data_input")
    output_data_dir = output_paths.get("data_output")
    
    # Extract environment variables
    label_field = environ_vars.get("LABEL_FIELD")
    
    log(f"Starting processing with job_type: {job_type}")
    log(f"Input directory: {input_data_dir}")
    log(f"Output directory: {output_data_dir}")
    log(f"Label field: {label_field}")
    
    # Create output directory
    Path(output_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Your processing logic here
    # ...
    
    log("Processing completed successfully")
    return {"status": "success"}

if __name__ == "__main__":
    try:
        # Set up logging
        logger = setup_logging()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Processing script")
        parser.add_argument("--job-type", type=str, required=True,
                          choices=["training", "validation", "testing", "calibration"])
        args = parser.parse_args()
        
        # Set up paths using container defaults
        input_paths = {
            "data_input": CONTAINER_PATHS["DATA_INPUT"]
        }
        
        output_paths = {
            "data_output": CONTAINER_PATHS["DATA_OUTPUT"]
        }
        
        # Collect environment variables
        environ_vars = {
            "LABEL_FIELD": os.environ.get("LABEL_FIELD"),
            # Add other environment variables as needed
        }
        
        # Validate required environment variables
        if not environ_vars["LABEL_FIELD"]:
            raise ValueError("LABEL_FIELD environment variable is required")
        
        # Log startup information
        logger.info("Starting processing script")
        logger.info(f"Job type: {args.job_type}")
        logger.info(f"Input paths: {input_paths}")
        logger.info(f"Output paths: {output_paths}")
        
        # Execute main processing
        result = main(input_paths, output_paths, environ_vars, args, logger.info)
        
        # Create success marker
        success_path = Path(output_paths["data_output"]) / "_SUCCESS"
        success_path.touch()
        logger.info(f"Created success marker: {success_path}")
        
        logger.info("Script completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Script failed with error: {e}")
        logging.error(traceback.format_exc())
        
        # Create failure marker
        try:
            failure_path = Path(output_paths.get("data_output", "/tmp")) / "_FAILURE"
            with open(failure_path, "w") as f:
                f.write(f"Error: {str(e)}")
            logging.error(f"Created failure marker: {failure_path}")
        except:
            pass
        
        sys.exit(1)
```

### Training Script Template

```python
#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, Any

# Container path constants
CONTAINER_PATHS = {
    "INPUT_DATA": "/opt/ml/input/data",
    "MODEL_DIR": "/opt/ml/model",
    "OUTPUT_DATA": "/opt/ml/output/data",
    "CONFIG_DIR": "/opt/ml/input/data/config",
    "HYPERPARAMETERS_FILE": "/opt/ml/input/data/config/hyperparameters.json"
}

def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    sys.stdout.flush()
    return logger

def load_hyperparameters(hparam_path: str) -> Dict[str, Any]:
    """Load and validate hyperparameters from JSON file."""
    try:
        with open(hparam_path, "r") as f:
            config = json.load(f)
        
        # Add validation logic here
        required_keys = ["label_name", "num_classes"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required hyperparameter: {key}")
        
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load hyperparameters from {hparam_path}: {e}")

def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger=None
) -> Dict[str, Any]:
    """
    Main training logic.
    
    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger function
    
    Returns:
        Training results
    """
    log = logger or print
    
    # Extract paths
    data_dir = input_paths["input_path"]
    model_dir = output_paths["model_output"]
    output_dir = output_paths["evaluation_output"]
    config_path = input_paths.get("hyperparameters_s3_uri", 
                                 os.path.join(data_dir, "config", "hyperparameters.json"))
    
    log("Starting training process")
    log(f"Data directory: {data_dir}")
    log(f"Model directory: {model_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Config path: {config_path}")
    
    # Load hyperparameters
    config = load_hyperparameters(config_path)
    log("Hyperparameters loaded successfully")
    
    # Create output directories
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Your training logic here
    # ...
    
    log("Training completed successfully")
    return {"status": "success", "model_path": model_dir}

if __name__ == "__main__":
    try:
        # Set up logging
        logger = setup_logging()
        
        # Training scripts typically don't take CLI arguments
        # All configuration comes from hyperparameters.json
        args = argparse.Namespace()
        
        # Set up paths using container defaults
        input_paths = {
            "input_path": CONTAINER_PATHS["INPUT_DATA"],
            "hyperparameters_s3_uri": CONTAINER_PATHS["CONFIG_DIR"]
        }
        
        output_paths = {
            "model_output": CONTAINER_PATHS["MODEL_DIR"],
            "evaluation_output": CONTAINER_PATHS["OUTPUT_DATA"]
        }
        
        # Collect environment variables
        environ_vars = {
            # Add environment variables as needed
        }
        
        # Log startup information
        logger.info("Starting training script")
        logger.info(f"Input paths: {input_paths}")
        logger.info(f"Output paths: {output_paths}")
        
        # Execute main training
        result = main(input_paths, output_paths, environ_vars, args, logger.info)
        
        logger.info("Training script completed successfully")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Training script failed with error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
```

## Best Practices

### 1. Error Handling and Logging

```python
def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility."""
    # Remove any existing handlers to avoid duplicates
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    sys.stdout.flush()
    
    return logger

# Use try-catch blocks with detailed error reporting
try:
    # Processing logic
    pass
except Exception as e:
    logger.error(f"Processing failed: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Create failure marker
    failure_path = Path(output_dir) / "_FAILURE"
    with open(failure_path, "w") as f:
        f.write(f"Error: {str(e)}")
    
    sys.exit(1)
```

### 2. Success/Failure Markers

```python
# Create success marker on completion
success_path = Path(output_dir) / "_SUCCESS"
success_path.touch()
logger.info(f"Created success marker: {success_path}")

# Create failure marker on error
failure_path = Path(output_dir) / "_FAILURE"
with open(failure_path, "w") as f:
    f.write(f"Error: {str(e)}")
```

### 3. Path Validation

```python
def validate_paths(input_paths: Dict[str, str], output_paths: Dict[str, str]) -> None:
    """Validate that required paths exist or can be created."""
    # Check input paths exist
    for logical_name, path in input_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path '{logical_name}' does not exist: {path}")
    
    # Create output directories
    for logical_name, path in output_paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {path}")
```

### 4. Environment Variable Validation

```python
def validate_environment_variables(environ_vars: Dict[str, str], required_vars: List[str]) -> None:
    """Validate that required environment variables are present."""
    missing_vars = []
    for var in required_vars:
        if not environ_vars.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
```

## Testing Your Scripts

### 1. Unit Testing

Create unit tests for your main function:

```python
import unittest
import tempfile
import shutil
from pathlib import Path
from your_script import main

class TestYourScript(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_main_function_success(self):
        """Test successful execution of main function."""
        # Set up test data
        test_data_file = self.input_dir / "test_data.csv"
        test_data_file.write_text("col1,col2,label\n1,2,0\n3,4,1\n")
        
        # Set up parameters
        input_paths = {"data_input": str(self.input_dir)}
        output_paths = {"data_output": str(self.output_dir)}
        environ_vars = {"LABEL_FIELD": "label"}
        args = argparse.Namespace(job_type="training")
        
        # Execute main function
        result = main(input_paths, output_paths, environ_vars, args)
        
        # Verify results
        self.assertEqual(result["status"], "success")
        self.assertTrue((self.output_dir / "_SUCCESS").exists())
    
    def test_main_function_missing_label_field(self):
        """Test error handling for missing label field."""
        input_paths = {"data_input": str(self.input_dir)}
        output_paths = {"data_output": str(self.output_dir)}
        environ_vars = {}  # Missing LABEL_FIELD
        args = argparse.Namespace(job_type="training")
        
        with self.assertRaises(ValueError):
            main(input_paths, output_paths, environ_vars, args)

if __name__ == "__main__":
    unittest.main()
```

### 2. Integration Testing

Test your script in a container-like environment:

```python
#!/usr/bin/env python3
"""
Integration test script that simulates SageMaker container environment.
"""
import os
import tempfile
import subprocess
import shutil
from pathlib import Path

def test_script_integration():
    """Test script in simulated container environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up container-like directory structure
        container_root = Path(temp_dir)
        input_dir = container_root / "opt" / "ml" / "processing" / "input" / "data"
        output_dir = container_root / "opt" / "ml" / "processing" / "output"
        
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        # Create test data
        test_data = input_dir / "test_data.csv"
        test_data.write_text("col1,col2,label\n1,2,0\n3,4,1\n")
        
        # Set up environment
        env = os.environ.copy()
        env["LABEL_FIELD"] = "label"
        
        # Run script
        result = subprocess.run([
            "python", "your_script.py",
            "--job-type", "training"
        ], env=env, cwd=str(container_root), capture_output=True, text=True)
        
        # Check results
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert (output_dir / "_SUCCESS").exists(), "Success marker not created"
        
        print("Integration test passed!")

if __name__ == "__main__":
    test_script_integration()
```

## Script Storage Locations

Scripts must be stored in one of two locations depending on your development approach:

### 1. Shared Workspace (Traditional)
```
src/cursus/steps/scripts/
├── __init__.py
├── your_script.py
├── another_script.py
└── ...
```

### 2. Isolated Development Workspace (Recommended)
```
development/projects/project_name/src/cursus_dev/steps/scripts/
├── __init__.py
├── your_script.py
├── another_script.py
└── ...
```

The UnifiedRegistryManager will automatically discover scripts in both locations, with workspace-specific scripts taking precedence over shared ones.

## SageMaker Container Requirements

### Processing Containers

- **Input Path**: `/opt/ml/processing/input/`
- **Output Path**: `/opt/ml/processing/output/`
- **Arguments**: Passed via command line
- **Environment Variables**: Set by the processing job

### Training Containers

- **Input Path**: `/opt/ml/input/data/`
- **Model Output**: `/opt/ml/model/`
- **Evaluation Output**: `/opt/ml/output/data/`
- **Hyperparameters**: `/opt/ml/input/data/config/hyperparameters.json`
- **No CLI Arguments**: All configuration via hyperparameters file

### Transform Containers

- **Input Path**: `/opt/ml/input/data/`
- **Output Path**: `/opt/ml/output/`
- **Model Path**: `/opt/ml/model/`

## Validation and Testing

Before deploying your script:

1. **Unit Test**: Test the main function with various inputs
2. **Integration Test**: Test in a simulated container environment
3. **Contract Validation**: Ensure alignment with script contract
4. **SageMaker Test**: Test in actual SageMaker environment

Use the [Validation Framework Guide](validation_framework_guide.md) for comprehensive validation tools.

## Common Pitfalls

1. **Hard-coded Paths**: Always use the path dictionaries, never hard-code paths
2. **Missing Error Handling**: Always include comprehensive error handling
3. **No Success Markers**: Always create success/failure markers
4. **Inconsistent Logging**: Use the standardized logging setup
5. **Environment Variable Access**: Use the environ_vars dictionary, not os.environ directly
6. **Argument Naming**: Remember CLI uses hyphens, Python uses underscores

## Related Documentation

- [Script Contract Development](script_contract.md) - Creating script contracts
- [Validation Framework Guide](validation_framework_guide.md) - Validating your scripts
- [Standardization Rules](standardization_rules.md) - Coding standards and conventions
- [Script Testability Implementation](script_testability_implementation.md) - Detailed testability patterns

## External References

- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/en/v2.23.2/overview.html) - Official SageMaker documentation
- [SageMaker Training Script Requirements](https://sagemaker.readthedocs.io/en/v2.23.2/overview.html#prepare-a-training-script) - Training script guidelines
- [SageMaker Processing Script Requirements](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html) - Processing script guidelines
