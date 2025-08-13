# Script Testability Implementation Guide

This guide explains how to implement the testability improvements for scripts in `cursus/steps/scripts` as outlined in the [Script Testability Refactoring Design](../1_design/script_testability_refactoring.md). The goal is to make scripts more testable by separating execution environment concerns from core functionality.

## Why Improve Script Testability?

Docker containers provide a consistent execution environment, but they present challenges for testing:

1. Container startup has overhead
2. Setting up test data within containers is complex
3. Debugging container issues is difficult
4. Running comprehensive test suites in containers is slow
5. Local development and rapid iteration is hindered

By refactoring scripts to separate environment setup from business logic, we enable efficient testing both locally and within containers.

## Refactoring Pattern

Follow these steps to refactor a script for improved testability:

### Step 1: Analyze the Script

Before refactoring, understand what the script does:

- Identify all input paths
- Identify all output paths
- Identify all environment variables used
- Identify all command-line arguments

### Step 2: Refactor the Main Function

Convert the main function to accept parameters instead of accessing environment directly:

```python
# BEFORE
def main():
    parser = argparse.ArgumentParser()
    # ... parse args ...
    args = parser.parse_args()
    
    # Direct environment access
    id_field = os.environ.get("ID_FIELD", "id")
    
    # Direct path usage
    model_dir = args.model_dir
    # ...
```

```python
# AFTER
def main(input_paths, output_paths, environ_vars, job_args):
    """
    Main function for processing.
    
    Args:
        input_paths (dict): Dictionary mapping logical names to physical paths for inputs
        output_paths (dict): Dictionary mapping logical names to physical paths for outputs
        environ_vars (dict): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
    """
    # Use parameters instead of direct access
    model_dir = input_paths["model_dir"]
    id_field = environ_vars.get("ID_FIELD", "id")
    job_type = job_args.job_type
    # ...
```

### Step 3: Create an Entry Point

Set up an entry point that collects environment values and calls the main function:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_data_dir", type=str, required=True)
    parser.add_argument("--output_eval_dir", type=str, required=True)
    parser.add_argument("--output_metrics_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Set up container paths from command-line arguments
    input_paths = {
        "model_dir": args.model_dir,
        "eval_data_dir": args.eval_data_dir,
    }
    
    output_paths = {
        "output_eval_dir": args.output_eval_dir,
        "output_metrics_dir": args.output_metrics_dir,
    }
    
    # Collect environment variables
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
    }
    
    # Ensure output directories exist
    os.makedirs(output_paths["output_eval_dir"], exist_ok=True)
    os.makedirs(output_paths["output_metrics_dir"], exist_ok=True)
    
    # Call main function
    main(input_paths, output_paths, environ_vars, args)
```

### Step 4: Update Helper Functions

Update any internal functions to use passed parameters instead of accessing environment directly:

```python
# BEFORE
def process_data(df):
    label_col = os.environ.get("LABEL_FIELD", "label")
    return df[df[label_col] > 0.5]

# AFTER
def process_data(df, label_col="label"):
    return df[df[label_col] > 0.5]
```

## Container Path Handling

### Container Constants

For container paths, define constants at the top of the script:

```python
# Container path constants
CONTAINER_PATHS = {
    "PROCESSING_INPUT_BASE": "/opt/ml/processing/input",
    "PROCESSING_OUTPUT_BASE": "/opt/ml/processing/output",
    "MODEL_DIR": "/opt/ml/processing/input/model",
    "EVAL_DATA_DIR": "/opt/ml/processing/input/data",
    "OUTPUT_EVAL_DIR": "/opt/ml/processing/output/evaluation",
    "OUTPUT_METRICS_DIR": "/opt/ml/processing/output/metrics"
}
```

### Hybrid Mode Support

For scripts that need to run both in containers and locally:

```python
def is_running_in_container():
    """Detect if the script is running inside a container."""
    return os.path.exists("/.dockerenv") or os.environ.get("CONTAINER_MODE") == "true"

if __name__ == "__main__":
    in_container = is_running_in_container()
    
    if in_container:
        # Container mode - use container paths
        input_base = "/opt/ml/processing/input"
        output_base = "/opt/ml/processing/output"
    else:
        # Local mode - use local paths
        input_base = os.environ.get("LOCAL_INPUT_PATH", "./input")
        output_base = os.environ.get("LOCAL_OUTPUT_PATH", "./output")
    
    # Build paths based on execution environment
    input_paths = {
        "model_dir": os.path.join(input_base, "model"),
        "eval_data_dir": os.path.join(input_base, "evaluation")
    }
```

## Writing Unit Tests

With the refactored structure, writing tests becomes straightforward:

### Basic Test Structure

```python
def test_xgboost_model_evaluation():
    """Test the XGBoost model evaluation script."""
    
    # Set up test paths
    input_paths = {
        "model_dir": "test/resources/model",
        "eval_data_dir": "test/resources/eval_data",
    }
    
    output_paths = {
        "output_eval_dir": "test/output/eval",
        "output_metrics_dir": "test/output/metrics",
    }
    
    # Set up test environment variables
    environ_vars = {
        "ID_FIELD": "test_id",
        "LABEL_FIELD": "test_label",
    }
    
    # Create mock arguments
    args = argparse.Namespace()
    args.job_type = "testing"
    
    # Create output directories
    os.makedirs(output_paths["output_eval_dir"], exist_ok=True)
    os.makedirs(output_paths["output_metrics_dir"], exist_ok=True)
    
    # Call the function under test
    from src.cursus.steps.scripts.xgboost_model_evaluation import main
    main(input_paths, output_paths, environ_vars, args)
    
    # Assertions to verify expected outputs
    assert os.path.exists(os.path.join(output_paths["output_eval_dir"], "eval_predictions.csv"))
    metrics_path = os.path.join(output_paths["output_metrics_dir"], "metrics.json")
    assert os.path.exists(metrics_path)
    
    # Verify metrics content
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert "auc_roc" in metrics
    assert metrics["auc_roc"] > 0.5
```

### Using Mocks for External Dependencies

For scripts with external dependencies, use mocks:

```python
@patch("src.cursus.steps.scripts.xgboost_model_evaluation.xgb.Booster")
def test_xgboost_model_evaluation_with_mocks(mock_booster):
    """Test with mocked XGBoost booster."""
    
    # Set up mock booster
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.1, 0.8, 0.2, 0.9, 0.3])
    mock_booster.return_value = mock_model
    
    # Set up test data
    # ...
    
    # Call function under test
    main(input_paths, output_paths, environ_vars, args)
    
    # Verify mock was called
    mock_model.predict.assert_called_once()
```

## Integration with Script Contracts

The refactored structure aligns with our [Script Contract](script_contract.md) framework:

```python
# Example script contract
XGBOOST_EVAL_CONTRACT = ScriptContract(
    entry_point="xgboost_model_evaluation.py",
    expected_input_paths={
        "model_dir": "/opt/ml/processing/input/model",
        "eval_data_dir": "/opt/ml/processing/input/data"
    },
    expected_output_paths={
        "output_eval_dir": "/opt/ml/processing/output/evaluation",
        "output_metrics_dir": "/opt/ml/processing/output/metrics"
    },
    expected_arguments={
        "job-type": "training",
        "model-dir": "/opt/ml/processing/input/model",
        "eval-data-dir": "/opt/ml/processing/input/data",
        "output-eval-dir": "/opt/ml/processing/output/evaluation",
        "output-metrics-dir": "/opt/ml/processing/output/metrics"
    },
    required_env_vars=[
        "LABEL_FIELD"
    ],
    optional_env_vars={
        "ID_FIELD": "id"
    }
)
```

Make sure the input/output paths in your refactored script match those in the contract.

## Handling Container-Specific Features

### Error Handling

Implement robust error handling for container execution:

```python
if __name__ == "__main__":
    try:
        # Parse arguments, set up paths, etc.
        # ...
        
        # Call main function
        main(input_paths, output_paths, environ_vars, args)
        
        # Signal success
        success_path = os.path.join(output_paths["output_metrics_dir"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")
        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(output_paths["output_metrics_dir"], "_FAILURE")
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
```

### Container Health Checks

Add container health check support:

```python
def create_health_check_file(output_paths):
    """Create a health check file to signal container readiness."""
    health_path = os.path.join(output_paths.get("health_dir", "/tmp"), "health_check")
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path
```

## Full Example

See the [Script Testability Refactoring Design](../1_design/script_testability_refactoring.md) document for a complete example implementation of a refactored script.

## Refactoring Checklist

Use this checklist to ensure your refactoring is complete:

1. [ ] Main function accepts `input_paths`, `output_paths`, `environ_vars`, and `job_args`
2. [ ] All direct environment variable access is replaced with dictionary access
3. [ ] All direct path access is replaced with dictionary access
4. [ ] Entry point collects all required environment variables
5. [ ] Entry point sets up all required paths
6. [ ] Helper functions accept necessary parameters instead of accessing environment
7. [ ] Error handling is robust
8. [ ] The script behaves identically before and after refactoring
9. [ ] Unit tests are added for the refactored script
10. [ ] Documentation is updated to reflect the new structure

## References

- [Script Contract Developer Guide](script_contract.md)
- [Script Testability Refactoring Design](../1_design/script_testability_refactoring.md)
- [Script Contract Design](../1_design/script_contract.md)
