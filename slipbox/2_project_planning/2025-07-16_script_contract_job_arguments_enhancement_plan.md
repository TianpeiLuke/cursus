# Script Contract Job Arguments Enhancement Plan

## Overview

This document outlines a comprehensive plan to enhance the Script Contract system with explicit job argument declarations. Currently, job arguments are implicitly handled in step builders without clear documentation or validation in the contract system. Adding explicit job argument definitions will improve alignment with the project's "Explicit over Implicit" design principle.

## Current State Analysis

1. **Implicit Argument Handling**: 
   - Step builders contain `_get_job_arguments()` methods that construct arguments in isolation
   - No central declaration of what arguments a script expects
   - No validation to ensure builders provide correct arguments to scripts
   - No documentation in contracts for required script arguments

2. **Problems with Current Approach**:
   - Hidden coupling between builders and scripts
   - Lack of validation for argument consistency
   - Documentation gap for script requirements
   - Maintenance risk when script arguments change

3. **Examples from Codebase**:
   - `MIMSPayloadStepBuilder._get_job_arguments()` returns default arguments if none in config
   - `TabularPreprocessingStepBuilder._get_job_arguments()` determines args based on job type
   - `DummyTrainingStepBuilder` has hardcoded arguments in `create_step()`

## Enhancement Plan

### 1. Base ScriptContract Class Modifications

Add an `expected_arguments` field to the `ScriptContract` class in `src/pipeline_script_contracts/base_script_contract.py`:

```python
class ScriptContract(BaseModel):
    """
    Script execution contract that defines explicit I/O, environment requirements, and CLI arguments
    """
    entry_point: str = Field(..., description="Script entry point filename")
    expected_input_paths: Dict[str, str] = Field(..., description="Mapping of logical names to expected input paths")
    expected_output_paths: Dict[str, str] = Field(..., description="Mapping of logical names to expected output paths")
    required_env_vars: List[str] = Field(..., description="List of required environment variables")
    optional_env_vars: Dict[str, str] = Field(default_factory=dict, description="Optional environment variables with defaults")
    expected_arguments: Dict[str, str] = Field(default_factory=dict, 
                                              description="Mapping of argument names to container paths or values")
    framework_requirements: Dict[str, str] = Field(default_factory=dict, description="Framework version requirements")
    description: str = Field(default="", description="Human-readable description of the script")
```

Add validation for the new field:

```python
@field_validator('expected_arguments')
@classmethod
def validate_arguments(cls, v: Dict[str, str]) -> Dict[str, str]:
    """Validate argument names follow CLI conventions (kebab-case)"""
    for arg_name in v.keys():
        if not all(c.isalnum() or c == '-' for c in arg_name):
            raise ValueError(f'Argument name contains invalid characters: {arg_name}')
        if not arg_name.lower() == arg_name:
            raise ValueError(f'Argument name should be lowercase: {arg_name}')
    return v
```

### 2. ScriptAnalyzer Enhancements

Extend the `ScriptAnalyzer` class to analyze script files for argument parsing:

```python
def get_argument_usage(self) -> Set[str]:
    """Extract command-line arguments used by the script"""
    if self._arguments is None:
        self._arguments = set()
        
        # Look for argparse patterns
        for node in ast.walk(self.ast_tree):
            # Look for parser.add_argument calls
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'add_argument'):
                
                # Check first argument for the argument name
                if node.args and (isinstance(node.args[0], ast.Str) or 
                                 (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str))):
                    arg_name = node.args[0].s if isinstance(node.args[0], ast.Str) else node.args[0].value
                    # Strip leading dashes
                    if arg_name.startswith('--'):
                        self._arguments.add(arg_name[2:])
                    elif arg_name.startswith('-'):
                        self._arguments.add(arg_name[1:])
        
    return self._arguments
```

Update the `_validate_against_analysis` method to check for arguments:

```python
def _validate_against_analysis(self, analyzer: 'ScriptAnalyzer') -> ValidationResult:
    # Existing validation...
    
    # Validate arguments
    script_args = analyzer.get_argument_usage()
    for arg_name in self.expected_arguments.keys():
        if arg_name not in script_args:
            warnings.append(f"Script doesn't seem to handle expected argument: --{arg_name}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

### 3. StepBuilderBase Updates

Add a generic `_get_job_arguments` method to `StepBuilderBase` in `src/pipeline_steps/builder_step_base.py`:

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs command-line arguments for the script based on script contract.
    If no arguments are defined in the contract, returns None (not an empty list).
    
    Returns:
        List of string arguments to pass to the script, or None if no arguments
    """
    if not hasattr(self, 'contract') or not self.contract:
        self.log_warning("No contract available for argument generation")
        return None
        
    # If contract has no expected arguments, return None
    if not hasattr(self.contract, 'expected_arguments') or not self.contract.expected_arguments:
        return None
        
    args = []
    
    # Add each expected argument with its value
    for arg_name, arg_value in self.contract.expected_arguments.items():
        args.extend([f"--{arg_name}", arg_value])
    
    # If we have arguments to return
    if args:
        self.log_info("Generated job arguments from contract: %s", args)
        return args
    
    # If we end up with an empty list, return None instead
    return None
```

### 4. Update Key Script Contracts

#### 4.1 DummyTrainingContract

Update `src/pipeline_script_contracts/dummy_training_contract.py`:

```python
DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        "pretrained_model_path": "/opt/ml/processing/input/model/model.tar.gz",
        "hyperparameters_s3_uri": "/opt/ml/processing/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"
    },
    expected_arguments={
        "pretrained-model-path": "/opt/ml/processing/input/model/model.tar.gz",
        "hyperparameters-s3-uri": "/opt/ml/processing/input/config/hyperparameters.json",
        "output-dir": "/opt/ml/processing/output/model"
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "pathlib": ">=1.0.0"
    },
    description="Contract for dummy training step that processes a pretrained model.tar.gz by unpacking it, "
                "adding a hyperparameters.json file inside, and repacking it for downstream steps"
)
```

#### 4.2 MIMSPackageContract

Update `src/pipeline_script_contracts/mims_package_contract.py`:

```python
MIMS_PACKAGE_CONTRACT = ScriptContract(
    # Existing fields...
    expected_arguments={
        "mode": "standard"
    },
    # Rest of existing fields...
)
```

#### 4.3 MIMSPayloadContract

Update `src/pipeline_script_contracts/mims_payload_contract.py`:

```python
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    # Existing fields...
    expected_arguments={
        "mode": "standard"
    },
    # Rest of existing fields...
)
```

#### 4.4 Model Evaluation Contract

Update `src/pipeline_script_contracts/model_evaluation_contract.py`:

```python
MODEL_EVALUATION_CONTRACT = ScriptContract(
    # Existing fields...
    expected_arguments={
        "job-type": "evaluation",
        "metrics-format": "detailed"  # Controls the level of metrics detail in logs
    },
    # Rest of existing fields...
)
```

### 5. Update Step Builders

#### 5.1 DummyTrainingStepBuilder

Update `src/pipeline_steps/builder_dummy_training.py` to use the contract-based arguments:

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs job arguments for dummy training script based on contract.
    
    Returns:
        List of command-line arguments or None
    """
    # Use base implementation that reads from contract
    return super()._get_job_arguments()
```

Update the `create_step` method to use this method instead of hardcoded arguments.

#### 5.2 MIMSPayloadStepBuilder

Refactor `src/pipeline_steps/builder_mims_payload_step.py` to first check the contract:

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs the list of command-line arguments to be passed to the processing script.

    Returns:
        List of string arguments or None
    """
    # If there are custom script arguments in the config, use those
    if hasattr(self.config, 'processing_script_arguments') and self.config.processing_script_arguments:
        return self.config.processing_script_arguments
    
    # Try to get arguments from contract
    contract_args = super()._get_job_arguments()
    if contract_args is not None:
        return contract_args
            
    # Return a standard argument to ensure we don't return an empty list
    self.log_info("Using default arguments for payload generation")
    return ["--mode", "standard"]
```

#### 5.3 ModelEvaluationStepBuilder

Update `src/pipeline_steps/builder_model_evaluation_step.py` to support metrics formatting:

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs job arguments for model evaluation script based on contract.
    
    Returns:
        List of command-line arguments or None
    """
    # Get base arguments from contract
    args = super()._get_job_arguments() or []
    
    # Allow config to override metrics format if specified
    if hasattr(self.config, 'metrics_format'):
        # Find and replace the metrics-format argument if it exists
        for i in range(len(args) - 1):
            if args[i] == '--metrics-format':
                args[i+1] = self.config.metrics_format
                break
        else:
            # If not found, add it
            args.extend(['--metrics-format', self.config.metrics_format])
            
    # If we have job type in config, make sure it's in the arguments
    if hasattr(self.config, 'job_type') and self.config.job_type:
        # Find and replace job-type if it exists
        for i in range(len(args) - 1):
            if args[i] == '--job-type':
                args[i+1] = self.config.job_type
                break
        else:
            # If not found, add it
            args.extend(['--job-type', self.config.job_type])
            
    return args if args else None
```

### 6. Script Updates for Model Evaluation

Update the model evaluation script (`src/pipeline_scripts/model_evaluation_xgb.py`) to support the metrics format argument:

```python
def main():
    """
    Main entry point for XGBoost model evaluation script.
    Loads model and data, runs evaluation, and saves results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-type", type=str, required=True)
    parser.add_argument("--metrics-format", type=str, choices=["basic", "detailed", "minimal"], 
                      default="detailed", help="Level of metrics detail in logs")
    args = parser.parse_args()

    # Get and validate contract
    contract = get_script_contract()
    
    # Use contract enforcement context manager
    with ContractEnforcer(contract) as enforcer:
        # Access validated environment variables (contract ensures these exist)
        ID_FIELD = os.environ["ID_FIELD"]
        LABEL_FIELD = os.environ["LABEL_FIELD"]

        # Use contract paths instead of hardcoded paths
        model_dir = enforcer.get_input_path('model_input')
        eval_data_dir = enforcer.get_input_path('eval_data_input')
        output_eval_dir = enforcer.get_output_path('eval_output')
        output_metrics_dir = enforcer.get_output_path('metrics_output')

        logger.info("Starting model evaluation script with metrics format: %s", args.metrics_format)
        # Set the metrics format for the current run
        METRICS_FORMAT = args.metrics_format
        
        # Rest of implementation...
```

Then modify the metrics functions to respect the chosen format:

```python
def log_metrics_summary(metrics, is_binary=True, format_level="detailed"):
    """
    Log a nicely formatted summary of metrics for easy visibility in logs.
    
    Args:
        metrics: Dictionary of metrics to log
        is_binary: Whether these are binary classification metrics
        format_level: Level of detail ("basic", "detailed", "minimal")
    """
    if format_level == "minimal":
        # Just log the key metrics with minimal formatting
        if is_binary:
            logger.info(f"METRICS: AUC={metrics.get('auc_roc', 'N/A'):.4f}, AP={metrics.get('average_precision', 'N/A'):.4f}, F1={metrics.get('f1_score', 'N/A'):.4f}")
        else:
            logger.info(f"METRICS: AUC-macro={metrics.get('auc_roc_macro', 'N/A'):.4f}, AUC-micro={metrics.get('auc_roc_micro', 'N/A'):.4f}, F1-macro={metrics.get('f1_score_macro', 'N/A'):.4f}")
        return
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 80)
    logger.info(f"METRICS SUMMARY - {timestamp}")
    logger.info("=" * 80)
    
    if format_level == "basic":
        # Show only key metrics with formatting
        logger.info("KEY METRICS")
        if is_binary:
            logger.info(f"METRIC: AUC-ROC               = {metrics.get('auc_roc', 'N/A'):.4f}")
            logger.info(f"METRIC: Average Precision     = {metrics.get('average_precision', 'N/A'):.4f}")
            logger.info(f"METRIC: F1 Score              = {metrics.get('f1_score', 'N/A'):.4f}")
        else:
            logger.info(f"METRIC: Macro AUC-ROC         = {metrics.get('auc_roc_macro', 'N/A'):.4f}")
            logger.info(f"METRIC: Micro AUC-ROC         = {metrics.get('auc_roc_micro', 'N/A'):.4f}")
            logger.info(f"METRIC: Macro F1              = {metrics.get('f1_score_macro', 'N/A'):.4f}")
            logger.info(f"METRIC: Micro F1              = {metrics.get('f1_score_micro', 'N/A'):.4f}")
        logger.info("=" * 80)
        return
    
    # Detailed format - logs all metrics with full formatting
    # Log each metric with a consistent format
    for name, value in metrics.items():
        # Format numeric values to 4 decimal places
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        # Add a special prefix for easy searching in logs
        logger.info(f"METRIC: {name.ljust(25)} = {formatted_value}")
    
    # Highlight key metrics based on task type
    logger.info("=" * 80)
    logger.info("KEY PERFORMANCE METRICS")
    logger.info("=" * 80)
    
    if is_binary:
        logger.info(f"METRIC_KEY: AUC-ROC               = {metrics.get('auc_roc', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Average Precision     = {metrics.get('average_precision', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: F1 Score              = {metrics.get('f1_score', 'N/A'):.4f}")
    else:
        logger.info(f"METRIC_KEY: Macro AUC-ROC         = {metrics.get('auc_roc_macro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Micro AUC-ROC         = {metrics.get('auc_roc_micro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Macro Average Precision = {metrics.get('average_precision_macro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Macro F1              = {metrics.get('f1_score_macro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Micro F1              = {metrics.get('f1_score_micro', 'N/A'):.4f}")
    
    logger.info("=" * 80)
```

### 7. Validation Tools

Extend `tools/validate_contracts.py` to validate argument consistency:

```python
def validate_script_arguments(contract_path, script_path):
    """
    Validate that script contract arguments are consistent with script implementation
    
    Args:
        contract_path: Path to the contract module
        script_path: Path to the script file
    
    Returns:
        List of issues found
    """
    issues = []
    
    # Import contract
    contract_module = importlib.import_module(contract_path)
    contract_vars = [v for v in dir(contract_module) if v.endswith('_CONTRACT')]
    
    for var_name in contract_vars:
        contract = getattr(contract_module, var_name)
        
        if hasattr(contract, 'expected_arguments'):
            # Get script arguments using ScriptAnalyzer
            analyzer = ScriptAnalyzer(script_path)
            script_args = analyzer.get_argument_usage()
            
            # Check all expected arguments are used in script
            for arg_name in contract.expected_arguments.keys():
                if arg_name not in script_args:
                    issues.append(f"Contract {var_name} expects argument '{arg_name}' not found in script {script_path}")
    
    return issues
```

Add a specialized validation tool for model evaluation:

```python
def validate_model_evaluation_script(script_path):
    """
    Validate that model evaluation script properly handles metrics format
    
    Args:
        script_path: Path to the model evaluation script
    
    Returns:
        List of issues found
    """
    issues = []
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for metrics format argument parsing
    if '--metrics-format' not in content:
        issues.append(f"Script {script_path} doesn't parse --metrics-format argument")
    
    # Check for format handling in log_metrics_summary
    if 'format_level=' not in content or 'format_level="detailed"' not in content:
        issues.append(f"Script {script_path} doesn't handle format_level parameter in log_metrics_summary")
    
    # Check for different format implementations
    if not all(x in content for x in ["minimal", "basic", "detailed"]):
        issues.append(f"Script {script_path} doesn't implement all metrics formats (minimal, basic, detailed)")
    
    return issues
```

### 8. Migration Plan

#### Phase 1: Framework Update (Week 1)
1. Add `expected_arguments` field to ScriptContract
2. Update ScriptAnalyzer for argument extraction
3. Add `_get_job_arguments` method to StepBuilderBase
4. Add validation tools
5. Write unit tests for new features

#### Phase 2: High Priority Contracts (Week 2)
1. Update DummyTrainingContract
2. Update MIMSPackageContract
3. Update MIMSPayloadContract
4. Update ModelEvaluationContract
5. Update key processing contracts

#### Phase 3: High Priority Builders (Week 2-3)
1. Update DummyTrainingStepBuilder
2. Update MIMSPackagingStepBuilder
3. Update MIMSPayloadStepBuilder
4. Update ModelEvaluationStepBuilder
5. Update key processing builders

#### Phase 4: Validation & Integration Testing (Week 3)
1. Run validation tools on all updated contracts
2. Fix any issues discovered
3. Run integration tests with sample pipelines

#### Phase 5: Remaining Contracts & Builders (Week 4)
1. Update remaining script contracts
2. Update corresponding step builders
3. Final validation and testing

### 9. Documentation Updates

#### 9.1 Design Principles

Update `slipbox/developer_guide/design_principles.md` to include the argument contract approach:

```markdown
### Explicit Argument Declaration

Script contracts should explicitly declare expected command-line arguments:

- **Arguments in Contract**: Declare expected arguments in the contract
- **Standard Format**: Use kebab-case for argument names
- **Value Sources**: Arguments values should reference paths in contract
- **Validation**: Arguments should be validated against script implementation

Example:
```python
ScriptContract(
    # Other fields...
    expected_arguments={
        "input-data-path": "/opt/ml/processing/input/data",
        "output-dir": "/opt/ml/processing/output/results",
        "mode": "standard"
    }
)
```
```

#### 9.2 Best Practices

Add a section on argument handling to `slipbox/developer_guide/best_practices.md`:

```markdown
### Command-line Argument Handling

#### In Script Contracts

1. **Declare all arguments**: Include all expected arguments in the `expected_arguments` field
2. **Use kebab-case**: Argument names should use kebab-case (e.g., `input-data-path`)
3. **Reference contract paths**: Use container paths defined in the contract for path arguments
4. **Document defaults**: If arguments have default values, document them in the description

#### In Step Builders

1. **Use contract arguments**: Get arguments from contract using `super()._get_job_arguments()`
2. **Handle config overrides**: Allow config to override contract arguments when needed
3. **Return None for no args**: Return None, not an empty list, when no arguments are needed
4. **Log generated arguments**: Always log the arguments being used for debugging
```

#### 9.3 Creation Process

Update `slipbox/developer_guide/creation_process.md` to include argument declaration:

```markdown
### 5. Declare Script Arguments

When defining a script contract, declare all expected command-line arguments:

```python
ScriptContract(
    # Other fields...
    expected_arguments={
        "arg-name": "value or path",
        # Other arguments...
    }
)
```

The arguments will be automatically generated by the step builder when creating the step.
```

### 10. Model Evaluation Script Specific Guidelines

Add a section on metrics formatting to the documentation:

```markdown
### Model Evaluation Metrics Formatting

The model evaluation scripts support configurable metrics formatting via the `--metrics-format` argument:

- **detailed**: Full metrics with formatted headers and all available metrics (default)
- **basic**: Only key metrics with minimal formatting
- **minimal**: Single-line summary of key metrics for compact logs

This can be configured in two ways:

1. In the contract's expected arguments:
   ```python
   expected_arguments={
       "metrics-format": "detailed"
   }
   ```

2. In the configuration object:
   ```python
   config = ModelEvaluationConfig(
       metrics_format="basic",
       # Other configuration...
   )
   ```

The configuration value will override the contract default if both are provided.
```

### 11. Backward Compatibility

To ensure backward compatibility:

1. Make `expected_arguments` field optional with empty dict default
2. Implement StepBuilderBase._get_job_arguments() to handle the case of no arguments in contract
3. Allow existing custom _get_job_arguments() methods to override the base implementation
4. Update each contract/builder pair incrementally, with thorough testing between updates
5. Add default values in argparse definitions for new arguments like metrics-format

### 12. Testing Strategy

1. **Unit Tests**:
   - Test argument extraction in ScriptAnalyzer
   - Test argument validation in ScriptContract
   - Test argument generation in StepBuilderBase
   - Test metrics formatting in model evaluation scripts

2. **Integration Tests**:
   - Test a sample pipeline with updated contracts and builders
   - Verify arguments are correctly passed to scripts
   - Check processing job logs to confirm correct arguments
   - Test different metrics format options and verify log output

3. **Validation Tests**:
   - Run validation tools on all contracts
   - Verify script-to-contract argument alignment

## Conclusion

This enhancement will make script arguments explicit in the contract system, improving documentation, validation, and alignment with design principles. The implementation approach ensures backward compatibility while providing a path forward for more explicit and maintainable code.

By following this plan, we will achieve:
1. Better documentation of script requirements
2. Improved validation of script-builder alignment
3. Reduced maintenance burden when arguments change
4. More consistent argument handling across the codebase
5. Configurable metrics formatting in model evaluation scripts

## References

- [Design Principles](../developer_guide/design_principles.md)
- [Best Practices](../developer_guide/best_practices.md)
- [Validation Checklist](../developer_guide/validation_checklist.md)
- [Script Contract Guide](../developer_guide/script_contract.md)
