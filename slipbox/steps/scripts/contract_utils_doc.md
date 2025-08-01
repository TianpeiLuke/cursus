# Contract Utilities Module Documentation

## Overview
The `contract_utils.py` module provides a robust framework for contract validation and enforcement within SageMaker pipeline scripts. It ensures alignment between step specifications, script contracts, and actual implementations by providing helpers to validate the execution environment, manage input/output paths, and enforce contract requirements.

## Key Components

### Functions

#### Environment Validation
- **validate_contract_environment(contract)**: Validates that the SageMaker execution environment matches contract expectations, including required environment variables, optional environment variables with defaults, and checking input/output path availability.

#### Path Management
- **get_contract_paths(contract)**: Returns a dictionary containing all input and output paths defined in the contract.
- **get_input_path(contract, logical_name)**: Retrieves a specific input path by its logical name.
- **get_output_path(contract, logical_name)**: Retrieves a specific output path by its logical name and ensures the directory exists.
- **find_files_in_input(contract, logical_name, pattern)**: Finds files matching a pattern in an input directory.
- **create_output_file_path(contract, logical_name, filename)**: Creates a full file path within an output directory.

#### Validation Utilities
- **validate_required_files(contract, required_files)**: Ensures that required files exist in input directories.
- **validate_framework_requirements(contract)**: Validates that required frameworks (like pandas, numpy, etc.) are available.
- **log_contract_summary(contract)**: Logs a comprehensive summary of the contract for debugging purposes.

### ContractEnforcer Class

A context manager that provides a streamlined way to enforce contracts in SageMaker scripts:

```python
with ContractEnforcer(contract, required_files) as enforcer:
    # Script logic here
    input_path = enforcer.get_input_path('data_input')
    output_path = enforcer.get_output_path('processed_output')
```

The ContractEnforcer handles:
- Logging contract details
- Validating the environment
- Validating framework requirements
- Checking required files
- Providing convenient methods for path management

## Usage Example

```python
from src.pipeline_script_contracts import my_script_contract
from src.pipeline_scripts.contract_utils import ContractEnforcer

# Define required files (optional)
required_files = {
    'data_input': ['training.csv', 'validation.csv']
}

# Use the ContractEnforcer context manager
with ContractEnforcer(my_script_contract, required_files) as enforcer:
    # Get paths from the contract
    training_data_path = enforcer.get_input_path('data_input')
    model_output_path = enforcer.get_output_path('model_output')
    
    # Find specific files
    config_files = enforcer.find_files_in_input('config_input', '*.json')
    
    # Create output file paths
    model_file_path = enforcer.create_output_file_path('model_output', 'model.joblib')
    
    # Your script logic here...
```

## Error Handling

The module raises informative exceptions when contract validation fails:
- `RuntimeError`: When contract validation fails (missing env vars, paths, files, etc.)
- `ValueError`: When requesting non-existent input/output paths

All validation functions include detailed logging to help diagnose issues.
