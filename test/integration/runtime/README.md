# XGBoost 3-Step Pipeline Runtime Testing Suite

This directory contains a comprehensive test suite for validating the Cursus pipeline runtime system using a real-world XGBoost machine learning pipeline.

## Overview

The test suite validates a 3-step XGBoost pipeline:
1. **XGBoost Training** - Train an XGBoost model on synthetic data
2. **XGBoost Model Evaluation** - Evaluate the trained model and generate metrics
3. **Model Calibration** - Calibrate model predictions for better probability estimates

**Pipeline Dependencies:**
```
XGBoost Training → XGBoost Model Eval → Model Calibration
```

## Test Suite Components

### 1. Setup and Data Preparation
**File:** `01_setup_and_data_preparation.ipynb`

**Purpose:** Environment setup and synthetic dataset generation
- Creates directory structure for the test suite
- Generates synthetic training data (800 samples, 10 features)
- Generates synthetic evaluation data (200 samples, 10 features)
- Saves datasets as CSV files with metadata
- Provides data exploration and validation

**Outputs:**
- `data/train_data.csv` - Training dataset
- `data/eval_data.csv` - Evaluation dataset
- `data/dataset_metadata.json` - Dataset metadata and statistics

### 2. Pipeline Configuration
**File:** `02_pipeline_configuration_complete.ipynb`

**Purpose:** Pipeline definition and step configuration
- Defines the 3-step pipeline with proper dependencies
- Creates comprehensive JSON configurations for each step
- Includes hyperparameters, input/output paths, and validation rules
- Validates configurations and creates combined pipeline config

**Outputs:**
- `configs/xgboost_training_config.json` - Training step configuration
- `configs/xgboost_eval_config.json` - Evaluation step configuration
- `configs/model_calibration_config.json` - Calibration step configuration
- `configs/pipeline_config.json` - Combined pipeline configuration

### 3. Individual Step Testing
**File:** `03_individual_step_testing.ipynb`

**Purpose:** Test each pipeline step individually in isolation
- Mock step tester implementation for isolated testing
- Individual step validation and error handling
- Output verification and data flow checks
- Comprehensive test reporting and metrics

**Key Features:**
- `MockStepTester` class for step-by-step validation
- Dependency validation between steps
- Error handling and failure reporting
- Execution time tracking
- Output file verification

**Outputs:**
- `outputs/workspace/` - Step execution workspace
- `outputs/results/individual_step_test_results.json` - Test results
- `outputs/logs/` - Execution logs

## Directory Structure

```
test/integration/runtime/
├── README.md                                    # This documentation
├── 01_setup_and_data_preparation.ipynb        # Data setup notebook
├── 02_pipeline_configuration_complete.ipynb   # Pipeline config notebook
├── 03_individual_step_testing.ipynb          # Individual step testing
├── step_testing_script.py                     # Python script version
├── script_to_notebook.py                      # Script-to-notebook converter
├── data/                                       # Generated datasets
│   ├── train_data.csv
│   ├── eval_data.csv
│   └── dataset_metadata.json
├── configs/                                    # Step configurations
│   ├── xgboost_training_config.json
│   ├── xgboost_eval_config.json
│   ├── model_calibration_config.json
│   └── pipeline_config.json
└── outputs/                                    # Test outputs
    ├── workspace/                              # Step execution files
    ├── results/                                # Test results
    └── logs/                                   # Execution logs
```

## Usage Instructions

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages: pandas, numpy, scikit-learn, xgboost
- Cursus framework (optional - mock implementations provided)

### Running the Test Suite

1. **Setup and Data Generation**
   ```bash
   jupyter notebook 01_setup_and_data_preparation.ipynb
   ```
   Run all cells to generate synthetic datasets and setup directory structure.

2. **Pipeline Configuration**
   ```bash
   jupyter notebook 02_pipeline_configuration_complete.ipynb
   ```
   Run all cells to create step configurations and pipeline definition.

3. **Individual Step Testing**
   ```bash
   jupyter notebook 03_individual_step_testing.ipynb
   ```
   Run all cells to test each pipeline step individually.

### Alternative: Python Script Execution

You can also run the individual step testing as a Python script:
```bash
cd test/integration/runtime
python step_testing_script.py
```

## Test Validation Features

### Mock Implementation
- **MockStepTester**: Simulates Cursus runtime components when not available
- **Synthetic Data**: Generates realistic ML datasets for testing
- **Error Simulation**: Tests error handling and recovery mechanisms

### Validation Checks
- **Data Validation**: Ensures required columns and data types
- **Dependency Validation**: Verifies step dependencies are satisfied
- **Output Validation**: Confirms expected output files are created
- **Configuration Validation**: Validates step configurations and parameters

### Reporting
- **Execution Metrics**: Tracks execution time for each step
- **Success/Failure Rates**: Provides comprehensive test statistics
- **Error Details**: Captures and reports detailed error information
- **File Verification**: Confirms all expected outputs are generated

## Integration with Cursus Framework

### Runtime Components Used
- `cursus.validation.runtime.jupyter.notebook_interface.NotebookInterface`
- `cursus.validation.runtime.core.data_flow_manager.DataFlowManager`
- `cursus.steps.registry.step_names.STEP_NAMES`

### Step Registry Integration
The test suite integrates with the Cursus step registry system:
- **Step Names**: Uses registry-defined step names
- **Step Specifications**: Validates against step specs in `cursus/steps/specs/`
- **Step Contracts**: Ensures compliance with contracts in `cursus/steps/contracts/`
- **Step Scripts**: Tests actual step scripts from `cursus/steps/scripts/`

## Troubleshooting

### Common Issues

1. **Import Errors**
   - The test suite includes mock implementations for when Cursus components are unavailable
   - Check that the Cursus framework is properly installed and accessible

2. **Missing Data Files**
   - Ensure you run notebooks in order: 01 → 02 → 03
   - Check that data generation completed successfully in notebook 01

3. **Configuration Errors**
   - Verify that pipeline configuration was created successfully in notebook 02
   - Check JSON configuration files for syntax errors

4. **Step Failures**
   - Review individual step error messages in the test output
   - Check that input files exist and have correct format
   - Verify step dependencies are satisfied

### Debug Mode
To enable detailed debugging, modify the notebooks to include:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extension Points

### Adding New Steps
To add additional pipeline steps:
1. Create step configuration in notebook 02
2. Add step testing method to `MockStepTester` class
3. Update the test execution flow in notebook 03

### Custom Validation
To add custom validation logic:
1. Extend the `MockStepTester` class with new validation methods
2. Add validation calls to the individual step test functions
3. Update the test summary generation to include new metrics

## Technical Notes

### Script-to-Notebook Conversion
The test suite uses a custom script-to-notebook converter (`script_to_notebook.py`) to avoid JSON truncation issues when creating large notebooks programmatically. This approach:
- Writes complex logic as Python scripts first
- Converts scripts to notebook format using AST parsing
- Preserves code structure and comments
- Avoids JSON size limitations

### Mock vs Real Implementation
The test suite is designed to work with both mock implementations (for testing) and real Cursus components (for production validation):
- Mock implementations simulate step execution for testing purposes
- Real implementations use actual Cursus runtime components
- The test framework automatically detects and uses available components

## Future Enhancements

1. **End-to-End Pipeline Testing**: Complete pipeline execution testing
2. **Performance Benchmarking**: Step execution performance analysis
3. **Error Injection Testing**: Systematic error condition testing
4. **Parallel Execution**: Multi-step parallel execution testing
5. **Resource Monitoring**: Memory and CPU usage tracking during execution

---

**Created:** August 2025  
**Version:** 1.0  
**Cursus Framework Integration:** Runtime Testing Suite
