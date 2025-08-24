# XGBoost 3-Step Pipeline Runtime Test

This directory contains a comprehensive test case for the Cursus Pipeline Runtime Testing System, specifically designed to test a 3-step XGBoost pipeline.

## Overview

The test validates a complete machine learning pipeline with the following steps:
1. **XGBoost Training** - Trains an XGBoost model on synthetic data
2. **XGBoost Model Evaluation** - Evaluates the trained model performance
3. **Model Calibration** - Calibrates the model for better probability estimates

The pipeline follows the DAG: `XGBoost Training → XGBoost Model Eval → Model Calibration`

## Files Structure

```
test/integration/runtime/
├── README.md                                    # This documentation
├── test_xgboost_3_step_pipeline.ipynb         # Main test notebook
├── data/                                       # Generated test datasets
├── configs/                                    # Configuration files
└── outputs/                                    # Test execution outputs
    ├── workspace/                              # Step workspaces
    ├── logs/                                   # Execution logs
    └── results/                                # Test results and reports
```

## Test Notebook Sections

The Jupyter notebook (`test_xgboost_3_step_pipeline.ipynb`) contains 8 comprehensive sections:

### Section 1: Setup and Imports
- Configures the test environment
- Imports required Cursus components
- Sets up logging and path configuration

### Section 2: Pipeline Definition and Configuration
- Defines the 3-step pipeline structure
- Configures step dependencies and workspace directories
- Sets up the test environment

### Section 3: Test Data Preparation
- Generates synthetic training and evaluation datasets
- Creates realistic feature matrices and target variables
- Saves datasets for pipeline consumption

### Section 4: Individual Step Testing
- Tests each pipeline step in isolation
- Validates step execution, environment setup, and output generation
- Tracks individual step performance metrics

### Section 5: End-to-End Pipeline Testing
- Executes the complete pipeline from start to finish
- Validates data flow between steps
- Tracks pipeline-level performance and status

### Section 6: Performance Analysis and Visualization
- Generates performance metrics and visualizations
- Creates charts for execution time, memory usage, and success rates
- Provides comprehensive performance analysis

### Section 7: Error Handling and Edge Cases
- Tests various error scenarios (missing data, invalid dependencies, etc.)
- Validates edge cases (empty datasets, single samples, uniform targets)
- Ensures robust error handling

### Section 8: Results Summary and Reporting
- Generates comprehensive test reports
- Provides final summary of all test results
- Creates detailed workspace structure documentation

## How to Run the Test

### Prerequisites
1. Ensure you have the Cursus package installed and configured
2. Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`
3. Jupyter notebook environment

### Execution Steps
1. Navigate to the test directory:
   ```bash
   cd test/integration/runtime/
   ```

2. Launch Jupyter notebook:
   ```bash
   jupyter notebook test_xgboost_3_step_pipeline.ipynb
   ```

3. Execute all cells in sequence (or use "Run All" from the Cell menu)

### Expected Outputs
- **Synthetic datasets** in the `data/` directory
- **Step workspaces** with individual step outputs
- **Performance visualizations** showing execution metrics
- **Comprehensive test report** in JSON format
- **Final summary** with all test results

## Test Validation Points

The test validates the following aspects of the Pipeline Runtime Testing System:

### ✅ Core Functionality
- Pipeline script executor initialization and configuration
- Notebook interface integration
- Data flow manager functionality
- Step registry integration

### ✅ Pipeline Execution
- Individual step execution and validation
- End-to-end pipeline orchestration
- Data flow between pipeline steps
- Environment variable management

### ✅ Error Handling
- Missing input data scenarios
- Invalid step dependencies
- Corrupted data format handling
- Resource constraint simulation

### ✅ Performance Monitoring
- Execution time tracking
- Memory usage monitoring
- Success rate calculation
- Data flow transition tracking

### ✅ Reporting and Visualization
- Performance metrics generation
- Visual analysis charts
- Comprehensive test reporting
- Workspace structure documentation

## Integration with Cursus Components

This test case integrates with the following Cursus components:

- **`cursus.validation.runtime.core.pipeline_script_executor`** - Core pipeline execution engine
- **`cursus.validation.runtime.jupyter.notebook_interface`** - Jupyter integration interface
- **`cursus.validation.runtime.data.data_flow_manager`** - Data flow management
- **`cursus.steps.registry.step_names`** - Step name registry for XGBoost components

## Customization

To adapt this test for different pipeline configurations:

1. **Modify Step Configuration**: Update the `pipeline_config` in Section 2
2. **Change Test Data**: Modify data generation in Section 3
3. **Add New Steps**: Extend the step testing functions in Section 4
4. **Custom Metrics**: Add additional performance metrics in Section 6

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure Cursus package is properly installed and in Python path
- **Missing Dependencies**: Install required packages (`pip install pandas numpy matplotlib seaborn`)
- **Permission Errors**: Ensure write permissions for the test workspace directory

### Debug Mode
To enable debug logging, modify the logging configuration in Section 1:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Expected Results

A successful test run should show:
- ✅ All 3 pipeline steps executing successfully
- ✅ End-to-end pipeline completion with SUCCESS status
- ✅ Performance visualizations displaying execution metrics
- ✅ Error scenarios handled appropriately
- ✅ Comprehensive test report generated

## Contributing

When extending this test case:
1. Follow the existing section structure
2. Add appropriate error handling
3. Include performance tracking
4. Update this README with new features
5. Ensure backward compatibility with existing test structure

---

This test case serves as both a validation tool for the Cursus Pipeline Runtime Testing System and a reference implementation for creating additional pipeline tests.
