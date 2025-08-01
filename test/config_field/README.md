# Config Field Manager Backward Compatibility Tests

This directory contains tests that verify the backward compatibility between the old implementation of configuration field management in `src/pipeline_steps/utils.py` and the new implementation in `src/config_field_manager/`.

## Purpose

The purpose of these tests is to ensure that:

1. The new implementation produces identical results to the old implementation
2. Files saved with the old implementation can be loaded with the new implementation and vice versa
3. Both implementations categorize fields the same way (shared, processing_shared, processing_specific, specific)
4. Special fields are handled consistently across implementations

## Running the Tests

To run the backward compatibility tests:

```bash
python test/config_field_manager/run_compatibility_tests.py
```

This script will:

1. Run the unit tests verifying backward compatibility
2. Generate example output files from both implementations
3. Display a comparison of the output structures

## Test Structure

- `test_backward_compatibility.py`: Contains the unit tests for backward compatibility
- `run_compatibility_tests.py`: Runner script for the tests with detailed output
- `example_outputs/`: Directory containing example outputs from both implementations

## Expected Results

The tests should confirm that:

1. Both implementations produce configuration files with the same structure
2. Special fields like `hyperparameters` are placed in the correct sections
3. Loading files works identically across implementations
4. Round-trip conversions (save with old, load with new and vice versa) work correctly

## Troubleshooting

If the tests fail, it may indicate:

1. A change in field categorization logic between implementations
2. Special field handling differences
3. Serialization or deserialization differences

Check the test output for specific differences and examine the generated example files in `example_outputs/` to identify the source of the discrepancy.
