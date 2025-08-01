# Alignment Check for Risk Table Mapping Implementation

## Introduction

This document verifies alignment between the Risk Table Mapping components according to the alignment rules in the developer guide. Proper alignment ensures seamless integration between components and reduces errors in pipeline execution.

## Key Components

1. **Script**: `src/pipeline_scripts/risk_table_mapping.py`
2. **Contract**: `src/pipeline_script_contracts/risk_table_mapping_contract.py`
3. **Specifications**:
   - `src/pipeline_step_specs/risk_table_mapping_training_spec.py`
   - `src/pipeline_step_specs/risk_table_mapping_validation_spec.py`
   - `src/pipeline_step_specs/risk_table_mapping_testing_spec.py`
   - `src/pipeline_step_specs/risk_table_mapping_calibration_spec.py`
4. **Builder**: `src/pipeline_steps/builder_risk_table_mapping_step.py`
5. **Config**: `src/pipeline_steps/config_risk_table_mapping_step.py`

## Alignment Verification

### 1. Script-to-Contract Alignment

| Script Path | Contract Path | Status | Notes |
|-------------|---------------|--------|-------|
| `/opt/ml/processing/input/data` | `data_input` | ✓ Aligned | Used for input data loading |
| `/opt/ml/processing/input/config` | `hyperparameters_s3_uri` | ✓ Aligned | Used for loading hyperparameters.json |
| `/opt/ml/processing/input/risk_tables` | `risk_tables` | ✓ Aligned | Used for risk tables in non-training modes |
| `/opt/ml/processing/output` | `processed_data`, `risk_tables` | ✓ Aligned | Used for saving outputs |

**Constants**: Script uses `RISK_TABLE_FILENAME` and `HYPERPARAMS_FILENAME` for consistent naming, ensuring alignment with paths specified in the contract.

### 2. Contract-to-Specification Logical Name Alignment

| Contract Logical Name | Spec Logical Name | Status | Notes |
|------------------------|-------------------|--------|-------|
| `data_input` | `data_input` | ✓ Aligned | Used in all specs |
| `hyperparameters_s3_uri` | `hyperparameters_s3_uri` | ✓ Aligned | Changed from "config_input" for consistency |
| `risk_tables` | `risk_tables` | ✓ Aligned | Used in non-training specs |

### 3. Specification-to-Builder Alignment

The builder correctly uses the logical names from specifications for:

- **Inputs**: Properly maps logical names to container paths using contract
  ```python
  container_path = self.contract.expected_input_paths.get("hyperparameters_s3_uri", "/opt/ml/processing/input/config")
  ```

- **Outputs**: Correctly maps outputs using the contract
  ```python
  container_path = self.contract.expected_output_paths[logical_name]
  ```

- **Job Arguments**: Correctly passes job_type from config
  ```python
  job_type = self.config.job_type
  return ["--job_type", job_type]
  ```

### 4. Builder-to-Script Alignment

The builder correctly creates:

- **ProcessingInput** objects that match the contract's expected input paths
- **ProcessingOutput** objects that match the contract's expected output paths
- **Environment variables** expected by the script

### 5. Script-to-Builder Alignment

The script correctly:

- Accesses files at the exact paths defined in the contract and used by the builder
- Reads hyperparameters from `/opt/ml/processing/input/config/hyperparameters.json`
- Saves bin_mapping.pkl output at the expected location

## Special Attention: hyperparameters_s3_uri Handling

This alignment was carefully handled:

1. **Contract**: Logical name changed from "config_input" to "hyperparameters_s3_uri"
2. **Specs**: All specs updated to use "hyperparameters_s3_uri" with matching dependency type
3. **Builder**: `_get_inputs()` updated to use "hyperparameters_s3_uri" as input_name
4. **Script**: Still reads from the correct path but using constants for clarity

## Cross-Component Semantic Matching

### Input Semantic Keywords

The training specification includes enriched semantic keywords for inputs:
```python
semantic_keywords=["training", "train", "data", "input", "preprocessed", "tabular"]
```

The hyperparameters dependency includes:
```python
semantic_keywords=["config", "params", "hyperparameters", "settings", "hyperparams"]
```

### Output Semantic Keywords

Risk tables output includes:
```python
semantic_keywords=["risk_tables", "bin_mapping", "categorical_mappings", "model_artifacts"]
```

## Potential Compatibility Score Analysis

Using the dependency resolver algorithm weights:
- Type compatibility: 40% - Scores high (matching types)
- Data type compatibility: 20% - Scores high (matching S3Uri types)
- Semantic name matching: 25% - Scores high (rich semantic keywords)
- Additional bonuses: 15% - Scores high (aliases enhance matching)

## Conclusion

The Risk Table Mapping implementation demonstrates excellent alignment between all components. The logical name change from "config_input" to "hyperparameters_s3_uri" was properly propagated across all relevant files, ensuring consistent naming and behavior.

The use of constants in the script for filenames and the enhanced S3 path handling in the builder further strengthens the alignment, making the implementation robust against potential path inconsistencies.
