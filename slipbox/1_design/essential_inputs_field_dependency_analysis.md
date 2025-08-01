# XGBoost Evaluation Notebook Field Dependency Analysis

This document provides a comprehensive analysis of the fields required in the `template_config_xgb_eval_v2.ipynb` notebook, identifying which fields are essential for user input, which ones are system inputs with fixed values, and which ones can be derived from other fields.

## Overview of Configuration Structure

The notebook creates configuration for a complete XGBoost evaluation pipeline with these steps:

1. Base Config (shared across all steps)
2. Cradle Data Loading (training and calibration)
3. Processing Config (shared across processing steps)
4. Tabular Preprocessing (training and calibration)
5. XGBoost Training
6. Model Calibration
7. Model Evaluation
8. Packaging
9. MIMS Model Registration
10. Payload Sample Generation

## Three-Tier Input Categorization

Based on detailed analysis, the 83 configuration fields can be divided into three categories:

### 1. Essential User Inputs (19 fields, 23%)
These are inputs that must come from the user as they represent core business decisions:

1. **Region selection** (`region_list`, `region_selection`) - Business decision on what region to target
2. **Data timeframes** (`training_start_datetime`, `training_end_datetime`, etc.) - Training and calibration periods
3. **Feature selection** (`full_field_list`, `cat_field_list`, `tab_field_list`) - Which fields to use for modeling
4. **Target definition** (`label_name`, `id_name`, `marketplace_id_col`) - Label and identifier fields
5. **Classification configuration** (`multiclass_categories`, `class_weights`) - Binary vs. multi-class setup
6. **Core hyperparameters** (`num_round`, `max_depth`, `min_child_weight`) - Key model parameters
7. **Pipeline identification** (`service_name`, `pipeline_version`, etc.) - Deployment information
8. **Data source configuration** (`tag_edx_provider`, `tag_edx_subject`, etc.) - Data source details
9. **Model registration details** (`model_owner`, `model_registration_domain`, `framework`) - Ownership and deployment
10. **Performance requirements** (`expected_tps`, `max_latency_in_millisecond`) - Service level expectations

### 2. System Inputs (Fixed Values)
These are inputs with standardized values that don't require user configuration:

1. **Infrastructure settings** (`processing_instance_type_large`, `processing_volume_size`, etc.)
2. **Framework settings** (`py_version`, `processing_framework_version`)
3. **Default hyperparameters** (`device`, `batch_size`, `lr`, `max_epochs`, `optimizer`) 
4. **Processing entry points** (Various script entry points for different processing steps)
5. **Standard metrics** (`metric_choices`, `eval_metric_choices`)
6. **Default payload settings** (`max_acceptable_error_rate`, `special_field_values`)

### 3. Derived Inputs
These are inputs that can be automatically generated from essential inputs and system defaults:

1. **Field derivatives** (`input_tab_dim`, `num_classes`, `is_binary`)
2. **SQL transformations** (`schema_list`, `training_transform_sql`) 
3. **Path constructions** (`pipeline_subdirectory`, `pipeline_s3_loc`)
4. **Regional mappings** (`aws_region`, `org`)
5. **Output specifications** (`training_output_path`, `output_schema`)
6. **Model configurations** (`objective`, `eval_metric`)

## Section-by-Section Field Analysis

### 1. Base Hyperparameters Section

**User Inputs in Original Notebook:**
- Region selection (`region_list`, `region_selection`)
- Field lists (`full_field_list`, `cat_field_list`, `tab_field_list`)
- Label and ID fields (`label_name`, `id_name`, `marketplace_id_col`)
- Classification type (`multiclass_categories`, `is_binary`, `num_classes`)
- Class weights (`class_weights`)
- Metrics (`metric_choices`)
- Device settings (`device`, `header`, `batch_size`, `lr`, `max_epochs`, `optimizer`)

**Field Categorization:**
- **Essential User Inputs**: `region_list`, `region_selection`, `full_field_list`, `cat_field_list`, `tab_field_list`, `label_name`, `id_name`, `marketplace_id_col`, `multiclass_categories`, `class_weights`
- **System Inputs**: `metric_choices`, `device`, `header`, `batch_size`, `lr`, `max_epochs`, `optimizer`
- **Derived Inputs**: `region`, `input_tab_dim`, `num_classes`, `is_binary`

**Dependencies and Derivation Possibilities:**
- `input_tab_dim` is derived from `tab_field_list`
- `is_binary` is derived from `num_classes`
- `region` is derived from `region_list` and `region_selection`
- While field lists could potentially be organized into logical groups, the specific fields in `full_field_list`, `cat_field_list`, and `tab_field_list` are essential user inputs and cannot be fully automated

### 2. XGBoost Hyperparameters Section

**User Inputs in Original Notebook:**
- `model_class`
- `objective` (derived from `is_binary`)
- `eval_metric` (derived from `is_binary`)
- Model parameters (`num_round`, `max_depth`, `min_child_weight`)

**Field Categorization:**
- **Essential User Inputs**: `model_class`, `num_round`, `max_depth`, `min_child_weight`
- **System Inputs**: None in this section
- **Derived Inputs**: `objective`, `eval_metric`

**Dependencies and Derivation Possibilities:**
- `objective` and `eval_metric` are directly derived from `is_binary`
- `model_params` combines several individual parameters
- `xgb_hyperparams` combines base hyperparameters with model-specific parameters

### 3. Base Config Section

**User Inputs in Original Notebook:**
- Service name (`service_name`)
- Author details (`author = sais_session.owner_alias()`)
- Pipeline naming (`pipeline_name`, `pipeline_description`, `pipeline_version`)
- Container setup (`framework_version`, `py_version`)
- Date (`current_date`)
- AWS region mapping (derived from `region`)
- Source directory paths

**Field Categorization:**
- **Essential User Inputs**: `service_name`, `pipeline_version`, `framework_version`, `current_date`, `source_dir`
- **System Inputs**: `py_version`
- **Derived Inputs**: `bucket`, `role`, `author`, `pipeline_name`, `pipeline_description`, `pipeline_subdirectory`, `pipeline_s3_loc`, `aws_region`, `current_dir`

**Dependencies and Derivation Possibilities:**
- `pipeline_name` follows a pattern based on `author`, `service_name`, and `region`
- `pipeline_subdirectory` and `pipeline_s3_loc` are derived from other fields
- `aws_region` is mapped from the selected region
- `current_dir` is based on the environment
- While some patterns exist, `service_name`, `pipeline_version`, and `framework_version` represent business decisions that cannot be fully automated

### 4. Cradle Data Loading Config Section

**User Inputs in Original Notebook:**
- MDS data source configuration (`service_name`, `org_id`, `region`, `output_schema`)
- EDX data source details (`tag_edx_provider`, `tag_edx_subject`, `tag_edx_dataset`)
- Schema definitions (`tag_schema`, `edx_schema_overrides`)
- ETL job IDs mapping (`etl_job_id_dict`)
- Date ranges (`training_start_datetime`, `training_end_datetime`, etc.)
- SQL transformation templates and schema lists
- Output path configuration (`output_dir`, `training_output_path`, etc.)
- Cradle job settings (`cluster_type`, `cradle_account`)

**Field Categorization:**
- **Essential User Inputs**: `tag_edx_provider`, `tag_edx_subject`, `tag_edx_dataset`, `tag_schema`, `etl_job_id_dict`, `training_start_datetime`, `training_end_datetime`, `calibration_start_datetime`, `calibration_end_datetime`, `merge_sql`, `cradle_account`, `job_type`
- **System Inputs**: `split_job`, `days_per_split`
- **Derived Inputs**: `mds_field_list`, `output_schema`, `edx_schema_overrides`, `etl_job_id`, `training_tag_edx_manifest`, `calibration_tag_edx_manifest`, `schema_list`, `training_transform_sql`, `calibration_transform_sql`, `output_dir`, `training_output_path`, `calibration_output_path`, `training_output_fields`, `calibration_output_fields`, `cluster_type`

**Dependencies and Derivation Possibilities:**
- `mds_field_list` is derived from combining `tab_field_list` and `cat_field_list`
- `output_schema` is derived from `mds_field_list`
- `edx_schema_overrides` is derived from `tag_schema`
- `etl_job_id` is derived from `etl_job_id_dict` and `region`
- EDX manifests are derived from provider, subject, dataset, and date information
- SQL transformation is generated from templates and field lists
- Output paths include random UUIDs for uniqueness

### 5. Base Processing Config Section

**User Inputs in Original Notebook:**
- Instance types (`processing_instance_type_large`, `processing_instance_type_small`)
- Resource allocation (`processing_instance_count`, `processing_volume_size`)
- Processing source directory and framework version

**Field Categorization:**
- **Essential User Inputs**: None
- **System Inputs**: `processing_instance_type_large`, `processing_instance_type_small`, `processing_instance_count`, `processing_volume_size`, `processing_framework_version`
- **Derived Inputs**: `processing_source_dir`

**Dependencies and Derivation Possibilities:**
- `processing_source_dir` is derived from `source_dir`
- All system inputs could be set to standardized values based on best practices
- Resource allocations could be adjusted based on data volume estimation

### 6. Tabular Preprocessing Config Section

**User Inputs in Original Notebook:**
- Processing entry point (`processing_entry_point`)
- Job type (`job_type`)
- Hyperparameters reference (reuse of `base_hyperparameter`)
- Test/validation ratio (`test_val_ratio`)

**Field Categorization:**
- **Essential User Inputs**: None
- **System Inputs**: `processing_entry_point`, `test_val_ratio`
- **Derived Inputs**: All other fields reused from base processing config and hyperparameters

**Dependencies and Derivation Possibilities:**
- Processing config is largely derived from base processing config
- Hyperparameters are reused from earlier section
- `processing_entry_point` follows standard naming conventions
- `test_val_ratio` could be set to a standard value like 0.5

### 7. Training Config Section

**User Inputs in Original Notebook:**
- Instance type selection (`training_instance_type`)
- Resource allocation (`training_instance_count`, `training_volume_size`)
- Entry point script (`training_entry_point`)
- Hyperparameters reference (reuse of `xgb_hyperparams`)

**Field Categorization:**
- **Essential User Inputs**: `training_entry_point`
- **System Inputs**: `training_instance_count`, `training_volume_size`
- **Derived Inputs**: `training_instance_type`, and all hyperparameter-related fields

**Dependencies and Derivation Possibilities:**
- `training_instance_type` could be selected based on data volume
- System inputs could use standardized values based on best practices
- Hyperparameters are reused from earlier sections

### 8. Model Calibration Config Section

**User Inputs in Original Notebook:**
- Calibration method (`calibration_method`)
- Label and score field references
- Classification settings (reuse of `is_binary`, `num_classes`)

**Field Categorization:**
- **Essential User Inputs**: `calibration_method`
- **System Inputs**: `processing_entry_point`, `score_field`, `score_field_prefix`
- **Derived Inputs**: Fields reused from base processing config and hyperparameters

**Dependencies and Derivation Possibilities:**
- Most fields are derived from earlier sections
- `processing_entry_point` follows standard naming convention
- Score field settings could be standardized

### 9. Model Evaluation Config Section

**User Inputs in Original Notebook:**
- Processing entry point (`model_eval_processing_entry_point`)
- Source directory (`model_eval_source_dir`)
- Job type (`model_eval_job_type`)
- Evaluation metrics (`eval_metric_choices`)
- Processing configuration (large instance flag)

**Field Categorization:**
- **Essential User Inputs**: None
- **System Inputs**: `model_eval_processing_entry_point`, `model_eval_job_type`, `eval_metric_choices`, `use_large_processing_instance` 
- **Derived Inputs**: `model_eval_source_dir`, `xgboost_framework_version`, and other fields reused from base processing config

**Dependencies and Derivation Possibilities:**
- `model_eval_processing_entry_point` follows standard naming convention
- `model_eval_source_dir` is derived from `source_dir`
- `eval_metric_choices` could be derived from model type (binary vs. multi-class)
- `xgboost_framework_version` is derived from `base_config.framework_version`

### 10. Packaging Config Section

**User Inputs in Original Notebook:**
- Processing entry point (`packaging_entry_point`)
- Large instance flag (`use_large_processing_instance`)

**Field Categorization:**
- **Essential User Inputs**: None
- **System Inputs**: `packaging_entry_point`, `use_large_processing_instance`
- **Derived Inputs**: All fields reused from base processing config

**Dependencies and Derivation Possibilities:**
- `packaging_entry_point` follows standard naming convention
- `use_large_processing_instance` could be set to standard value
- Configuration is based on processing base config

### 11. MIMS Model Registration Section

**User Inputs in Original Notebook:**
- Model owner (`model_owner`)
- Model domain (`model_registration_domain`)
- Model objective (`model_registration_objective`)
- Input/output variable lists and types
- Content and response types
- Framework and entry point (`framework`, `inference_entry_point`)
- Instance type (`inference_instance_type`)

**Field Categorization:**
- **Essential User Inputs**: `model_owner`, `model_registration_domain`, `framework`, `inference_entry_point`
- **System Inputs**: `inference_instance_type`, `source_model_inference_content_types`, `source_model_inference_response_types`, `source_model_inference_output_variable_list`
- **Derived Inputs**: `model_registration_objective`, `source_model_inference_input_variable_list`

**Dependencies and Derivation Possibilities:**
- `model_registration_objective` is derived from `service_name` and `region`
- `source_model_inference_input_variable_list` is derived from field lists
- Content and response types can use standard values
- `inference_instance_type` could be set to standard resource allocation

### 12. Payload Config Section

**User Inputs in Original Notebook:**
- Processing entry point (`processing_entry_point`)
- Performance requirements (`expected_tps`, `max_latency_in_millisecond`, `max_acceptable_error_rate`)
- Default values and special field values

**Field Categorization:**
- **Essential User Inputs**: `expected_tps`, `max_latency_in_millisecond`
- **System Inputs**: `processing_entry_point`, `max_acceptable_error_rate`, `special_field_values`
- **Derived Inputs**: Fields reused from model registration and base processing config

**Dependencies and Derivation Possibilities:**
- `processing_entry_point` follows standard naming convention
- `max_acceptable_error_rate` could be set to a standard value
- `special_field_values` could use standardized default
- Configuration reuses many values from model registration

## Consolidated Essential User Inputs

Based on the analysis, the truly essential user inputs can be consolidated to:

1. **Data Configuration**
   - Region selection
   - Training and calibration date ranges
   - Service name (for data sources)
   - Field lists (full_field_list, cat_field_list, tab_field_list) - These are critical user inputs that define the model features

2. **Model Configuration**
   - Label field name and ID field
   - Binary vs. multi-class choice
   - Key hyperparameters (num_round, max_depth, min_child_weight)

3. **Pipeline Configuration**
   - Pipeline description
   - Pipeline version 
   - Author information

4. **Registration Configuration**
   - Model owner and domain
   - Performance requirements

While many configuration fields can be derived automatically based on these essential inputs using reasonable defaults, naming conventions, and project structure knowledge, the field lists and certain pipeline identification fields remain essential user inputs that represent critical business decisions.

## Derivation Rules for Automated Fields

Here are key derivation rules that could be implemented:

1. **Field Organization**
   - While keeping field lists as essential inputs, organize them into logical categories (buyer metrics, order metrics, etc.)
   - Provide templates or presets for common field combinations
   - Apply naming conventions to assist in categorizing fields as tabular or categorical

2. **SQL Transformations**
   - Generate SQL based on selected fields and standard join patterns
   - Use templates with field substitution

3. **Infrastructure Settings**
   - Select instance types based on data volume and complexity
   - Set resource allocations using standard sizing rules

4. **Path and File Naming**
   - Generate consistent path structures based on pipeline name
   - Create unique identifiers where needed

5. **Model Settings**
   - Choose appropriate objective and evaluation metrics based on problem type
   - Set reasonable defaults for secondary hyperparameters

6. **Processing Configurations**
   - Use standard entry points based on step type
   - Apply consistent configuration patterns across steps

For detailed technical implementations of these derivation rules, see:
- [DefaultValuesProvider Design](./default_values_provider_design.md) - Complete design for system input defaults
- [FieldDerivationEngine Design](./field_derivation_engine_design.md) - Complete design for derivation logic

## Conclusion

The `template_config_xgb_eval_v2.ipynb` notebook contains a large number of configuration fields. While many of these can be derived automatically, certain key inputs like field lists, service name, pipeline version, and description represent important business decisions that must remain user-specified. By organizing these essential inputs into logical groups and providing sensible defaults and templates for the derivable fields, the user experience could be significantly improved while maintaining necessary control over critical configuration aspects.

This analysis provides the foundation for implementing a streamlined interface that focuses on essential inputs while automatically deriving all other configuration values.
