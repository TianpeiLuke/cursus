# XGBoost Pipeline Configuration Field Dependency Table

This document provides a comprehensive analysis of all fields in the `template_config_xgb_eval_v2.ipynb` notebook, detailing their relationships, dependencies, and identifying which ones are essential user inputs versus derived fields.

## Field Dependency Analysis Table

| Field Name | Pydantic Base Class | Config Class | Derived Field? | Source Field | Dependent Logic | Essential User Input? | System Input? |
|------------|---------------------|--------------|----------------|--------------|-----------------|----------------------|--------------|
| `region_list` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `region_selection` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `region` | BaseModel | BasePipelineConfig | ✓ | `region_list`, `region_selection` | `region = region_list[region_selection]` | ✗ | ✗ |
| `full_field_list` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `cat_field_list` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `tab_field_list` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `input_tab_dim` | ModelHyperparameters | ModelHyperparameters | ✓ | `tab_field_list` | `input_tab_dim = len(tab_field_list)` | ✗ | ✗ |
| `label_name` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `id_name` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `marketplace_id_col` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `multiclass_categories` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `num_classes` | ModelHyperparameters | ModelHyperparameters | ✓ | `multiclass_categories` | `num_classes = len(multiclass_categories)` | ✗ | ✗ |
| `is_binary` | ModelHyperparameters | ModelHyperparameters | ✓ | `num_classes` | `is_binary = (num_classes == 2)` | ✗ | ✗ |
| `class_weights` | ModelHyperparameters | ModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `metric_choices` | ModelHyperparameters | ModelHyperparameters | ✓ | `is_binary` | `metric_choices = ['f1_score', 'auroc'] if is_binary else ['accuracy', 'f1_score']` | ✗ | ✓ |
| `device` | ModelHyperparameters | ModelHyperparameters | ✓ | N/A | `device = -1` | ✗ | ✓ |
| `header` | ModelHyperparameters | ModelHyperparameters | ✓ | N/A | `header = 'true'` | ✗ | ✓ |
| `batch_size` | ModelHyperparameters | ModelHyperparameters | ✓ | N/A | `batch_size = 32` | ✗ | ✓ |
| `lr` | ModelHyperparameters | ModelHyperparameters | ✓ | N/A | `lr = 0.01` | ✗ | ✓ |
| `max_epochs` | ModelHyperparameters | ModelHyperparameters | ✓ | N/A | `max_epochs = 100` | ✗ | ✓ |
| `optimizer` | ModelHyperparameters | ModelHyperparameters | ✓ | N/A | `optimizer = 'adam'` | ✗ | ✓ |
| `model_class` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `objective` | XGBoostModelHyperparameters | XGBoostModelHyperparameters | ✓ | `is_binary` | `objective = "binary:logistic" if is_binary else "multi:softmax"` | ✗ | ✗ |
| `eval_metric` | XGBoostModelHyperparameters | XGBoostModelHyperparameters | ✓ | `is_binary` | `eval_metric = ['logloss', 'auc'] if is_binary else ['mlogloss', 'merror']` | ✗ | ✗ |
| `num_round` | XGBoostModelHyperparameters | XGBoostModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `max_depth` | XGBoostModelHyperparameters | XGBoostModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `min_child_weight` | XGBoostModelHyperparameters | XGBoostModelHyperparameters | ✗ | N/A | N/A | ✓ | ✗ |
| `service_name` | BaseModel | BasePipelineConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `bucket` | BaseModel | BasePipelineConfig | ✓ | `sais_session` | `bucket=sais_session.team_owned_s3_bucket_name()` | ✗ | ✗ |
| `role` | BaseModel | BasePipelineConfig | ✓ | `PipelineSession` | `role=PipelineSession().get_caller_identity_arn()` | ✗ | ✗ |
| `author` | BaseModel | BasePipelineConfig | ✓ | `sais_session` | `author = sais_session.owner_alias()` | ✗ | ✗ |
| `pipeline_name` | BaseModel | BasePipelineConfig | ✓ | `author`, `service_name`, `region` | `pipeline_name = f"{author}-{service_name}-XGBoostModel-{region}"` | ✗ | ✗ |
| `pipeline_description` | BaseModel | BasePipelineConfig | ✓ | `service_name`, `region` | `pipeline_description = f'{service_name} XGBoost Model {region}'` | ✗ | ✗ |
| `pipeline_version` | BaseModel | BasePipelineConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `pipeline_subdirectory` | N/A (raw variable) | N/A | ✓ | `pipeline_name`, `pipeline_version` | `pipeline_subdirectory = f"{pipeline_name}_{pipeline_version}"` | ✗ | ✗ |
| `pipeline_s3_loc` | BaseModel | BasePipelineConfig | ✓ | `sais_session`, `pipeline_subdirectory` | `pipeline_s3_loc = f"s3://{Path(sais_session.team_owned_s3_bucket_name()) / 'MODS' / pipeline_subdirectory}"` | ✗ | ✗ |
| `framework_version` | BaseModel | BasePipelineConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `py_version` | BaseModel | BasePipelineConfig | ✓ | N/A | `py_version = 'py3'` | ✗ | ✓ |
| `current_date` | BaseModel | BasePipelineConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `aws_region` | BaseModel | BasePipelineConfig | ✓ | `region` | `aws_region = "us-east-1" if region == 'NA' else "eu-west-1" if region == 'EU' else "us-west-2"` | ✗ | ✗ |
| `current_dir` | N/A (raw variable) | N/A | ✓ | N/A | `current_dir = Path.cwd()` | ✗ | ✗ |
| `source_dir` | BaseModel | BasePipelineConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `org` | N/A (raw variable) | N/A | ✓ | `region` | `org = "na" if region == "NA" else "eu" if region == "EU" else "fe"` | ✗ | ✗ |
| `mds_field_list` | N/A (raw variable) | N/A | ✓ | `tab_field_list`, `cat_field_list` | `mds_field_list = ['objectId', 'transactionDate', 'Abuse.currency_exchange_rate_inline.exchangeRate', 'baseCurrency'] + tab_field_list + cat_field_list` | ✗ | ✗ |
| `output_schema` | MdsDataSourceConfig | MdsDataSourceConfig | ✓ | `mds_field_list` | `output_schema = [{'field_name': field,'field_type':'STRING'} for field in mds_field_list]` | ✗ | ✗ |
| `tag_edx_provider` | EdxDataSourceConfig | EdxDataSourceConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `tag_edx_subject` | EdxDataSourceConfig | EdxDataSourceConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `tag_edx_dataset` | EdxDataSourceConfig | EdxDataSourceConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `tag_schema` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `edx_schema_overrides` | EdxDataSourceConfig | EdxDataSourceConfig | ✓ | `tag_schema` | `edx_schema_overrides = [{'field_name': field,'field_type':'STRING'} for field in tag_schema]` | ✗ | ✗ |
| `etl_job_id_dict` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `etl_job_id` | N/A (raw variable) | N/A | ✓ | `etl_job_id_dict`, `region` | `etl_job_id = etl_job_id_dict[region]` | ✗ | ✗ |
| `training_start_datetime` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `training_end_datetime` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `training_tag_edx_manifest` | EdxDataSourceConfig | EdxDataSourceConfig | ✓ | `tag_edx_provider`, `tag_edx_subject`, `tag_edx_dataset`, `etl_job_id`, `training_start_datetime`, `training_end_datetime`, `region` | `training_tag_edx_manifest = f'arn:amazon:edx:iad::manifest/{tag_edx_provider}/{tag_edx_subject}/{tag_edx_dataset}/["{etl_job_id}",{training_start_datetime}Z,{training_end_datetime}Z,"{region}"]'` | ✗ | ✗ |
| `calibration_start_datetime` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `calibration_end_datetime` | N/A (raw variable) | N/A | ✗ | N/A | N/A | ✓ | ✗ |
| `calibration_tag_edx_manifest` | EdxDataSourceConfig | EdxDataSourceConfig | ✓ | `tag_edx_provider`, `tag_edx_subject`, `tag_edx_dataset`, `etl_job_id`, `calibration_start_datetime`, `calibration_end_datetime`, `region` | `calibration_tag_edx_manifest = f'arn:amazon:edx:iad::manifest/{tag_edx_provider}/{tag_edx_subject}/{tag_edx_dataset}/["{etl_job_id}",{calibration_start_datetime}Z,{calibration_end_datetime}Z,"{region}"]'` | ✗ | ✗ |
| `split_job` | JobSplitOptionsConfig | JobSplitOptionsConfig | ✓ | N/A | `split_job = False` | ✗ | ✓ |
| `days_per_split` | JobSplitOptionsConfig | JobSplitOptionsConfig | ✓ | N/A | `days_per_split = 7` | ✗ | ✓ |
| `merge_sql` | JobSplitOptionsConfig | JobSplitOptionsConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `schema_list` | N/A (raw variable) | N/A | ✓ | `select_variable_text_list` | `schema_list = ',\n'.join(select_variable_text_list)` | ✗ | ✗ |
| `training_transform_sql` | TransformSpecificationConfig | TransformSpecificationConfig | ✓ | `schema_list`, `mds_data_source.data_source_name`, `training_edx_data_source.data_source_name` | `training_transform_sql = transform_sql_template.format(schema_list=schema_list, data_source_name=mds_data_source.data_source_name, tag_source_name=training_edx_data_source.data_source_name)` | ✗ | ✗ |
| `calibration_transform_sql` | TransformSpecificationConfig | TransformSpecificationConfig | ✓ | `schema_list`, `mds_data_source.data_source_name`, `calibration_edx_data_source.data_source_name` | `calibration_transform_sql = transform_sql_template.format(schema_list=schema_list, data_source_name=mds_data_source.data_source_name, tag_source_name=calibration_edx_data_source.data_source_name)` | ✗ | ✗ |
| `output_dir` | N/A (raw variable) | N/A | ✓ | uuid | `output_dir=f'cradle_download_output/{uuid.uuid4()}'` | ✗ | ✗ |
| `training_output_path` | OutputSpecificationConfig | OutputSpecificationConfig | ✓ | `sandbox_session`, `output_dir` | `training_output_path = f's3://{sandbox_session.my_owned_s3_bucket_name()}/{output_dir}/train'` | ✗ | ✗ |
| `calibration_output_path` | OutputSpecificationConfig | OutputSpecificationConfig | ✓ | `sandbox_session`, `output_dir` | `calibration_output_path = f's3://{sandbox_session.my_owned_s3_bucket_name()}/{output_dir}/test'` | ✗ | ✗ |
| `training_output_fields` | OutputSpecificationConfig | OutputSpecificationConfig | ✓ | `training_data_sources_spec` | `training_output_fields = get_all_fields(training_data_sources_spec)` | ✗ | ✗ |
| `calibration_output_fields` | OutputSpecificationConfig | OutputSpecificationConfig | ✓ | `calibration_data_sources_spec` | `calibration_output_fields = get_all_fields(calibration_data_sources_spec)` | ✗ | ✗ |
| `cluster_type` | CradleJobSpecificationConfig | CradleJobSpecificationConfig | ✓ | `available_cluster_types`, `cluster_choice` | `cluster_type=available_cluster_types[cluster_choice]` | ✗ | ✗ |
| `cradle_account` | CradleJobSpecificationConfig | CradleJobSpecificationConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `job_type` | CradleDataLoadConfig | CradleDataLoadConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `processing_instance_type_large` | ProcessingStepConfigBase | ProcessingStepConfigBase | ✓ | N/A | `processing_instance_type_large = 'ml.m5.4xlarge'` | ✗ | ✓ |
| `processing_instance_type_small` | ProcessingStepConfigBase | ProcessingStepConfigBase | ✓ | N/A | `processing_instance_type_small = 'ml.m5.xlarge'` | ✗ | ✓ |
| `processing_instance_count` | ProcessingStepConfigBase | ProcessingStepConfigBase | ✓ | N/A | `processing_instance_count = 1` | ✗ | ✓ |
| `processing_volume_size` | ProcessingStepConfigBase | ProcessingStepConfigBase | ✓ | N/A | `processing_volume_size = 500` | ✗ | ✓ |
| `processing_source_dir` | ProcessingStepConfigBase | ProcessingStepConfigBase | ✓ | `source_dir` | `processing_source_dir = source_dir / 'pipeline_scripts'` | ✗ | ✗ |
| `processing_framework_version` | ProcessingStepConfigBase | ProcessingStepConfigBase | ✓ | N/A | `processing_framework_version = '1.2-1'` | ✗ | ✓ |
| `processing_entry_point` | TabularPreprocessingConfig | TabularPreprocessingConfig | ✓ | N/A | `processing_entry_point = "tabular_preprocess.py"` | ✗ | ✓ |
| `test_val_ratio` | TabularPreprocessingConfig | TabularPreprocessingConfig | ✓ | N/A | `test_val_ratio = 0.5` | ✗ | ✓ |
| `training_instance_type` | XGBoostTrainingConfig | XGBoostTrainingConfig | ✓ | `instance_type_list`, `instance_select` | `training_instance_type = instance_type_list[instance_select]` | ✗ | ✗ |
| `training_instance_count` | XGBoostTrainingConfig | XGBoostTrainingConfig | ✓ | N/A | `training_instance_count = 1` | ✗ | ✓ |
| `training_volume_size` | XGBoostTrainingConfig | XGBoostTrainingConfig | ✓ | N/A | `training_volume_size = 800` | ✗ | ✓ |
| `training_entry_point` | XGBoostTrainingConfig | XGBoostTrainingConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `processing_entry_point` (ModelCalibration) | ModelCalibrationConfig | ModelCalibrationConfig | ✓ | N/A | `base_processing_config_dict['processing_entry_point'] = 'model_calibration.py'` | ✗ | ✓ |
| `calibration_method` | ModelCalibrationConfig | ModelCalibrationConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `score_field` | ModelCalibrationConfig | ModelCalibrationConfig | ✓ | N/A | `score_field = 'prob_class_1'` | ✗ | ✓ |
| `score_field_prefix` | ModelCalibrationConfig | ModelCalibrationConfig | ✓ | N/A | `score_field_prefix = 'prob_class_'` | ✗ | ✓ |
| `model_eval_processing_entry_point` | XGBoostModelEvalConfig | XGBoostModelEvalConfig | ✓ | N/A | `model_eval_processing_entry_point = 'model_eval_xgb.py'` | ✗ | ✓ |
| `model_eval_source_dir` | XGBoostModelEvalConfig | XGBoostModelEvalConfig | ✓ | `source_dir` | `model_eval_source_dir = source_dir` | ✗ | ✗ |
| `model_eval_job_type` | XGBoostModelEvalConfig | XGBoostModelEvalConfig | ✓ | N/A | `model_eval_job_type = 'evaluation'` | ✗ | ✓ |
| `eval_metric_choices` | XGBoostModelEvalConfig | XGBoostModelEvalConfig | ✓ | `is_binary` | `eval_metric_choices = ['auc', 'accuracy'] if is_binary else ['accuracy', 'f1']` | ✗ | ✓ |
| `use_large_processing_instance` | XGBoostModelEvalConfig | XGBoostModelEvalConfig | ✓ | N/A | `previous_processing_config['use_large_processing_instance'] = True` | ✗ | ✓ |
| `xgboost_framework_version` | XGBoostModelEvalConfig | XGBoostModelEvalConfig | ✓ | `base_config.framework_version` | `xgboost_framework_version=base_config.framework_version` | ✗ | ✗ |
| `packaging_entry_point` | PackageStepConfig | PackageStepConfig | ✓ | N/A | `packaging_entry_point = 'mims_package.py'` | ✗ | ✓ |
| `model_owner` | ModelRegistrationConfig | ModelRegistrationConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `model_registration_domain` | ModelRegistrationConfig | ModelRegistrationConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `model_registration_objective` | ModelRegistrationConfig | ModelRegistrationConfig | ✓ | `service_name`, `region` | `model_registration_objective = f'AtoZ_Claims_SM_Model_{region}'` | ✗ | ✗ |
| `source_model_inference_output_variable_list` | ModelRegistrationConfig | ModelRegistrationConfig | ✓ | `is_binary` | `source_model_inference_output_variable_list = {'legacy-score': 'NUMERIC', 'calibrated-score': 'NUMERIC', 'custom-output-label': 'TEXT'}` | ✗ | ✓ |
| `source_model_inference_content_types` | ModelRegistrationConfig | ModelRegistrationConfig | ✓ | N/A | `source_model_inference_content_types = ["text/csv"]` | ✗ | ✓ |
| `source_model_inference_response_types` | ModelRegistrationConfig | ModelRegistrationConfig | ✓ | N/A | `source_model_inference_response_types = ["application/json"]` | ✗ | ✓ |
| `source_model_inference_input_variable_list` | ModelRegistrationConfig | ModelRegistrationConfig | ✓ | `adjusted_full_field_list`, `tab_field_list`, `cat_field_list` | `source_model_inference_input_variable_list = create_model_variable_list(adjusted_full_field_list, tab_field_list, cat_field_list)` | ✗ | ✗ |
| `framework` | ModelRegistrationConfig | ModelRegistrationConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `inference_entry_point` | ModelRegistrationConfig | ModelRegistrationConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `inference_instance_type` | ModelRegistrationConfig | ModelRegistrationConfig | ✓ | N/A | `inference_instance_type = "ml.m5.4xlarge"` | ✗ | ✓ |
| `expected_tps` | PayloadConfig | PayloadConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `max_latency_in_millisecond` | PayloadConfig | PayloadConfig | ✗ | N/A | N/A | ✓ | ✗ |
| `max_acceptable_error_rate` | PayloadConfig | PayloadConfig | ✓ | N/A | `max_acceptable_error_rate = 0.2` | ✗ | ✓ |
| `special_field_values` | PayloadConfig | PayloadConfig | ✓ | N/A | `special_field_values = None` | ✗ | ✓ |
| `processing_entry_point` (Payload) | PayloadConfig | PayloadConfig | ✓ | N/A | `processing_base_dict['processing_entry_point'] = 'mims_payload.py'` | ✗ | ✓ |

## Summary Statistics

- **Total Fields**: 83
- **Essential User Inputs**: 19 (23%)
- **Derived Fields**: 64 (77%)
- **Pydantic Models**: 14 distinct base classes

## Key Insights

1. **Most Essential User Inputs**: Base hyperparameters contain the largest number of essential user inputs, particularly field lists, which define the model features.

2. **Common Derivation Patterns**:
   - Field aggregations (joining lists, counting lengths)
   - Path construction (combining directory paths)
   - Conditional logic based on model type (binary vs. multi-class)
   - Resource naming based on user and service identifiers

3. **Potential for Simplification**:
   - Field lists could be organized into logical feature groups
   - Many derived fields follow standard patterns and could be automatically generated
   - Infrastructure settings could be determined based on data volume

4. **Critical Business Decisions**:
   - Field lists (defining model features)
   - Service name and pipeline details (versioning, description)
   - Data timeframes (training and calibration periods)
   - Model owner and domain information

This analysis provides a foundation for streamlining the configuration process while ensuring users maintain control over essential business decisions.

## System Inputs Categorized

The following fields have fixed default values and are considered system inputs that can be preconfigured and don't require user input:

### 1. Base Model Hyperparameters
- `metric_choices` - Default evaluation metrics based on binary vs multi-class
- `device` - Default computation device (-1)
- `header` - Default header setting ('true')
- `batch_size` - Default batch size (32)
- `lr` - Default learning rate (0.01)
- `max_epochs` - Default maximum training epochs (100)
- `optimizer` - Default optimizer ('adam')

### 2. Infrastructure Configuration
#### 2.1 Framework Settings
- `py_version` - Python version ('py3')
- `processing_framework_version` - Processing framework version ('1.2-1')

#### 2.2 Processing Resources
- `processing_instance_type_large` - Large processing instance type ('ml.m5.4xlarge')
- `processing_instance_type_small` - Small processing instance type ('ml.m5.xlarge')
- `processing_instance_count` - Default processing instance count (1)
- `processing_volume_size` - Default processing volume size (500 GB)
- `test_val_ratio` - Default test/validation split ratio (0.5)

#### 2.3 Training Resources
- `training_instance_count` - Default training instance count (1)
- `training_volume_size` - Default training volume size (800 GB)

#### 2.4 Inference Resources
- `inference_instance_type` - Default inference instance type ('ml.m5.4xlarge')

### 3. Processing Step Entry Points
- `processing_entry_point` (TabularPreprocessing) - "tabular_preprocess.py"
- `processing_entry_point` (ModelCalibration) - "model_calibration.py"
- `model_eval_processing_entry_point` - "model_eval_xgb.py"
- `packaging_entry_point` - "mims_package.py"
- `processing_entry_point` (Payload) - "mims_payload.py"

### 4. Payload Configuration
- `max_acceptable_error_rate` - Default maximum acceptable error rate (0.2)
- `special_field_values` - Default special field values (None)

## Essential User Inputs Categorized

Based on the analysis above, here are the 19 essential user inputs categorized into logical groups:

### 1. Base Configuration
#### 1.1 Region Information
- `region_list` and `region_selection` - Geographic region for deployment

#### 1.2 Feature Definitions
- `full_field_list`, `cat_field_list`, `tab_field_list` - Model feature field definitions

#### 1.3 Identity and Classification Fields
- `label_name`, `id_name`, `marketplace_id_col` - Critical identifier fields 
- `multiclass_categories`, `class_weights` - Classification configuration

### 2. Model Configuration
#### 2.1 Model Type
- `model_class` - The model type to use (XGBoost)

#### 2.2 Hyperparameters
- `num_round` - Number of boosting rounds
- `max_depth` - Maximum tree depth
- `min_child_weight` - Minimum child weight

#### 2.3 Model Calibration
- `calibration_method` - Method for calibrating model probabilities

### 3. Data Sources Configuration
#### 3.1 EDX Data Configuration
- `tag_edx_provider`, `tag_edx_subject`, `tag_edx_dataset` - EDX data source details
- `tag_schema` - Schema for tagging
- `etl_job_id_dict` - ETL job identifiers by region

#### 3.2 Data Timeframes
- `training_start_datetime`, `training_end_datetime` - Training data time period
- `calibration_start_datetime`, `calibration_end_datetime` - Calibration data time period

#### 3.3 Data Transformation
- `merge_sql` - SQL for merging data

### 4. Pipeline and Infrastructure
#### 4.1 Pipeline Identification
- `service_name` - Service identifier
- `pipeline_version` - Version identifier
- `current_date` - Execution date
- `framework_version` - Framework version constraint

#### 4.2 Source Code
- `source_dir` - Source code directory
- `training_entry_point` - Training script entry point
- `inference_entry_point` - Inference script entry point

#### 4.3 Infrastructure
- `job_type` - Type of job to run
- `cradle_account` - Cradle account information

### 5. Model Registration and Deployment
#### 5.1 Registration Details
- `model_owner` - Model ownership information
- `model_registration_domain` - Domain for model registration
- `framework` - Framework for model registration

#### 5.2 Performance Requirements
- `expected_tps` - Expected transactions per second
- `max_latency_in_millisecond` - Maximum acceptable latency
