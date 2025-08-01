# XGBoost with Evaluation Configuration Notebook

This document summarizes the `create_config_xgb_w_eval.ipynb` notebook, which is used to create a configuration file for an XGBoost model pipeline with evaluation capabilities.

## Overview

The notebook guides users through creating a comprehensive configuration for an XGBoost machine learning pipeline. It collects various user inputs to configure each step of the pipeline, from data loading to model evaluation, and saves them into a unified JSON configuration file that can be used to execute the pipeline.

## User Inputs by Pipeline Step

### 1. Environment Setup

**User Inputs:**
- Region selection (`NA`, `EU`, or `FE`)
- Current date (for versioning and S3 paths)

### 2. Base Hyperparameters

**User Inputs:**
- Field lists:
  - `full_field_list`: Complete list of all fields in the dataset
  - `cat_field_list`: List of categorical fields
  - `tab_field_list`: List of tabular (numerical) fields
- Label and ID configuration:
  - `label_name`: Name of the target variable (e.g., 'is_abuse')
  - `id_name`: Name of the ID field (e.g., 'order_id')
  - `marketplace_id_col`: Name of the marketplace ID column
- Classification settings:
  - `multiclass_categories`: List of class values
  - `class_weights`: Weights for each class
- Training parameters:
  - `batch_size`: Batch size for training
  - `lr`: Learning rate
  - `max_epochs`: Maximum number of training epochs
  - `optimizer`: Optimization algorithm
  - `metric_choices`: Evaluation metrics to track

**Related Documentation:**
- [Base Pipeline Configuration](../pipeline_steps/README.md)

### 3. XGBoost Hyperparameters

**User Inputs:**
- XGBoost specific parameters:
  - `objective`: Training objective ('binary:logistic' or 'multi:softmax')
  - `eval_metric`: Evaluation metrics for XGBoost
  - `num_round`: Number of boosting rounds
  - `max_depth`: Maximum tree depth
  - `min_child_weight`: Minimum sum of instance weight needed in a child

**Related Documentation:**
- [XGBoost Training Step](../pipeline_steps/training_step_xgboost.md)

### 4. Base Pipeline Configuration

**User Inputs:**
- Pipeline identification:
  - `service_name`: Name of the service (e.g., 'AtoZ')
  - `author`: Owner of the pipeline
  - `pipeline_name`: Name of the pipeline
  - `pipeline_description`: Description of the pipeline
  - `pipeline_version`: Version number
- S3 configuration:
  - `bucket`: S3 bucket name
  - `pipeline_s3_loc`: S3 location for pipeline artifacts
- Framework configuration:
  - `framework_version`: XGBoost framework version
  - `py_version`: Python version
  - `source_dir`: Directory containing source code

**Related Documentation:**
- [Base Pipeline Configuration](../pipeline_steps/README.md)

### 5. Cradle Data Loading Configuration

**User Inputs:**
- MDS data source configuration:
  - `service_name`: Name of the MDS service
  - `org`: Organization ID
  - `mds_field_list`: List of fields to extract from MDS
- Data source specification:
  - `start_date`: Start date for data extraction
  - `end_date`: End date for data extraction
- Job configuration:
  - `job_type`: Type of job ('training', 'validation', 'testing', 'calibration')
  - `cluster_type`: Type of cluster for the job
  - `cradle_account`: Cradle account name

**Related Documentation:**
- [Cradle Data Load Step](../pipeline_steps/data_load_step_cradle.md)

### 6. Tabular Preprocessing Configuration

**User Inputs:**
- Processing configuration:
  - `processing_instance_type`: Instance type for processing
  - `processing_instance_count`: Number of instances for processing
  - `processing_volume_size`: Volume size for processing
- Data splitting parameters:
  - `train_ratio`: Ratio of data for training
  - `test_val_ratio`: Ratio of test to validation data

**Related Documentation:**
- [Tabular Preprocessing Step](../pipeline_steps/tabular_preprocessing_step.md)

### 7. XGBoost Training Configuration

**User Inputs:**
- Training infrastructure:
  - `training_instance_type`: Instance type for training
  - `training_instance_count`: Number of instances for training
  - `training_volume_size`: Volume size for training
- Training script:
  - `training_entry_point`: Entry point script for training

**Related Documentation:**
- [XGBoost Training Step](../pipeline_steps/training_step_xgboost.md)

### 8. Model Evaluation Configuration

**User Inputs:**
- Evaluation parameters:
  - `eval_metric_choices`: Metrics to use for evaluation
  - `job_type`: Type of evaluation job

**Related Documentation:**
- [XGBoost Model Evaluation Step](../pipeline_steps/model_eval_step_xgboost.md)

### 9. Model Creation Configuration

**User Inputs:**
- Inference configuration:
  - `inference_instance_type`: Instance type for inference
  - `inference_entry_point`: Entry point script for inference
  - Container settings:
    - `container_startup_health_check_timeout`: Timeout for container health check
    - `container_memory_limit`: Memory limit for container
    - `inference_memory_limit`: Memory limit for inference
    - `max_concurrent_invocations`: Maximum concurrent invocations
    - `max_payload_size`: Maximum payload size

**Related Documentation:**
- [XGBoost Model Step](../pipeline_steps/model_step_xgboost.md)

### 10. MIMS Packaging Configuration

**User Inputs:**
- Packaging parameters:
  - `processing_entry_point`: Entry point script for packaging
  - `processing_source_dir`: Source directory for packaging scripts

**Related Documentation:**
- [MIMS Packaging Step](../pipeline_steps/mims_packaging_step.md)

### 11. MIMS Registration Configuration

**User Inputs:**
- Registration parameters:
  - `model_owner`: Owner of the model
  - `model_registration_domain`: Domain for model registration
  - `model_registration_objective`: Objective of the model
  - `source_model_inference_content_types`: Content types for inference
  - `source_model_inference_response_types`: Response types for inference
  - `source_model_inference_input_variable_list`: Input variables for inference
  - `source_model_inference_output_variable_list`: Output variables for inference

**Related Documentation:**
- [MIMS Registration Step](../pipeline_steps/mims_registration_step.md)

### 12. Batch Transform Configuration

**User Inputs:**
- Transform parameters:
  - `job_type`: Type of transform job
  - `transform_instance_type`: Instance type for transform
  - `transform_instance_count`: Number of instances for transform

**Related Documentation:**
- [Batch Transform Step](../pipeline_steps/batch_transform_step.md)

## Pipeline Flow

The notebook configures a complete end-to-end pipeline with the following flow:

1. **Data Loading**: Load data from MDS using Cradle
2. **Data Preprocessing**: Preprocess the tabular data
3. **Model Training**: Train an XGBoost model
4. **Model Evaluation**: Evaluate the model performance
5. **Model Creation**: Create a SageMaker model artifact
6. **Model Packaging**: Package the model for MIMS
7. **Model Registration**: Register the model with MIMS
8. **Batch Transform**: Generate predictions using the model

The configuration created by this notebook can be used with the `mods_pipeline_xgb_w_eval.ipynb` notebook to execute the complete pipeline.
