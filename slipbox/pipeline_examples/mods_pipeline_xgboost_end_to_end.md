# XGBoost End-to-End Pipeline

## Pipeline Summary
The XGBoost End-to-End Pipeline builds a complete machine learning workflow that includes data loading, preprocessing, model training, model creation, packaging, and registration. It handles both training and calibration data flows in a single pipeline.

## Steps Involved

1. **Data Loading (Training)**
   - Uses [Cradle Data Load Step](../pipeline_steps/data_load_step_cradle.md)
   - Loads training data from source systems (MDS, EDX, or ANDES)

2. **Tabular Preprocessing (Training)**
   - Uses [Tabular Preprocessing Step](../pipeline_steps/tabular_preprocessing_step.md)
   - Prepares training data for model training

3. **XGBoost Model Training**
   - Uses [XGBoost Training Step](../pipeline_steps/training_step_xgboost.md)
   - Trains an XGBoost model on the preprocessed data

4. **XGBoost Model Creation**
   - Uses [XGBoost Model Step](../pipeline_steps/model_step_xgboost.md)
   - Creates a SageMaker model artifact from the trained model

5. **MIMS Packaging**
   - Uses [MIMS Packaging Step](../pipeline_steps/mims_packaging_step.md)
   - Packages the model for deployment in MIMS

6. **Model Registration**
   - Uses [MIMS Registration Step](../pipeline_steps/mims_registration_step.md)
   - Registers the model with MIMS

7. **Data Loading (Calibration)**
   - Uses [Cradle Data Load Step](../pipeline_steps/data_load_step_cradle.md)
   - Loads calibration data from source systems

8. **Tabular Preprocessing (Calibration)**
   - Uses [Tabular Preprocessing Step](../pipeline_steps/tabular_preprocessing_step.md)
   - Prepares calibration data for model evaluation

## Step Connections (Adjacency List)

```
Training Flow:
- Data Loading (Training) → Tabular Preprocessing (Training)
- Tabular Preprocessing (Training) → XGBoost Model Training
- XGBoost Model Training → XGBoost Model Creation
- XGBoost Model Training → MIMS Packaging (model artifacts)
- XGBoost Model Creation → MIMS Packaging (dependency)
- MIMS Packaging → Model Registration

Calibration Flow:
- Data Loading (Calibration) → Tabular Preprocessing (Calibration)
```

## Input/Output Connections

### Data Loading (Training) → Tabular Preprocessing (Training)
- **Output**: Cradle data loading outputs data to S3 with three channels:
  - `OUTPUT_TYPE_DATA`: Contains the actual data files
  - `OUTPUT_TYPE_METADATA`: Contains metadata about the data
  - `OUTPUT_TYPE_SIGNATURE`: Contains data signatures
- **Input**: Tabular preprocessing takes the `OUTPUT_TYPE_DATA` channel as its `data_input`

### Tabular Preprocessing (Training) → XGBoost Model Training
- **Output**: Tabular preprocessing outputs processed data to `{pipeline_s3_loc}/tabular_preprocessing/training`
- **Input**: XGBoost training dynamically sets its `input_path` to the preprocessing output location

### XGBoost Model Training → XGBoost Model Creation
- **Output**: XGBoost training outputs model artifacts to `{pipeline_s3_loc}/xgboost_model_artifacts`
- **Input**: XGBoost model creation takes the `ModelArtifacts.S3ModelArtifacts` property as input

### XGBoost Model Training → MIMS Packaging
- **Output**: XGBoost training provides model artifacts via `ModelArtifacts.S3ModelArtifacts`
- **Input**: MIMS packaging takes these model artifacts as input for packaging

### MIMS Packaging → Model Registration
- **Output**: MIMS packaging outputs packaged model to S3
- **Input**: Model registration takes the packaged model S3 URI from `ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`

### Data Loading (Calibration) → Tabular Preprocessing (Calibration)
- **Output**: Cradle data loading outputs data to S3 with three channels
- **Input**: Tabular preprocessing takes the `OUTPUT_TYPE_DATA` channel as its `data_input`

## Notes
- The pipeline enforces region settings, forcing model training and creation to occur in the NA region (us-east-1)
- The pipeline handles execution document configuration for both Cradle data loading and model registration
- The pipeline supports multiple regions for model registration
- The calibration flow runs independently from the training flow
- The pipeline uses a configuration-driven approach, extracting all settings from a JSON configuration file
