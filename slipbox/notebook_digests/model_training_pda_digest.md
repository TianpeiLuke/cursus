# XGBoost PDA Model Training Notebook Digest

## Original Notebook
[dockers/xgboost_pda/model_training_pda.ipynb](../../../dockers/xgboost_pda/model_training_pda.ipynb)

## Notebook Summary
This Jupyter notebook implements a complete machine learning workflow for Potential Delivery Abuse (PDA) detection using XGBoost. The notebook loads historical order data, performs data cleaning and preprocessing, identifies key features through feature importance analysis, trains an XGBoost classifier, and evaluates model performance across different market segments and fulfillment types. It compares the new model's performance against an existing baseline model (ContactRiskPDAModelNA) using metrics like AUC, recall, and dollar recall. The final model and selected features are saved for deployment.

## Task Breakdown

### Task 1: Environment Setup
**Cells**: [1-7]
**Summary**: Sets up the SageMaker environment by checking memory availability, installing required packages (scikit-learn, XGBoost, sklearn2pmml), and importing necessary libraries.
**Inputs**: None
**Outputs**: Configured Python environment with required dependencies
**Key Parameters**: 
- Package versions: scikit-learn, xgboost, sklearn2pmml, sklearn-pandas
- Libraries: pandas, numpy, sklearn components, matplotlib

### Task 2: Data Loading
**Cells**: [8-16]
**Summary**: Loads PDA order data from CSV files with transaction histories and signature information.
**Inputs**: 
- Data path: '/home/ec2-user/SageMaker/PDA/output_tabular/2025-01-01_2025-04-30T_output.csv'
- Signature path: '/home/ec2-user/SageMaker/PDA/output_tabular/output_signature.csv'
**Outputs**: 
- Raw dataframe (df) containing order features and target variables
**Key Parameters**: None

### Task 3: Data Preprocessing
**Cells**: [17-22]
**Summary**: Cleans and prepares data by converting timestamps, dropping columns with unique/null values and duplicate orders, and splitting into training and out-of-time validation sets.
**Inputs**: Raw dataframe (df)
**Outputs**: 
- Training data for non-abusive orders (df_good)
- Training data for abusive orders (df_bad)
- Out-of-time validation data (df_oot)
**Key Parameters**: 
- Split date: '2025-04-23' (data before this date used for training, after for validation)

### Task 4: Feature Selection Setup
**Cells**: [23-25]
**Summary**: Identifies numerical and categorical features for modeling, while excluding non-predictive columns such as IDs, timestamps, and redundant information.
**Inputs**: Preprocessed dataframes
**Outputs**: 
- List of numerical features (num_cols)
- List of categorical features (cat_cols)
**Key Parameters**: 
- Excluded columns list containing 40+ columns like objectId, transactionDate, customerId, etc.

### Task 5: Model Training Functions Definition
**Cells**: [26-31]
**Summary**: Defines utility functions for building the XGBoost pipeline, including functions for data sampling, feature mapping, train/test splitting, pipeline creation, feature importance extraction, and model evaluation.
**Inputs**: None
**Outputs**: Function definitions
**Key Parameters**:
- XGBoost parameters: objective="binary:logistic", learning_rate=0.1, n_estimators=100, max_depth=4, subsample=0.4, colsample_bytree=0.4, gamma=5
- Functions for data preprocessing: get_df_in, get_mapper, get_df_train_test, get_pipeline, get_xgb_importance, get_xgb_performance

### Task 6: Baseline Model Training & Evaluation
**Cells**: [32-38]
**Summary**: Trains an initial XGBoost model using all available features and evaluates its performance on the out-of-time validation set.
**Inputs**: 
- Training data (df_good, df_bad)
- Out-of-time validation data (df_oot)
- Feature lists (num_cols, cat_cols)
**Outputs**: 
- Trained pipeline (pipeline_base)
- Feature importance data (xgb_importance_base)
- Performance metrics (AUC, recall, dollar recall)
**Key Parameters**: 
- Sample size: All available good samples
- Test/validation split: 1% test, 99% train

### Task 7: Feature Importance Analysis & Selection
**Cells**: [39-43]
**Summary**: Performs feature selection by iteratively training models and identifying features with positive importance scores, then filters the feature list to include only these important features.
**Inputs**: 
- Training data (df_good, df_bad)
- Out-of-time validation data (df_oot)
- Feature lists (num_cols, cat_cols)
**Outputs**: 
- Selected numerical features (num_cols_selected)
- Selected categorical features (cat_cols_selected)
- Feature importance rankings (selected_features)
**Key Parameters**: 
- Maximum iterations: 20
- Importance threshold: 0
- Sample size: All available good samples

### Task 8: Final Model Training
**Cells**: [44-49]
**Summary**: Trains the final XGBoost model using only the selected important features.
**Inputs**: 
- Training data with selected features
- Selected feature lists (num_cols_selected, cat_cols_selected)
**Outputs**: 
- Final trained model (pipeline)
- Final feature importance data (xgb_importance)
**Key Parameters**: Same XGBoost parameters as baseline model

### Task 9: Model Persistence
**Cells**: [50-53]
**Summary**: Saves the trained model, feature importance data, and selected feature list to files for future use in deployment.
**Inputs**: 
- Final trained pipeline (pipeline)
- Feature importance data (xgb_importance)
- Selected features list (feature_selected)
**Outputs**: 
- Saved model file: "xgb_model_na_0702.pkl"
- Feature importance CSV: '/home/ec2-user/SageMaker/PDA/feature_selection/feature_importance_na_0702.csv'
- Selected features file: '/home/ec2-user/SageMaker/PDA/feature_selection/feature_selected_na_0702.sig'
**Key Parameters**: None

### Task 10: Comprehensive Model Evaluation
**Cells**: [54-66]
**Summary**: Evaluates the final model's performance on the overall dataset and on specific segments (by country and fulfillment type), comparing against the baseline model.
**Inputs**: 
- Final model (pipeline)
- Out-of-time validation data (df_oot)
**Outputs**: 
- Performance metrics: AUC, recall, dollar recall
- Performance comparison across segments
**Key Parameters**: 
- Evaluation segments: Overall, US, CA, MX, BR, Amazon fulfillment, Marketplace fulfillment
- Metrics: AUC, recall at 5% FPR, dollar recall at 5% FPR

## Task-to-Pipeline-Step Mapping

| Task | Pipeline Step Type | Notes |
|------|-------------------|-------|
| Environment Setup | Base | This is typically handled outside the pipeline or as pipeline initialization |
| Data Loading | CradleDataLoading | Would need to adapt to load from S3 rather than local paths; requires configuring data_sources_spec, transform_spec, and output_spec |
| Data Preprocessing | TabularPreprocessing | Would handle timestamp conversion, dropping columns, and data splitting; needs categorical_columns and numerical_columns parameters |
| Feature Selection Setup | TabularPreprocessing | Can be included as part of preprocessing step by specifying the excluded columns |
| Model Training Functions Definition | N/A | These functions would be incorporated into scripts used by various pipeline steps |
| Baseline Model Training & Evaluation | XGBoostTraining + XGBoostModelEval | Initial model training and evaluation with all features |
| Feature Importance Analysis & Selection | Custom Step Needed | The iterative feature selection process isn't directly supported by existing steps |
| Final Model Training | XGBoostTraining | Second XGBoost training run with selected features |
| Model Persistence | XGBoostModel + Registration | Convert trained model to SageMaker model and register it |
| Comprehensive Model Evaluation | XGBoostModelEval | Would need custom script to evaluate on multiple segments |

## Proposed Pipeline Structure

```
CradleDataLoading (load raw PDA data)
    ↓
TabularPreprocessing (clean data, convert timestamps, split by date, prepare features)
    ↓
XGBoostTraining (train initial model with all features)
    ↓                      ↘
    ↓                       XGBoostModelEval (evaluate initial model)
    ↓                            ↓
FeatureSelectionStep (custom) (select important features)
    ↓
TabularPreprocessing (filter to selected features)
    ↓
XGBoostTraining (train final model with selected features)
    ↓                      ↘
    ↓                       XGBoostModelEval (evaluate final model by segments)
    ↓
XGBoostModel (create model artifact)
    ↓
Registration (save model and metadata)
```

## New Pipeline Step Recommendations

Based on the notebook analysis, I recommend developing the following new pipeline steps:

1. **FeatureSelectionStep**: This custom step would:
   - Take a trained model as input
   - Extract feature importance scores
   - Apply threshold-based feature selection
   - Support iterative model training to refine feature selection
   - Output a selected feature list
   
   Key parameters would include:
   - Importance threshold
   - Maximum iterations
   - Sample size strategy
   - Feature persistence method

2. **SegmentedEvaluationStep**: This step would extend the XGBoostModelEval step to:
   - Evaluate model performance on multiple data segments (by country, fulfillment type, etc.)
   - Generate comparative metrics against a baseline model
   - Calculate specialized metrics like dollar recall
   - Create visualization artifacts for performance comparison
   
   Key parameters would include:
   - Segmentation columns
   - Baseline model reference
   - Custom metrics configuration
   - Visualization options

3. **ModelComparisonStep**: This step would:
   - Compare multiple model versions
   - Generate comparative metrics tables and visualizations
   - Support A/B testing analysis
   - Provide statistical significance testing for performance differences
   
   This would be particularly useful for comparing new models against production baselines.

## Implementation Considerations

1. **Data Sources**: The notebook uses hardcoded file paths on the SageMaker instance. In a pipeline, these would need to be configured as S3 URIs in the CradleDataLoadingStep or by creating a custom data loading step that handles CSV files.

2. **Custom Preprocessing Functions**: The notebook uses custom functions for data preprocessing and risk transformations. These would need to be incorporated into the TabularPreprocessingStep scripts or implemented as a custom processing step.

3. **Two-stage Training**: The notebook employs a two-stage approach (train with all features, select important ones, train final model). The pipeline would need to support this workflow, possibly through chained steps or a custom orchestration step.

4. **PMMLPipeline Integration**: The notebook uses PMMLPipeline for model creation, which might require custom scripts in the XGBoostTraining step to replicate this functionality.

5. **Feature Selection Logic**: The iterative feature selection approach would require custom implementation, as it's not directly supported by the existing steps.

6. **Segmented Evaluation**: The comprehensive evaluation by region and fulfillment type would require custom evaluation logic beyond standard XGBoostModelEval capabilities.

7. **Baseline Model Comparison**: The notebook compares against an existing production model. The pipeline would need a mechanism to access and compare against this baseline.

8. **Data Splitting Strategy**: The notebook uses a date-based split rather than a random split, which would require custom implementation in the preprocessing step.

9. **Memory Management**: The notebook includes garbage collection and memory monitoring, which should be considered for long-running pipeline steps.

10. **Configuration Management**: The notebook references a configuration file (ipynb.cfg) which would need to be incorporated into the pipeline configuration approach.
