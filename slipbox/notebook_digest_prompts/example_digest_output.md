# Example Notebook Digest: XGBoost PDA Model Training

## Original Notebook
[dockers/xgboost_pda/model_training_pda.ipynb](../../../dockers/xgboost_pda/model_training_pda.ipynb)

## Notebook Summary
This notebook implements an end-to-end XGBoost model training workflow for a Potential Delivery Abuse (PDA) detection system. It loads historical order data, preprocesses features, trains an XGBoost classifier, evaluates model performance against a baseline model, performs feature selection, and saves the final model and selected features for future use.

## Task Breakdown

### Task 1: Environment Setup
**Cells**: [1-7]
**Summary**: Sets up the SageMaker environment, checks available memory, installs required packages, and imports necessary libraries.
**Input**: None
**Output**: Configured environment with required dependencies
**Key Parameters**: 
- Packages: scikit-learn, xgboost, sklearn2pmml, sklearn-pandas

### Task 2: Data Loading
**Cells**: [8-14]
**Summary**: Loads order data from CSV files containing feature data and target variables.
**Input**: 
- Data path: '/home/ec2-user/SageMaker/PDA/output_tabular/2025-01-01_2025-04-30T_output.csv'
- Signature path: '/home/ec2-user/SageMaker/PDA/output_tabular/output_signature.csv'
**Output**: Raw dataframe with order features and target variables
**Key Parameters**: None

### Task 3: Data Preprocessing
**Cells**: [15-21]
**Summary**: Cleans and preprocesses the data by converting timestamps, removing columns with unique/null values, dropping duplicates, and splitting data into training and testing sets.
**Input**: Raw dataframe
**Output**: 
- Training data (df_good, df_bad)
- Testing data (df_oot)
**Key Parameters**: 
- Split date: '2025-04-23'

### Task 4: Feature Selection
**Cells**: [22-25]
**Summary**: Identifies numerical and categorical features, excluding specific columns not useful for modeling.
**Input**: Preprocessed dataframes
**Output**: Lists of numerical and categorical features
**Key Parameters**: 
- Excluded columns: order IDs, timestamps, customer IDs, etc.

### Task 5: Model Training Functions Definition
**Cells**: [26-31]
**Summary**: Defines functions for creating data mappers, training/test splits, pipeline creation, and model evaluation.
**Input**: None
**Output**: Function definitions
**Key Parameters**:
- XGBoost parameters: learning_rate=0.1, n_estimators=100, max_depth=4, etc.

### Task 6: Model Training and Baseline Evaluation
**Cells**: [32-38]
**Summary**: Trains an initial XGBoost model and evaluates its performance on test data.
**Input**: Training data (df_good, df_bad), test data (df_oot)
**Output**: Trained pipeline, performance metrics
**Key Parameters**: 
- Sample size: Using all available good samples
- Metrics: AUC, recall, dollar recall

### Task 7: Feature Importance Analysis and Selection
**Cells**: [39-41]
**Summary**: Analyzes feature importance from the initial model and selects the most important features for the final model.
**Input**: Initial model, training data
**Output**: 
- Selected features list
- Feature importance rankings
**Key Parameters**: 
- Feature selection threshold: importance > 0

### Task 8: Final Model Training and Evaluation
**Cells**: [42-59]
**Summary**: Trains the final model using selected features and comprehensively evaluates its performance across different market segments and fulfillment types.
**Input**: 
- Training data with selected features
- Test data with selected features
**Output**:
- Final trained model
- Performance metrics by region and fulfillment type
**Key Parameters**: Same XGBoost parameters as initial model

### Task 9: Model Persistence
**Cells**: [60-62]
**Summary**: Saves the trained model, feature importance data, and selected feature list for deployment.
**Input**: Trained pipeline, feature importance data, selected features
**Output**:
- Saved model file: "xgb_model_na_0702.pkl"
- Feature importance CSV: "feature_importance_na_0702.csv"
- Selected features file: "feature_selected_na_0702.sig"
**Key Parameters**: None

## Task-to-Pipeline-Step Mapping

| Task | Pipeline Step Type | Notes |
|------|-------------------|-------|
| Environment Setup | Base | This is typically handled outside the pipeline or as pipeline initialization |
| Data Loading | CradleDataLoading | Would need to adapt to load from the specific S3 paths |
| Data Preprocessing | TabularPreprocessing | Would handle timestamp conversion, cleaning, and other preprocessing tasks |
| Feature Selection (initial) | TabularPreprocessing | Can be incorporated into preprocessing step |
| Model Training (initial) | XGBoostTraining | This would use the core XGBoost parameters defined in the notebook |
| Feature Importance Analysis | XGBoostModelEval | Analysis of feature importance could be part of model evaluation |
| Final Model Training | XGBoostTraining | Second training run with selected features |
| Model Evaluation | XGBoostModelEval | Comprehensive evaluation across regions and fulfillment types |
| Model Persistence | Registration | Saving model and metadata for deployment |

## Proposed Pipeline Structure

```
CradleDataLoading
    → TabularPreprocessing (includes data cleaning and feature selection)
        → XGBoostTraining (initial model)
            → XGBoostModelEval (feature importance analysis)
                → TabularPreprocessing (feature filtering)
                    → XGBoostTraining (final model)
                        → XGBoostModelEval (comprehensive evaluation)
                            → Registration (save model and metadata)
```

## New Pipeline Step Recommendations

Based on the analysis of the notebook tasks and available pipeline steps, the following new pipeline steps should be considered for development:

1. **Feature Selection Step**: While feature selection can be incorporated into the TabularPreprocessing step, the notebook demonstrates a sophisticated feature importance-based selection process that may warrant its own dedicated step. This step would:
   - Take a trained model as input
   - Extract feature importance scores
   - Apply threshold-based feature filtering
   - Output a selected feature list for downstream steps
   - Key Parameters: importance threshold, selection strategy

2. **Regional Evaluation Step**: The notebook performs evaluation across different marketplace regions (US, CA, MX, BR) and fulfillment types. A specialized evaluation step that supports:
   - Segmented evaluation by region/category
   - Comparative metrics against baseline models
   - Customizable metrics (AUC, recall, dollar recall)
   - Visualization capabilities for regional comparisons

3. **Multi-Stage Training Orchestrator**: The notebook employs a two-stage approach (train initial model → select features → train final model). A specialized orchestration step could:
   - Manage the workflow between multiple training iterations
   - Handle state transfer between training stages
   - Optimize hyperparameters between stages
   - Implement early stopping based on feature importance stability

These new pipeline steps would provide more specialized support for the sophisticated modeling approach demonstrated in the notebook and make future pipeline implementations more maintainable and reusable.

## Implementation Considerations

1. **Data Sources**: The notebook hardcodes file paths. The pipeline would need to parameterize these to read from S3 buckets.

2. **Feature Engineering**: The notebook uses custom functions for feature transformation. These would need to be encapsulated in the TabularPreprocessing step.

3. **Model Parameters**: XGBoost parameters should be extracted into a configuration that can be passed to the XGBoostTraining step.

4. **Evaluation Strategy**: The notebook performs extensive evaluations by region and fulfillment type. The pipeline might need custom evaluation logic beyond standard XGBoostModelEval capabilities.

5. **Multi-step Training**: The notebook uses a two-stage approach (train, select features, train again). This would require coordination between multiple pipeline steps.

6. **Custom Functions**: The notebook defines many custom functions that would need to be incorporated into the appropriate pipeline steps or provided as helper scripts.

7. **Output Artifacts**: The pipeline would need to organize and store outputs (model, feature lists, metrics) in a structured way for tracking and deployment.
