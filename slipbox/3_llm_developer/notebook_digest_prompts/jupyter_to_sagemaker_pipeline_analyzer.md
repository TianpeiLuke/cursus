# Jupyter Notebook to SageMaker Pipeline Analyzer

## Task Description
You will analyze a Jupyter notebook and extract the key tasks it performs. Then, you'll compare these tasks with available SageMaker pipeline step types to help the user transform the notebook into a proper SageMaker pipeline.

## Input
Notebook path: {notebook_path}

## Instructions

1. First, read the entire notebook content carefully.

2. Divide the notebook into logical task groups. Each group should accomplish a specific function or purpose.

3. For each task group:
   - Identify the code cells that belong to this group
   - Summarize what the code accomplishes in 1-3 sentences
   - Identify input data sources and output artifacts
   - Note any key parameters or configurations

4. After identifying all task groups, compare each task to the following available pipeline step types:
   
   - Base steps
   - Processing steps (TabularPreprocessing, CurrencyConversion)
   - Data loading steps (CradleDataLoading)
   - Training steps (PytorchTraining, XGBoostTraining, DummyTraining)
   - Model creation steps (PytorchModel, XGBoostModel)
   - Evaluation steps (XGBoostModelEval, PytorchModelEval)
   - Deployment steps (Package, Registration, Payload)
   - Transform steps (BatchTransform)
   - Utility steps (HyperparameterPrep)

5. For each task group, suggest:
   - Which pipeline step type(s) would be most appropriate
   - What modifications might be needed to fit the standardized pipeline architecture
   - Any parameters that would need to be extracted from the notebook

6. Provide a complete summary table that shows:
   - Each identified task
   - The appropriate pipeline step type
   - Key parameters/configurations
   - Input data sources
   - Output artifacts

7. Conclude with a suggested overall pipeline structure showing the sequence of steps.

## Output Format
Your output should be a markdown document with these sections:

1. **Notebook Summary**: Brief overview of what the notebook does
2. **Task Breakdown**: List of identified tasks with details
3. **Task-to-Pipeline-Step Mapping**: Map tasks to appropriate SageMaker pipeline steps
4. **Proposed Pipeline Structure**: Suggested pipeline architecture
5. **New Pipeline Step Recommendations**: Suggestions for new pipeline steps that should be created based on gaps identified
6. **Implementation Considerations**: Any special considerations or challenges

The output should be detailed enough to serve as a reference for implementing the SageMaker pipeline.
