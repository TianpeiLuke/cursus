# Notebook Digest Prompt Templates

This directory contains prompt templates for using Claude 3.7 to analyze Jupyter notebooks and transform them into SageMaker pipelines.

## Available Templates

### 1. `jupyter_to_sagemaker_pipeline_analyzer.md`
A detailed prompt template with instructions for analyzing a Jupyter notebook and mapping its tasks to SageMaker pipeline steps.

### 2. `claude_notebook_analyzer_template.md`
A structured template specifically designed for Claude 3.7 that includes a system prompt, user input format, and comprehensive instructions.

### 3. `example_digest_output.md`
An example of the expected output format, showing how a notebook analysis should be structured.

## Usage

1. Select the appropriate template based on your needs.

2. Replace the placeholder variables:
   - `{notebook_path}` - Path to the Jupyter notebook to analyze
   - `{available_step_types}` - List of available pipeline step types from your system
   - `{output_path}` - Destination path for the analysis output

3. Submit the completed prompt to Claude 3.7.

4. The LLM will analyze the notebook and produce a comprehensive digest that:
   - Breaks down the notebook into logical task groups
   - Summarizes each task's purpose and requirements
   - Maps tasks to appropriate SageMaker pipeline steps
   - Proposes an overall pipeline structure
   - Recommends new pipeline steps that should be created
   - Identifies implementation considerations

5. Review the output and use it as a guide for implementing your SageMaker pipeline.

## Example

```
I need to transform the Jupyter notebook at "dockers/xgboost_pda/model_training_pda.ipynb" into a SageMaker pipeline. Please analyze the notebook, break it down into task groups, and help me map these tasks to the appropriate pipeline steps defined in our system.

Our available pipeline step types are:
- Base
- TabularPreprocessing
- CradleDataLoading
- XGBoostTraining
- XGBoostModel
- XGBoostModelEval
- Registration

Please write your detailed analysis to the following output location:
slipbox/notebook_digests/xgboost_pda_notebook_digest.md
```

## Customization

You can customize these templates to fit your specific needs by:

- Adding project-specific context or requirements
- Modifying the output format
- Including additional instructions for handling specialized code patterns
- Adding examples of your pipeline architecture
