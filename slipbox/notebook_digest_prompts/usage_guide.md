# Usage Guide: Choosing the Right Template

## Claude 3.7 Template Options

This directory contains two main templates for analyzing Jupyter notebooks with Claude 3.7:

### 1. `claude_notebook_analyzer_template.md` (Recommended)

**Recommended for:** Direct use with Claude 3.7 Sonnet or Opus models

This is the primary template you should use with Claude coding agents. It's specifically designed for Claude 3.7 and includes:
- A complete system prompt that sets Claude in the right context
- A structured user input template
- Detailed step-by-step instructions
- Example inputs/outputs

**How to use:**
1. Copy the entire content of this file
2. Paste it into your conversation with Claude
3. Fill in the placeholders with your specific notebook path and step types
4. Send the prompt to Claude

### 2. `jupyter_to_sagemaker_pipeline_analyzer.md`

**Use case:** Simplified prompt or for adaptation into other systems

This is a more simplified version that focuses on the core task description and instructions. It's suitable if you:
- Want a more concise prompt
- Are integrating with other systems
- Need to modify the instructions significantly

## Recommendation

For most users working directly with Claude, use the **`claude_notebook_analyzer_template.md`** file as your starting point. It provides the most complete guidance to Claude and will likely produce the most consistent results.

## Example Usage with Claude

```
[Paste the entire content of claude_notebook_analyzer_template.md here]

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

The output will follow the structure shown in `example_digest_output.md`.
