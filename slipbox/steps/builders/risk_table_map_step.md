# Risk Table Mapping Step

## Task Summary
The Risk Table Mapping Step processes raw data and applies risk table mappings to prepare it for model training. This step:

1. Pulls raw dataset (sharded CSVs) from the specified S3 location
2. Loads configuration files including risk table definitions
3. Applies risk table mappings to convert categorical variables to numerical values
4. Optionally performs currency conversion if enabled
5. Generates pickle files containing bin mappings, missing value imputation rules, and configuration

## Input and Output Format

### Input
- **Raw Data**: Sharded CSV files in gzip format from S3 (under data_type subfolder)
- **Config Files**: Configuration files including config.json and metadata.csv
- **Optional Currency Table**: Currency conversion table if currency conversion is enabled
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Mapping Files**: Three pickle files stored in S3:
  - bin_mapping.pkl: Contains mappings for categorical variables
  - missing_value_imputation.pkl: Rules for handling missing values
  - config.pkl: Configuration used for the mapping process
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| processing_entry_point | Entry point script for risk table mapping | risk_table_mapping.py |
| processing_source_dir | Directory containing processing scripts | Required |
| use_large_processing_instance | Whether to use large instance type | False |
| input_data_s3_uri | S3 URI for raw data | Required |
| config_s3_uri | S3 URI for config files | Required |
| enable_currency_conversion | Whether to perform currency conversion | False |
| currency_conversion_table_s3_uri | S3 URI for currency conversion table | None (Required if enable_currency_conversion=True) |
| output_s3_uri | S3 URI for output files | Required |
| input_names | Dictionary mapping input names | Default dictionary |
| output_names | Dictionary mapping output names | Default dictionary |

## Validation Rules
- input_data_s3_uri, config_s3_uri, and output_s3_uri must start with 's3://'
- If enable_currency_conversion=True, currency_conversion_table_s3_uri must be provided
- processing_entry_point must be provided
- If processing_source_dir is local, the script must exist at that location

## Usage Example
```python
from src.pipeline_steps.config_risk_table_map_step import RiskTableMappingStepConfig
from src.pipeline_steps.builder_risk_table_map_step import RiskTableMappingStepBuilder

# Create configuration
config = RiskTableMappingStepConfig(
    processing_entry_point="risk_table_mapping.py",
    processing_source_dir="s3://my-bucket/scripts/",
    input_data_s3_uri="s3://my-bucket/raw-data/",
    config_s3_uri="s3://my-bucket/config-files/",
    output_s3_uri="s3://my-bucket/processed-data/",
    enable_currency_conversion=True,
    currency_conversion_table_s3_uri="s3://my-bucket/config-files/currency_conversion_table.csv"
)

# Create builder and step
builder = RiskTableMappingStepBuilder(config=config)
risk_mapping_step = builder.create_step(
    data_input="s3://my-bucket/raw-data/",
    config_input="s3://my-bucket/config-files/",
    data_type="training",
    dependencies=[previous_step]
)

# Add to pipeline
pipeline.add_step(risk_mapping_step)
```

## Processing Inputs and Outputs

### Processing Inputs
- **data_input**: Raw data input (destination: /opt/ml/processing/input/data)
- **config_input**: Configuration files input (destination: /opt/ml/processing/input/config)
- **currency_conversion_table**: Optional currency conversion table (destination: /opt/ml/processing/input/config/currency_conversion_table.csv)

### Processing Outputs
- **risk_mapping**: Output for mapping files (source: /opt/ml/processing/output, destination: {output_s3_uri}/risk_table_mapping/)

## Integration with Pipeline Builder Template

### Input Arguments

The `RiskTableMappingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| data_input | Raw data input location | Yes | Previous step's data output |
| config_input | Configuration files location | Yes | Configured in the step builder |
| data_type | Type of data to process | Yes | Configured in the step builder |

### Output Properties

The `RiskTableMappingStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| risk_mapping | Risk mapping files location | `step.properties.ProcessingOutputConfig.Outputs["risk_mapping"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("risk_mapping")
dag.add_node("preprocess")
dag.add_edge("data_load", "risk_mapping")
dag.add_edge("risk_mapping", "preprocess")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "risk_mapping": risk_mapping_config,
    "preprocess": preprocess_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "RiskTableMappingStep": RiskTableMappingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
}

# Create the template
template = PipelineBuilderTemplate(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=sagemaker_session,
    role=role,
)

# Generate the pipeline
pipeline = template.generate_pipeline("my-pipeline")
```

For more details on how the Pipeline Builder Template handles connections between steps, see the [Pipeline Builder documentation](../pipeline_builder/README.md).
