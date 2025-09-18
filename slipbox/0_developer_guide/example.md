# Example: Adding a Feature Selection Step

This document provides a complete example of adding a new step to the pipeline. We'll implement a `FeatureSelectionStep` that selects the most important features from preprocessed data.

## Step Overview

Our new step will:
1. Take preprocessed data as input
2. Select the most important features using various methods (correlation, mutual information, etc.)
3. Output the reduced feature set and feature importance metadata

This step will fit between data preprocessing and model training in the pipeline.

## Prerequisites

First, let's ensure we have all the required information:

- **Task Description**: A step that selects the most important features from preprocessed data
- **Step Name**: `FeatureSelection`
- **Node Type**: `INTERNAL` (has both inputs and outputs)
- **SageMaker Component Type**: `ProcessingStep`
- **Step Type**: Processing step (uses processing containers, not training containers)

## Step-by-Step Implementation

This example demonstrates the complete process of adding a new step to the pipeline system, following the updated creation process that includes script creation as an explicit step.

### Step 1: Set Up Workspace Context

First, determine your development approach and set up the appropriate workspace context:

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Set main workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("main")
```

### Step 2: Create the Step Configuration

Let's create the configuration class for our step using the three-tier field classification design.

**File**: `src/cursus/steps/configs/config_feature_selection.py`

```python
from pydantic import BaseModel, Field, field_validator, PrivateAttr, ConfigDict
from typing import Dict, Any, Optional
from ...core.base.config_base import BasePipelineConfig

class FeatureSelectionConfig(BasePipelineConfig):
    """
    Configuration for Feature Selection step using three-tier field classification.
    
    Tier 1: Essential fields (required user inputs)
    Tier 2: System fields (with defaults, can be overridden)
    Tier 3: Derived fields (private with property access)
    """
    
    # Tier 1: Essential user inputs (required, no defaults)
    selection_method: str = Field(..., description="Method for feature selection (mutual_info, correlation, tree_based)")
    n_features: int = Field(..., ge=1, description="Number of features to select")
    target_column: str = Field(..., description="Name of the target/label column")
    
    # Tier 2: System inputs with defaults (can be overridden)
    min_importance: float = Field(default=0.01, ge=0.0, description="Minimum importance threshold for features")
    instance_type: str = Field(default="ml.m5.xlarge", description="SageMaker instance type")
    instance_count: int = Field(default=1, ge=1, description="Number of instances")
    volume_size_gb: int = Field(default=30, ge=1, description="EBS volume size in GB")
    max_runtime_seconds: int = Field(default=3600, ge=1, description="Maximum runtime in seconds")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    
    # Tier 3: Derived fields (private with property access)
    _script_path: Optional[str] = PrivateAttr(default=None)
    _output_path: Optional[str] = PrivateAttr(default=None)
    
    # Pydantic v2 model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
        protected_namespaces=(),
    )
    
    @field_validator('selection_method')
    @classmethod
    def validate_selection_method(cls, v: str) -> str:
        """Validate selection method is supported."""
        valid_methods = ["mutual_info", "correlation", "tree_based"]
        if v not in valid_methods:
            raise ValueError(f"Selection method must be one of: {valid_methods}")
        return v
    
    @field_validator('instance_type')
    @classmethod
    def validate_instance_type(cls, v: str) -> str:
        """Validate SageMaker instance type."""
        valid_instances = [
            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
            "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge"
        ]
        if v not in valid_instances:
            raise ValueError(f"Invalid instance type: {v}. Must be one of: {valid_instances}")
        return v
    
    # Public properties for derived fields
    @property
    def script_path(self) -> str:
        """Get script path."""
        if self._script_path is None:
            self._script_path = "feature_selection.py"
        return self._script_path
    
    @property
    def output_path(self) -> str:
        """Get output path."""
        if self._output_path is None:
            self._output_path = f"{self.pipeline_s3_loc}/feature_selection/{self.region}"
        return self._output_path
    
    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["script_path"] = self.script_path
        data["output_path"] = self.output_path
        return data
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..contracts.feature_selection_contract import FEATURE_SELECTION_CONTRACT
        return FEATURE_SELECTION_CONTRACT
```

### Step 3: Create the Script Contract

Create the script contract that defines the interface between our script and the SageMaker container environment.

**File**: `src/cursus/steps/contracts/feature_selection_contract.py`

```python
"""
Feature Selection Script Contract

Defines the contract for the feature selection script that reduces feature dimensionality
from preprocessed tabular data using various selection methods.
"""

from ...core.base.contract_base import ScriptContract

FEATURE_SELECTION_CONTRACT = ScriptContract(
    entry_point="feature_selection.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "config": "/opt/ml/processing/input/config"
    },
    expected_output_paths={
        "selected_features": "/opt/ml/processing/output/features",
        "feature_importance": "/opt/ml/processing/output/importance"
    },
    required_env_vars=[
        "SELECTION_METHOD",
        "N_FEATURES",
        "TARGET_COLUMN"
    ],
    optional_env_vars={
        "MIN_IMPORTANCE": "0.01",
        "RANDOM_SEED": "42",
        "DEBUG_MODE": "False"
    },
    framework_requirements={
        "pandas": ">=1.2.0,<2.0.0",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "numpy": ">=1.19.0",
        "pyarrow": ">=4.0.0,<6.0.0"
    },
    description="""
    Feature selection script for tabular data that:
    1. Loads preprocessed training data from input directory
    2. Applies feature selection using configurable methods (mutual_info, correlation, tree_based)
    3. Selects top N features based on importance scores
    4. Filters features by minimum importance threshold
    5. Outputs reduced feature set and importance rankings
    6. Supports both binary and multiclass classification scenarios
    
    Input Structure:
    - /opt/ml/processing/input/data: Preprocessed data files (.csv, .parquet, .json)
    - /opt/ml/processing/input/config: Optional configuration files (JSON format)
    
    Output Structure:
    - /opt/ml/processing/output/features: Selected features dataset
      - /opt/ml/processing/output/features/selected_features.parquet: Reduced dataset
      - /opt/ml/processing/output/features/feature_list.json: List of selected feature names
    - /opt/ml/processing/output/importance: Feature importance analysis
      - /opt/ml/processing/output/importance/feature_importance.csv: Importance scores
      - /opt/ml/processing/output/importance/importance_plot.json: Visualization data
    
    Contract aligned with step specification:
    - Inputs: input_data (required), config (optional)
    - Outputs: selected_features (primary), feature_importance (secondary)
    
    Environment Variables:
    - SELECTION_METHOD: Feature selection method (mutual_info, correlation, tree_based)
    - N_FEATURES: Number of top features to select
    - TARGET_COLUMN: Name of the target/label column in the dataset
    - MIN_IMPORTANCE: Minimum importance threshold for feature filtering (optional)
    - RANDOM_SEED: Random seed for reproducible results (optional)
    - DEBUG_MODE: Enable debug logging (optional)
    """
)
```

### Step 4: Create the Processing Script

Now let's create the actual processing script that implements our business logic using the unified main function interface:

**File**: `src/cursus/steps/scripts/feature_selection.py`

```python
#!/usr/bin/env python3
"""
Processing script for Feature Selection Step.

This script implements the business logic for feature selection using the unified
main function interface for testability and SageMaker container compatibility.
"""
import os
import sys
import argparse
import logging
import traceback
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.feature_selection import (
    mutual_info_classif, 
    f_classif, 
    SelectKBest,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# Container path constants
CONTAINER_PATHS = {
    "PROCESSING_INPUT_BASE": "/opt/ml/processing/input",
    "PROCESSING_OUTPUT_BASE": "/opt/ml/processing/output",
    "DATA_INPUT": "/opt/ml/processing/input/data",
    "CONFIG_INPUT": "/opt/ml/processing/input/config",
    "FEATURES_OUTPUT": "/opt/ml/processing/output/features",
    "IMPORTANCE_OUTPUT": "/opt/ml/processing/output/importance"
}

def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    sys.stdout.flush()
    return logger

def read_input_data(input_path: str, logger) -> pd.DataFrame:
    """Read input data from the specified path."""
    try:
        logger(f"Reading input data from {input_path}")
        
        # Check if directory exists
        if not os.path.exists(input_path):
            raise ValueError(f"Input path does not exist: {input_path}")
            
        # List files in directory
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.endswith('.csv') or f.endswith('.parquet')]
        
        if not files:
            raise ValueError(f"No CSV or Parquet files found in {input_path}")
        
        # Read the first file
        file_path = files[0]
        logger(f"Reading file: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger(f"Read {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger(f"Error reading input data: {str(e)}")
        raise

def select_features_mutual_info(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_features: int, 
    random_state: int,
    logger
) -> Tuple[List[str], pd.DataFrame]:
    """Select features using mutual information."""
    logger(f"Selecting {n_features} features using mutual information")
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_processed = X.copy()
    
    for col in categorical_cols:
        X_processed[col] = X_processed[col].astype('category').cat.codes
    
    # Apply mutual information
    selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    selector.fit(X_processed, y)
    
    # Get selected features and importance scores
    feature_idx = selector.get_support(indices=True)
    feature_names = X.columns[feature_idx].tolist()
    importance_scores = selector.scores_[feature_idx]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    logger(f"Selected {len(feature_names)} features")
    return feature_names, importance_df

def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger=None
) -> Dict[str, Any]:
    """
    Main processing function with unified interface.
    
    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger function
    
    Returns:
        Processing results dictionary
    """
    log = logger or print
    
    # Extract parameters from environment variables (aligned with contract)
    selection_method = environ_vars.get("SELECTION_METHOD")
    n_features = int(environ_vars.get("N_FEATURES", "10"))
    target_column = environ_vars.get("TARGET_COLUMN")
    min_importance = float(environ_vars.get("MIN_IMPORTANCE", "0.01"))
    random_seed = int(environ_vars.get("RANDOM_SEED", "42"))
    
    # Extract paths (aligned with contract)
    input_data_dir = input_paths.get("input_data")
    config_dir = input_paths.get("config")
    output_features_dir = output_paths.get("selected_features")
    output_importance_dir = output_paths.get("feature_importance")
    
    log(f"Starting feature selection with parameters:")
    log(f"  Selection method: {selection_method}")
    log(f"  Number of features: {n_features}")
    log(f"  Target column: {target_column}")
    log(f"  Input data directory: {input_data_dir}")
    log(f"  Output features directory: {output_features_dir}")
    log(f"  Output importance directory: {output_importance_dir}")
    
    # Validate required parameters
    if not selection_method:
        raise ValueError("SELECTION_METHOD environment variable is required")
    if not target_column:
        raise ValueError("TARGET_COLUMN environment variable is required")
    
    # Create output directories
    Path(output_features_dir).mkdir(parents=True, exist_ok=True)
    Path(output_importance_dir).mkdir(parents=True, exist_ok=True)
    
    # Your business logic implementation here
    try:
        log("Starting feature selection processing...")
        
        # Read input data
        df = read_input_data(input_data_dir, log)
        
        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data columns: {list(df.columns)}")
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Select features based on method
        if selection_method == "mutual_info":
            feature_names, importance_df = select_features_mutual_info(X, y, n_features, random_seed, log)
        else:
            raise ValueError(f"Unsupported selection method: {selection_method}")
        
        # Filter by importance threshold if specified
        if min_importance > 0:
            filtered_df = importance_df[importance_df['importance'] >= min_importance]
            feature_names = filtered_df['feature'].tolist()
            importance_df = filtered_df
            log(f"Filtered to {len(feature_names)} features with importance >= {min_importance}")
        
        # Create selected features dataset
        selected_df = df[feature_names + [target_column]]
        
        # Write selected features
        features_file = Path(output_features_dir) / "selected_features.parquet"
        selected_df.to_parquet(features_file, index=False)
        log(f"Saved selected features to: {features_file}")
        
        # Write feature list
        feature_list_file = Path(output_features_dir) / "feature_list.json"
        with open(feature_list_file, "w") as f:
            json.dump(feature_names, f, indent=2)
        
        # Write importance scores
        importance_file = Path(output_importance_dir) / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        log(f"Saved importance scores to: {importance_file}")
        
        # Write visualization data
        viz_file = Path(output_importance_dir) / "importance_plot.json"
        viz_data = {
            'features': importance_df['feature'].tolist(),
            'importance': importance_df['importance'].tolist()
        }
        with open(viz_file, "w") as f:
            json.dump(viz_data, f, indent=2)
        
        log(f"Feature selection completed successfully. Selected {len(feature_names)} features.")
        
        return {
            "status": "success",
            "selected_features_count": len(feature_names),
            "features_file": str(features_file),
            "importance_file": str(importance_file)
        }
        
    except Exception as e:
        log(f"Error during feature selection: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Set up logging
        logger = setup_logging()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Feature Selection processing script")
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        args = parser.parse_args()
        
        # Set up paths using container defaults (aligned with contract)
        input_paths = {
            "input_data": CONTAINER_PATHS["DATA_INPUT"],
            "config": CONTAINER_PATHS["CONFIG_INPUT"]
        }
        
        output_paths = {
            "selected_features": CONTAINER_PATHS["FEATURES_OUTPUT"],
            "feature_importance": CONTAINER_PATHS["IMPORTANCE_OUTPUT"]
        }
        
        # Collect environment variables (aligned with contract)
        environ_vars = {
            "SELECTION_METHOD": os.environ.get("SELECTION_METHOD"),
            "N_FEATURES": os.environ.get("N_FEATURES"),
            "TARGET_COLUMN": os.environ.get("TARGET_COLUMN"),
            "MIN_IMPORTANCE": os.environ.get("MIN_IMPORTANCE", "0.01"),
            "RANDOM_SEED": os.environ.get("RANDOM_SEED", "42"),
            "DEBUG_MODE": os.environ.get("DEBUG_MODE", "False")
        }
        
        # Validate required environment variables
        required_vars = ["SELECTION_METHOD", "TARGET_COLUMN"]
        missing_vars = [var for var in required_vars if not environ_vars.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Set debug mode if requested
        if args.debug or environ_vars.get("DEBUG_MODE", "False").lower() == "true":
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        
        # Log startup information
        logger.info("Starting Feature Selection processing script")
        logger.info(f"Input paths: {input_paths}")
        logger.info(f"Output paths: {output_paths}")
        logger.info(f"Environment variables: {list(environ_vars.keys())}")
        
        # Execute main processing
        result = main(input_paths, output_paths, environ_vars, args, logger.info)
        
        # Create success marker
        success_path = Path(output_paths["selected_features"]) / "_SUCCESS"
        success_path.touch()
        logger.info(f"Created success marker: {success_path}")
        
        logger.info("Script completed successfully")
        logger.info(f"Processing result: {result}")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Script failed with error: {e}")
        logging.error(traceback.format_exc())
        
        # Create failure marker
        try:
            failure_path = Path(output_paths.get("selected_features", "/tmp")) / "_FAILURE"
            with open(failure_path, "w") as f:
                f.write(f"Error: {str(e)}")
            logging.error(f"Created failure marker: {failure_path}")
        except:
            pass
        
        sys.exit(1)
```

### Step 5: Create the Step Specification

Now, let's create the step specification that defines how our step connects with other steps in the pipeline. For processing steps, we create a single specification file that defines the step's dependencies and outputs.

**File**: `src/cursus/steps/specs/feature_selection_spec.py`

```python
"""
Feature Selection Step Specification.

This module defines the declarative specification for feature selection steps,
including their dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ...registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_feature_selection_contract():
    from ..contracts.feature_selection_contract import FEATURE_SELECTION_CONTRACT
    return FEATURE_SELECTION_CONTRACT

# Feature Selection Step Specification
FEATURE_SELECTION_SPEC = StepSpecification(
    step_type=get_spec_step_type("FeatureSelection"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_feature_selection_contract(),
    dependencies={
        "input_data": DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "CradleDataLoading"],
            semantic_keywords=["data", "processed", "tabular", "features", "preprocessing", "input"],
            data_type="S3Uri",
            description="Preprocessed tabular data for feature selection"
        ),
        "config": DependencySpec(
            logical_name="config",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ProcessingStep", "ConfigGeneration"],
            semantic_keywords=["config", "parameters", "settings", "hyperparameters"],
            data_type="S3Uri",
            description="Optional configuration parameters for feature selection"
        )
    },
    outputs={
        "selected_features": OutputSpec(
            logical_name="selected_features",
            aliases=["features", "reduced_features", "feature_subset", "processed_features"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selected_features'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Selected subset of features with reduced dimensionality"
        ),
        "feature_importance": OutputSpec(
            logical_name="feature_importance",
            aliases=["importance", "feature_rankings", "feature_scores"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['feature_importance'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Feature importance scores and rankings from selection process"
        )
    }
)
```

### Step 6: Create the Step Builder

Now, let's implement the builder class that creates the SageMaker step using the real patterns from the Cursus codebase.

**File**: `src/cursus/steps/builders/builder_feature_selection.py`

```python
from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ...core.base.builder_base import StepBuilderBase
from ..configs.config_feature_selection import FeatureSelectionConfig
from ..specs.feature_selection_spec import FEATURE_SELECTION_SPEC
from ...registry.hybrid.manager import UnifiedRegistryManager

class FeatureSelectionStepBuilder(StepBuilderBase):
    """
    Builder for Feature Selection Processing Step.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that performs feature selection on preprocessed data.
    """

    def __init__(
        self,
        config: FeatureSelectionConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None
    ):
        """
        Initializes the builder with a specific configuration for the processing step.

        Args:
            config: A FeatureSelectionConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, FeatureSelectionConfig):
            raise ValueError("FeatureSelectionStepBuilder requires a FeatureSelectionConfig instance.")
            
        # Initialize with specification
        super().__init__(
            config=config,
            spec=FEATURE_SELECTION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: FeatureSelectionConfig = config

    def validate_configuration(self) -> None:
        """Validate the provided configuration."""
        required_attrs = [
            'selection_method', 'n_features', 'target_column',
            'instance_type', 'instance_count'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"FeatureSelectionConfig missing required attribute: {attr}")

    def _create_processor(self):
        """Create and return a SageMaker processor."""
        from sagemaker.sklearn import SKLearnProcessor
        
        return SKLearnProcessor(
            framework_version="0.23-1",
            role=self.role,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            volume_size_in_gb=self.config.volume_size_gb,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            env=self._get_environment_variables()
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor."""
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()
        
        # Add step-specific environment variables
        step_env_vars = {
            "SELECTION_METHOD": self.config.selection_method,
            "N_FEATURES": str(self.config.n_features),
            "TARGET_COLUMN": self.config.target_column,
            "MIN_IMPORTANCE": str(self.config.min_importance),
            "RANDOM_SEED": str(self.config.random_seed),
            "DEBUG_MODE": str(self.config.debug_mode).lower()
        }
        
        env_vars.update(step_env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the step using specification and contract."""
        if not self.spec or not self.contract:
            raise ValueError("Step specification and contract are required")
            
        processing_inputs = []
        
        # Process each dependency in the specification
        for logical_name, dependency_spec in self.spec.dependencies.items():
            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue
                
            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            
            # Get container path from contract
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
                processing_inputs.append(ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path
                ))
            else:
                raise ValueError(f"No container path found for input: {logical_name}")
                
        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the step using specification and contract with consistent folder structure."""
        if not self.spec or not self.contract:
            raise ValueError("Step specification and contract are required")
            
        processing_outputs = []
        
        # Get the base output path (using PIPELINE_EXECUTION_TEMP_DIR if available)
        base_output_path = self._get_base_output_path()
        
        # Process each output in the specification
        for logical_name, output_spec in self.spec.outputs.items():
            # Get container path from contract
            if logical_name in self.contract.expected_output_paths:
                container_path = self.contract.expected_output_paths[logical_name]
                
                # Generate destination using consistent folder structure
                if logical_name in outputs:
                    destination = outputs[logical_name]
                else:
                    # Generate consistent output path using Join pattern
                    from sagemaker.workflow.functions import Join
                    step_type = self.spec.step_type.lower() if hasattr(self.spec, 'step_type') else 'feature_selection'
                    destination = Join(on="/", values=[base_output_path, step_type, logical_name])
                    self.log_info(
                        "Generated consistent output path for '%s': %s",
                        logical_name,
                        destination,
                    )
                
                processing_outputs.append(ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination
                ))
            else:
                raise ValueError(f"No container path found for output: {logical_name}")
                
        return processing_outputs

    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the ProcessingStep."""
        # Extract parameters
        inputs_raw = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Handle inputs
        inputs = {}
        
        # If dependencies are provided, extract inputs from them
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
                
        # Add explicitly provided inputs
        inputs.update(inputs_raw)
        
        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        
        # Get step name and script path
        step_name = self._get_step_name()
        script_path = self.config.get_script_path()
        
        # Create step
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        
        # Attach specification to the step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

### Step 7: Register Step with Hybrid Registry System

With the modern hybrid registry system, step registration is handled automatically through the UnifiedRegistryManager. However, you need to ensure your step is properly registered:

#### 7.1 Option A: Automatic Registration (Recommended)

The UnifiedRegistryManager automatically discovers and registers your step if you follow the naming conventions:

1. **File Naming**: Your builder file should follow the pattern `builder_feature_selection.py`
2. **Class Naming**: Your builder class should be named `FeatureSelectionStepBuilder`
3. **Location**: Place your builder in `src/cursus/steps/builders/`

The registry will automatically:
- Discover your step builder
- Extract step metadata from your configuration and specification
- Register the step with the appropriate workspace context

#### 7.2 Option B: Explicit Registration (For Custom Cases)

If you need explicit control over registration, use the registry's validation-enabled registration:

```python
from cursus.registry.step_names import add_new_step_with_validation

# Register your step with validation
warnings = add_new_step_with_validation(
    step_name="FeatureSelection",
    config_class="FeatureSelectionConfig", 
    builder_name="FeatureSelectionStepBuilder",
    sagemaker_type="Processing",  # Based on create_step() return type
    description="Feature selection step for dimensionality reduction",
    validation_mode="warn",  # Options: "warn", "strict", "auto_correct"
    workspace_id=None  # Use current workspace context
)

# Check for any validation warnings
if warnings:
    for warning in warnings:
        print(f"⚠️ {warning}")
```

### Step 8: Run Validation Framework Tests

Before proceeding with unit tests, run the comprehensive validation framework to ensure your step implementation is correct.

#### 8.1 Unified Alignment Tester

Execute the **Unified Alignment Tester** to perform 4-tier validation:

```bash
# Validate a specific script with detailed output
python -m cursus.cli.alignment_cli validate feature_selection --verbose --show-scoring

# Generate visualization and scoring reports
python -m cursus.cli.alignment_cli visualize feature_selection --output-dir ./validation_reports --verbose
```

#### 8.2 Universal Step Builder Test

Execute the **Universal Step Builder Test** for comprehensive builder testing:

```bash
# Run all tests for your builder with scoring
python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_feature_selection.FeatureSelectionStepBuilder --scoring --verbose

# Test all builders of Processing type
python -m cursus.cli.builder_test_cli test-by-type Processing --verbose --scoring
```

#### 8.3 Script Runtime Testing

Execute the **Script Runtime Tester** for actual script execution validation:

```bash
# Test single script functionality
cursus runtime test-script feature_selection --workspace-dir ./test_workspace --verbose

# Test data compatibility between connected scripts
cursus runtime test-compatibility tabular_preprocessing feature_selection --workspace-dir ./test_workspace --verbose

# Test complete pipeline flow
cursus runtime test-pipeline pipeline_with_feature_selection.json --workspace-dir ./test_workspace --verbose
```

### Step 9: Create Unit Tests

Implement tests to verify your components work correctly:

**File**: `test/steps/builders/test_builder_feature_selection.py`

```python
import unittest
from unittest.mock import MagicMock, patch

from cursus.steps.builders.builder_feature_selection import FeatureSelectionStepBuilder
from cursus.steps.configs.config_feature_selection import FeatureSelectionConfig
from cursus.steps.specs.feature_selection_spec import FEATURE_SELECTION_SPEC
from cursus.core.base.specification_base import NodeType, DependencyType

class TestFeatureSelectionStepBuilder(unittest.TestCase):
    def setUp(self):
        self.config = FeatureSelectionConfig(
            region="us-west-2",
            pipeline_s3_loc="s3://bucket/prefix",
            selection_method="mutual_info",
            n_features=20,
            target_column="target"
        )
        self.builder = FeatureSelectionStepBuilder(self.config)
    
    def test_initialization(self):
        """Test that the builder initializes correctly with specification."""
        self.assertIsNotNone(self.builder.spec)
        self.assertEqual(self.builder.spec.step_type, FEATURE_SELECTION_SPEC.step_type)
        self.assertEqual(self.builder.spec.node_type, NodeType.INTERNAL)
    
    def test_get_inputs(self):
        """Test that inputs are correctly derived from dependencies."""
        inputs = {
            "input_data": "s3://bucket/input/data",
            "config": "s3://bucket/input/config"
        }
        
        processing_inputs = self.builder._get_inputs(inputs)
        
        self.assertEqual(len(processing_inputs), 2)
        self.assertEqual(processing_inputs[0].source, "s3://bucket/input/data")
        self.assertEqual(processing_inputs[1].source, "s3://bucket/input/config")
    
    def test_get_outputs(self):
        """Test that outputs are correctly configured."""
        processing_outputs = self.builder._get_outputs({})
        
        self.assertEqual(len(processing_outputs), 2)
        output_names = [output.output_name for output in processing_outputs]
        self.assertIn("selected_features", output_names)
        self.assertIn("feature_importance", output_names)
    
    def test_get_environment_variables(self):
        """Test that environment variables are correctly set."""
        env_vars = self.builder._get_environment_variables()
        
        self.assertEqual(env_vars["SELECTION_METHOD"], "mutual_info")
        self.assertEqual(env_vars["N_FEATURES"], "20")
        self.assertEqual(env_vars["TARGET_COLUMN"], "target")
    
    @patch('cursus.steps.builders.builder_feature_selection.FeatureSelectionStepBuilder._create_processor')
    def test_create_step(self, mock_create_processor):
        """Test step creation with dependencies."""
        # Mock processor
        mock_processor = MagicMock()
        mock_create_processor.return_value = mock_processor
        
        # Create step
        inputs = {"input_data": "s3://bucket/input/data"}
        step = self.builder.create_step(inputs=inputs, step_name="TestStep")
        
        # Verify step was created
        self.assertIsNotNone(step)
        self.assertTrue(hasattr(step, '_spec'))

if __name__ == '__main__':
    unittest.main()
```

### Step 10: Integrate With Pipeline Templates

Once your step is created and validated, it becomes available for use in the Pipeline Catalog system. The key is to follow the established patterns from the existing pipeline catalog.

#### 10.1 Create Shared DAG Definition (Optional)

First, create a shared DAG definition that can be reused across different pipeline variants:

**File**: `src/cursus/pipeline_catalog/shared_dags/sklearn/feature_selection_training_dag.py`

```python
"""
Shared DAG definition for Feature Selection Training pipeline.

This DAG definition can be used by both regular and MODS compilers,
ensuring consistency while avoiding code duplication.
"""

import logging
from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_feature_selection_training_dag() -> PipelineDAG:
    """
    Create a feature selection training pipeline DAG.
    
    This DAG represents a training workflow with feature selection:
    1) Data Loading (training)
    2) Preprocessing (training) 
    3) Feature Selection
    4) Model Training
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("CradleDataLoading_training")       # Data load for training
    dag.add_node("TabularPreprocessing_training")    # Tabular preprocessing for training
    dag.add_node("FeatureSelection")                 # Our new feature selection step
    dag.add_node("SKLearnTraining")                  # SKLearn training step
    
    # Create pipeline flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "FeatureSelection")
    dag.add_edge("FeatureSelection", "SKLearnTraining")
    
    logger.debug(f"Created feature selection training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the feature selection training DAG definition.
    
    Returns:
        DAGMetadata: Metadata including description, complexity, features
    """
    metadata = DAGMetadata(
        description="Feature selection training pipeline with data loading, preprocessing, feature selection, and model training",
        complexity="standard",
        features=["training", "feature_selection", "preprocessing", "data_loading"],
        framework="sklearn",
        node_count=4,
        edge_count=3,
        extra_metadata={
            "name": "feature_selection_training",
            "task_type": "training",
            "entry_points": ["CradleDataLoading_training"],
            "exit_points": ["SKLearnTraining"],
            "required_configs": [
                "CradleDataLoading_training",
                "TabularPreprocessing_training",
                "FeatureSelection",
                "SKLearnTraining"
            ]
        }
    )
    
    return metadata
```

#### 10.2 Create Pipeline Template

Now create the main pipeline template that uses the shared DAG:

**File**: `src/cursus/pipeline_catalog/pipelines/feature_selection_training_pipeline.py`

```python
"""
Feature Selection Training Pipeline

This pipeline implements a training workflow with feature selection:
1) Data Loading (training)
2) Preprocessing (training)
3) Feature Selection
4) Model Training

This demonstrates how to integrate custom feature selection steps
into a complete ML training pipeline.

Example:
    from cursus.pipeline_catalog.pipelines.feature_selection_training_pipeline import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create the pipeline
    pipeline, report, dag_compiler = create_pipeline(
        config_path="path/to/config.json",
        session=pipeline_session,
        role=role
    )
    
    # Execute the pipeline
    pipeline.upsert()
    execution = pipeline.start()
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.sklearn.feature_selection_training_dag import create_feature_selection_training_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """
    Get enhanced DAG metadata with Zettelkasten integration for feature selection training pipeline.
    
    Returns:
        EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
    """
    # Create Zettelkasten metadata with comprehensive properties
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="feature_selection_training_pipeline",
        title="Feature Selection Training Pipeline",
        single_responsibility="ML training pipeline with automated feature selection",
        input_interface=["Training dataset path", "Feature selection parameters", "Model hyperparameters"],
        output_interface=["Trained model artifact", "Selected features metadata"],
        side_effects="Creates model artifacts and feature selection results in S3",
        independence_level="fully_self_contained",
        node_count=4,
        edge_count=3,
        framework="sklearn",
        complexity="standard",
        use_case="ML training with feature selection",
        features=["training", "feature_selection", "preprocessing"],
        mods_compatible=False,
        source_file="pipelines/feature_selection_training_pipeline.py",
        migration_source="custom_development",
        created_date="2025-09-06",
        priority="medium",
        framework_tags=["sklearn", "sagemaker"],
        task_tags=["training", "feature_selection", "preprocessing"],
        complexity_tags=["standard", "intermediate"],
        domain_tags=["machine_learning", "feature_engineering"],
        pattern_tags=["atomic_workflow", "independent"],
        integration_tags=["sagemaker", "s3"],
        quality_tags=["production_ready", "tested"],
        data_tags=["tabular", "structured"],
        creation_context="Custom feature selection integration for ML training",
        usage_frequency="medium",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="20-45 minutes",
        resource_requirements="ml.m5.xlarge or equivalent",
        use_cases=[
            "Binary classification with feature selection",
            "Regression with dimensionality reduction",
            "Custom feature engineering workflows"
        ],
        skill_level="intermediate"
    )
    
    # Create enhanced metadata using the new pattern
    enhanced_metadata = EnhancedDAGMetadata(
        dag_id="feature_selection_training_pipeline",
        description="Training pipeline with automated feature selection step",
        complexity="standard",
        features=["training", "feature_selection", "preprocessing"],
        framework="sklearn",
        node_count=4,
        edge_count=3,
        zettelkasten_metadata=zettelkasten_metadata
    )
    
    return enhanced_metadata


def create_dag() -> PipelineDAG:
    """
    Create a feature selection training pipeline DAG.
    
    This function uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_feature_selection_training_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
    """
    Create a SageMaker Pipeline from the DAG for feature selection training.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        validate: Whether to validate the DAG before compilation
        
    Returns:
        Tuple containing:
            - Pipeline: The created SageMaker pipeline
            - Dict: Conversion report with details about the compilation
            - PipelineDAGCompiler: The compiler instance for further operations
            - Any: The pipeline template instance for further operations
    """
    dag = create_dag()
    
    # Create compiler with the configuration
    dag_compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    # Set optional pipeline properties
    if pipeline_name:
        dag_compiler.pipeline_name = pipeline_name
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    
    # Validate the DAG if requested
    if validate:
        validation = dag_compiler.validate_dag_compatibility(dag)
        if not validation.is_valid:
            logger.warning(f"DAG validation failed: {validation.summary()}")
            if validation.missing_configs:
                logger.warning(f"Missing configs: {validation.missing_configs}")
            if validation.unresolvable_builders:
                logger.warning(f"Unresolvable builders: {validation.unresolvable_builders}")
            if validation.config_errors:
                logger.warning(f"Config errors: {validation.config_errors}")
            if validation.dependency_issues:
                logger.warning(f"Dependency issues: {validation.dependency_issues}")
    
    # Preview resolution for logging
    preview = dag_compiler.preview_resolution(dag)
    logger.info("DAG node resolution preview:")
    for node, config_type in preview.node_config_map.items():
        confidence = preview.resolution_confidence.get(node, 0.0)
        logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Get the pipeline template instance for further operations
    pipeline_template = dag_compiler.get_last_template()
    if pipeline_template is None:
        logger.warning("Pipeline template instance not found after compilation")
    else:
        logger.info("Pipeline template instance retrieved for further operations")
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    
    # Sync to registry after successful pipeline creation
    sync_to_registry()
    
    return pipeline, report, dag_compiler, pipeline_template


def fill_execution_document(
    pipeline: Pipeline,
    document: Dict[str, Any],
    dag_compiler: PipelineDAGCompiler
) -> Dict[str, Any]:
    """
    Fill an execution document for the pipeline with all necessary parameters.
    
    Args:
        pipeline: The compiled SageMaker pipeline
        document: Initial parameter document with user-provided values
        dag_compiler: The DAG compiler used to create the pipeline
    
    Returns:
        Dict: Complete execution document ready for pipeline execution
    """
    # Create execution document with all required parameters
    execution_doc = dag_compiler.create_execution_document(document)
    return execution_doc


def sync_to_registry() -> bool:
    """
    Synchronize this pipeline's metadata to the catalog registry.
    
    Returns:
        bool: True if synchronization was successful, False otherwise
    """
    try:
        registry = CatalogRegistry()
        enhanced_metadata = get_enhanced_dag_metadata()
        
        # Add or update the pipeline node using the enhanced metadata
        success = registry.add_or_update_enhanced_node(enhanced_metadata)
        
        if success:
            logger.info(f"Successfully synchronized {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry")
        else:
            logger.warning(f"Failed to synchronize {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry")
            
        return success
        
    except Exception as e:
        logger.error(f"Error synchronizing to registry: {e}")
        return False


if __name__ == "__main__":
    # Example usage following the established pattern
    import os
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a feature selection training pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    parser.add_argument('--sync-registry', action='store_true', help='Sync pipeline metadata to registry')
    parser.add_argument('--save-execution-doc', type=str, help='Save execution document to specified path')
    args = parser.parse_args()
    
    # Sync to registry if requested
    if args.sync_registry:
        success = sync_to_registry()
        if success:
            print("Successfully synchronized pipeline metadata to registry")
        else:
            print("Failed to synchronize pipeline metadata to registry")
        exit(0)
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Use provided config path or fallback to default
    config_path = args.config_path
    if not config_path:
        config_dir = Path.cwd().parent / "pipeline_config"
        config_path = os.path.join(config_dir, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Default config file not found: {config_path}")
    
    # Create the pipeline
    pipeline, report, dag_compiler, pipeline_template = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="Feature-Selection-Training",
        pipeline_description="Training pipeline with automated feature selection"
    )
    
    # Create execution document
    execution_doc = fill_execution_document(
        pipeline=pipeline, 
        document={
            "training_dataset": "my-dataset",
            "selection_method": "mutual_info",
            "n_features": "20",
            "target_column": "target"
        }, 
        dag_compiler=dag_compiler
    )
    
    # Save execution document if requested
    if args.save_execution_doc:
        save_execution_document(
            document=execution_doc,
            output_path=args.save_execution_doc
        )
    
    # Upsert if requested
    if args.upsert or args.execute:
        pipeline.upsert()
        logger.info(f"Pipeline '{pipeline.name}' upserted successfully")
```

## Summary

We have successfully implemented a complete Feature Selection step with:

✅ **Step 1: Workspace Context Setup** - Configured development environment  
✅ **Step 2: Configuration Class** - Three-tier field classification with validation  
✅ **Step 3: Script Contract** - Interface definition between script and container  
✅ **Step 4: Processing Script** - Business logic with unified main function interface  
✅ **Step 5: Step Specification** - Dependencies and outputs definition  
✅ **Step 6: Step Builder** - SageMaker step creation with specification-driven approach  
✅ **Step 7: Registry Integration** - Automatic discovery and registration  
✅ **Step 8: Validation Framework** - Three-level validation (alignment, builder, runtime)  
✅ **Step 9: Unit Tests** - Comprehensive test coverage  
✅ **Step 10: Pipeline Integration** - Template showing pipeline usage  

## Key Best Practices Demonstrated

1. **Specification-Driven Development**: Using specifications to drive input/output generation
2. **Contract Alignment**: Ensuring logical names match between contract and specification
3. **Three-Tier Configuration**: Clear separation of user inputs, system defaults, and derived fields
4. **Unified Main Function Interface**: Standardized script interface for testability
5. **Comprehensive Validation**: Three-level validation covering all aspects
6. **Error Handling**: Robust error handling with informative messages
7. **Environment Variable Management**: Proper handling of required and optional variables
8. **Path Handling**: Using contract paths and creating directories properly
9. **Registry Integration**: Modern hybrid registry system with automatic discovery
10. **Pipeline Catalog Integration**: Zettelkasten-based pipeline discovery and connection

## Usage Example

```python
from cursus.steps.configs.config_feature_selection import FeatureSelectionConfig
from cursus.steps.builders.builder_feature_selection import FeatureSelectionStepBuilder

# Create configuration
config = FeatureSelectionConfig(
    region="us-west-2",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    selection_method="mutual_info",
    n_features=20,
    target_column="target"
)

# Create step builder
builder = FeatureSelectionStepBuilder(config)

# Create step (with dependencies)
step = builder.create_step(
    step_name="FeatureSelection",
    dependencies=[preprocessing_step]
)
```

## CLI Testing

Test the step using Cursus CLI commands:

```bash
# Test script functionality
cursus runtime test-script feature_selection --workspace-dir ./test_workspace

# Test data compatibility
cursus runtime test-compatibility tabular_preprocessing feature_selection

# Test pipeline flow
cursus runtime test-pipeline pipeline_with_feature_selection.json
```

This example demonstrates the complete workflow for adding a new pipeline step using the Cursus framework's modern development approach, ensuring robust, reliable, and well-tested pipeline components that integrate seamlessly with the existing system.
