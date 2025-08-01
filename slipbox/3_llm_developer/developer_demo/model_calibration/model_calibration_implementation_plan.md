# Implementation Plan for ModelCalibration Step

## 1. Step Overview
- **Purpose**: Take a trained model's raw prediction scores and calibrate them to better reflect true probabilities, which is essential for risk-based decision-making, threshold setting, and confidence in model outputs.
- **Inputs**: Dataset with features, ground truth labels, and precomputed model prediction scores (from XGBoost model training step's evaluation_output)
- **Outputs**: 
  - Calibration mapping between original scores and calibrated probabilities
  - Calibration metrics (reliability diagrams, calibration error)
  - Calibrated dataset with new probabilities
- **Position in pipeline**: After XGBoost model training and before MIMS packaging
- **Architectural considerations**: 
  - Must support multiple calibration methods (GAM, Isotonic Regression, Platt Scaling)
  - Must maintain the pipeline's specification-driven design
  - Will use ProcessingStep since it's not training a new model but transforming outputs
- **Alignment with design principles**: Maintains strict alignment between script contract, step specification, and builder implementation following the architectural patterns

## 2. Components to Create
### Script Contract: src/pipeline_script_contracts/model_calibration_contract.py
- **Input paths**:
  - `evaluation_data`: `/opt/ml/processing/input/eval_data` (from XGBoost training evaluation output)
  - `model_artifacts`: `/opt/ml/processing/input/model` (trained model for reference)
  - `code`: `/opt/ml/processing/input/code` (custom code dependencies)
- **Output paths**:
  - `calibration_output`: `/opt/ml/processing/output/calibration` (calibration mapping and artifacts)
  - `metrics_output`: `/opt/ml/processing/output/metrics` (calibration quality metrics)
  - `calibrated_data`: `/opt/ml/processing/output/calibrated_data` (dataset with calibrated probabilities)
- **Environment variables**:
  - `CALIBRATION_METHOD`: Required - calibration method to use (GAM, isotonic, platt)
  - `LABEL_FIELD`: Required - name of the label column 
  - `SCORE_FIELD`: Required - name of the prediction score column
  - `MONOTONIC_CONSTRAINT`: Optional - whether to enforce monotonicity in GAM (default: "True")
  - `GAM_SPLINES`: Optional - number of splines for GAM (default: "10")
  - `ERROR_THRESHOLD`: Optional - acceptable calibration error threshold (default: "0.05")

### Step Specification: src/pipeline_step_specs/model_calibration_spec.py
- **Dependencies**:
  - `evaluation_data`:
    - logical_name: "evaluation_data"
    - dependency_type: DependencyType.PROCESSING_OUTPUT
    - required: True
    - compatible_sources: ["XGBoostTraining", "XGBoostModelEval"]
    - semantic_keywords: ["evaluation", "predictions", "scores"]
  - `model_artifacts`:
    - logical_name: "model_artifacts"
    - dependency_type: DependencyType.MODEL_ARTIFACTS
    - required: True
    - compatible_sources: ["XGBoostTraining", "TrainingStep"]
- **Outputs**:
  - `calibration_output`:
    - logical_name: "calibration_output"
    - output_type: DependencyType.PROCESSING_OUTPUT
    - property_path: "properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri"
  - `metrics_output`:
    - logical_name: "metrics_output" 
    - output_type: DependencyType.PROCESSING_OUTPUT
    - property_path: "properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri"
  - `calibrated_data`:
    - logical_name: "calibrated_data"
    - output_type: DependencyType.PROCESSING_OUTPUT
    - property_path: "properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri"
- **Job type variants**: No variants needed as calibration methods are controlled via configuration parameter

### Configuration: src/pipeline_steps/config_model_calibration_step.py
- **Step-specific parameters**:
  - `calibration_method`: str - Method to use for calibration (GAM, isotonic, platt)
  - `monotonic_constraint`: bool - Whether to enforce monotonicity in GAM
  - `gam_splines`: int - Number of splines for GAM
  - `error_threshold`: float - Acceptable calibration error threshold
- **SageMaker parameters**:
  - `processing_instance_type`: str (default: "ml.m5.xlarge")
  - `processing_instance_count`: int (default: 1)
  - `processing_volume_size`: int (default: 30)
  - `max_runtime_seconds`: int (default: 3600)
- **Required validation checks**:
  - `calibration_method` must be one of the supported methods
  - `gam_splines` must be > 0
  - `error_threshold` must be between 0 and 1

### Step Builder: src/pipeline_steps/builder_model_calibration_step.py
- **Special handling**:
  - Set environment variables from config for calibration parameters
  - Manage file paths for calibration artifacts
  - Handle model artifact loading
- **Required helper methods**:
  - `_get_processor_env_vars`: Set environment variables for calibration
  - `_get_inputs`: Map dependency inputs to processor inputs
  - `_get_outputs`: Define processor outputs
  - `validate_configuration`: Validate calibration-specific parameters
- **Input/output handling**: Standard specification-driven processor input/output mapping

### Processing Script: src/pipeline_scripts/model_calibration.py
- **Algorithm**: Implementation of multiple calibration methods (GAM with monotonic constraints, Isotonic Regression, Platt Scaling)
- **Main functions**:
  - `load_data_and_model`: Load evaluation data and model artifacts
  - `fit_calibration_model`: Train calibration model on prediction scores
  - `calibrate_probabilities`: Apply calibration to raw scores
  - `evaluate_calibration`: Calculate calibration metrics (reliability, ECE)
  - `plot_reliability_diagram`: Create reliability diagram visualizations
  - `save_calibration_artifacts`: Save calibration model and artifacts
- **Error handling strategy**: Detailed exception handling with informative messages for calibration-specific failures

## 3. Files to Update

### src/pipeline_registry/step_names.py
```python
STEP_NAMES = {
    # ... existing steps ...
    
    "ModelCalibration": {
        "config_class": "ModelCalibrationConfig",
        "builder_step_name": "ModelCalibrationStepBuilder",
        "spec_type": "ModelCalibration",
        "description": "Calibrates model prediction scores to accurate probabilities"
    },
}
```

### src/pipeline_steps/__init__.py
```python
# Add to existing imports
from .builder_model_calibration_step import ModelCalibrationStepBuilder
from .config_model_calibration_step import ModelCalibrationConfig
```

### src/pipeline_step_specs/__init__.py
```python
# Add to existing imports
from .model_calibration_spec import MODEL_CALIBRATION_SPEC
```

### src/pipeline_script_contracts/__init__.py
```python
# Add to existing imports
from .model_calibration_contract import MODEL_CALIBRATION_CONTRACT
```

## 4. Integration Strategy

### Upstream Steps
- XGBoostTraining: Provides model artifacts and evaluation data with prediction scores

### Downstream Steps
- MIMSPackaging: Will consume the calibrated model and calibration artifacts

### DAG Updates
- Add ModelCalibration node after XGBoostTraining
- Connect XGBoostTraining.evaluation_output → ModelCalibration.evaluation_data
- Connect XGBoostTraining.model_artifacts → ModelCalibration.model_artifacts
- Connect ModelCalibration.calibration_output → MIMSPackaging.calibration_artifacts

## 5. Contract-Specification Alignment

### Input Alignment
- Script contract input path for `evaluation_data` maps to specification dependency with logical name "evaluation_data"
- Script contract input path for `model_artifacts` maps to specification dependency with logical name "model_artifacts"

### Output Alignment
- Script contract output path for `calibration_output` maps to specification output with logical name "calibration_output"
- Script contract output path for `metrics_output` maps to specification output with logical name "metrics_output"
- Script contract output path for `calibrated_data` maps to specification output with logical name "calibrated_data"

### Validation Strategy
- Create unit tests that verify alignment between contract and specification
- Add validation to script that ensures expected paths exist and have correct content
- Include property path validation in step specification tests

## 6. Error Handling Strategy

### Input Validation
- Verify evaluation data contains required columns (predictions, labels)
- Check model artifacts exist and are valid
- Validate calibration parameters are within allowed ranges

### Script Robustness
- Handle sparse or imbalanced data during calibration
- Implement fallback approaches if primary calibration method fails
- Create contingency for low data scenarios

### Logging Strategy
- Log detailed information about calibration process and parameters
- Include warnings for potential issues (e.g., insufficient data for reliable calibration)
- Log before/after calibration metrics for performance comparison

### Error Reporting
- Generate comprehensive error messages for common failure modes
- Include suggestions for fixing calibration issues
- Create visualization of problematic score distributions when calibration fails

## 7. Testing and Validation Plan

### Unit Tests
- Test configuration validation logic
- Test builder input/output mapping
- Test script contract alignment with specification
- Test calibration methods with synthetic data

### Integration Tests
- Test step in pipeline context with real model outputs
- Verify correct artifacts are produced and stored

### Validation Criteria
- Calibrated probabilities should have lower calibration error than raw scores
- Reliability diagram should be closer to the ideal diagonal line
- Original model discrimination performance (AUC) should be preserved

### S3 Path Handling Tests
- Test handling of S3 paths for inputs and outputs
- Verify proper resolution of PipelineVariable references

### PipelineVariable Handling Tests
- Test handling of SageMaker PipelineVariable objects in inputs
- Verify proper string conversion and normalization

## Implementation Details

### 1. Step Registry Addition

```python
# In src/pipeline_registry/step_names.py
STEP_NAMES = {
    # ... existing steps ...
    
    "ModelCalibration": {
        "config_class": "ModelCalibrationConfig",
        "builder_step_name": "ModelCalibrationStepBuilder",
        "spec_type": "ModelCalibration",
        "description": "Calibrates model prediction scores to accurate probabilities"
    },
}
```

### 2. Script Contract Implementation

```python
# src/pipeline_script_contracts/model_calibration_contract.py
from .base_script_contract import ScriptContract

MODEL_CALIBRATION_CONTRACT = ScriptContract(
    entry_point="model_calibration.py",
    expected_input_paths={
        "evaluation_data": "/opt/ml/processing/input/eval_data",
        "model_artifacts": "/opt/ml/processing/input/model",
        "code": "/opt/ml/processing/input/code"
    },
    expected_output_paths={
        "calibration_output": "/opt/ml/processing/output/calibration",
        "metrics_output": "/opt/ml/processing/output/metrics",
        "calibrated_data": "/opt/ml/processing/output/calibrated_data"
    },
    required_env_vars=[
        "CALIBRATION_METHOD",
        "LABEL_FIELD", 
        "SCORE_FIELD"
    ],
    optional_env_vars={
        "MONOTONIC_CONSTRAINT": "True",
        "GAM_SPLINES": "10",
        "ERROR_THRESHOLD": "0.05"
    },
    framework_requirements={
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.20.0",
        "pygam": ">=0.8.0",
        "matplotlib": ">=3.3.0"
    },
    description="Contract for model calibration processing step"
)
```

### 3. Step Specification Implementation

```python
# src/pipeline_step_specs/model_calibration_spec.py
from ..pipeline_deps.base_specifications import StepSpecification, NodeType, DependencySpec, OutputSpec, DependencyType
from ..pipeline_registry.step_names import get_spec_step_type

def _get_model_calibration_contract():
    from ..pipeline_script_contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT
    return MODEL_CALIBRATION_CONTRACT

MODEL_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("ModelCalibration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_calibration_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["XGBoostTraining", "XGBoostModelEval"],
            semantic_keywords=["evaluation", "predictions", "scores"],
            data_type="S3Uri",
            description="Evaluation dataset with ground truth labels and model predictions"
        ),
        "model_artifacts": DependencySpec(
            logical_name="model_artifacts",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=["model", "artifacts", "trained"],
            data_type="S3Uri",
            description="Trained model artifacts for reference"
        )
    },
    outputs={
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration mapping and artifacts"
        ),
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration quality metrics and visualizations"
        ),
        "calibrated_data": OutputSpec(
            logical_name="calibrated_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Dataset with calibrated probabilities"
        )
    }
)
```

### 4. Configuration Class Implementation

```python
# src/pipeline_steps/config_model_calibration_step.py
from .config_processing_step_base import ProcessingStepConfigBase

class ModelCalibrationConfig(ProcessingStepConfigBase):
    """Configuration for ModelCalibration step."""
    
    def __init__(
        self,
        region: str,
        pipeline_s3_loc: str,
        processing_instance_type: str = "ml.m5.xlarge",
        processing_instance_count: int = 1,
        processing_volume_size: int = 30,
        max_runtime_seconds: int = 3600,
        pipeline_name: str = None,
        calibration_method: str = "gam",
        monotonic_constraint: bool = True,
        gam_splines: int = 10,
        error_threshold: float = 0.05,
        label_field: str = "label",
        score_field: str = "prob_class_1"
    ):
        """Initialize ModelCalibration configuration.
        
        Args:
            region: AWS region
            pipeline_s3_loc: S3 location for pipeline artifacts
            processing_instance_type: SageMaker instance type
            processing_instance_count: Number of processing instances
            processing_volume_size: EBS volume size in GB
            max_runtime_seconds: Maximum runtime in seconds
            pipeline_name: Name of the pipeline
            calibration_method: Method to use for calibration (gam, isotonic, platt)
            monotonic_constraint: Whether to enforce monotonicity in GAM
            gam_splines: Number of splines for GAM
            error_threshold: Acceptable calibration error threshold
            label_field: Name of the label column
            score_field: Name of the score column to calibrate
        """
        super().__init__(
            region=region,
            pipeline_s3_loc=pipeline_s3_loc,
            processing_instance_type=processing_instance_type,
            processing_instance_count=processing_instance_count,
            processing_volume_size=processing_volume_size,
            max_runtime_seconds=max_runtime_seconds,
            pipeline_name=pipeline_name,
            processing_entry_point="model_calibration.py",
            processing_source_dir="src/pipeline_scripts"
        )
        
        self.calibration_method = calibration_method
        self.monotonic_constraint = monotonic_constraint
        self.gam_splines = gam_splines
        self.error_threshold = error_threshold
        self.label_field = label_field
        self.score_field = score_field
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..pipeline_script_contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT
        return MODEL_CALIBRATION_CONTRACT
        
    def get_script_path(self):
        """Return the script path relative to the source directory."""
        return self.processing_entry_point
```

### 5. Step Builder Implementation

```python
# src/pipeline_steps/builder_model_calibration_step.py
from typing import Dict, List, Any, Optional
import logging

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ..pipeline_deps.base_specifications import StepSpecification
from ..pipeline_script_contracts.base_script_contract import ScriptContract
from .builder_step_base import StepBuilderBase
from .config_model_calibration_step import ModelCalibrationConfig
from ..pipeline_step_specs.model_calibration_spec import MODEL_CALIBRATION_SPEC

logger = logging.getLogger(__name__)

class ModelCalibrationStepBuilder(StepBuilderBase):
    """Builder for ModelCalibration processing step."""
    
    def __init__(
        self, 
        config, 
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        super().__init__(
            config=config,
            spec=MODEL_CALIBRATION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: ModelCalibrationConfig = config
        
    def validate_configuration(self) -> None:
        """Validates the provided configuration."""
        self.log_info("Validating ModelCalibrationConfig...")
        
        # Validate required attributes
        required_attrs = [
            'processing_entry_point',
            'processing_source_dir',
            'processing_instance_count',
            'processing_volume_size',
            'calibration_method',
            'label_field',
            'score_field'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelCalibrationConfig missing required attribute: {attr}")
        
        # Validate calibration method
        valid_methods = ['gam', 'isotonic', 'platt']
        if self.config.calibration_method.lower() not in valid_methods:
            raise ValueError(f"Invalid calibration method: {self.config.calibration_method}. "
                            f"Must be one of: {valid_methods}")
        
        # Validate numeric parameters
        if self.config.gam_splines <= 0:
            raise ValueError(f"gam_splines must be > 0, got {self.config.gam_splines}")
            
        if not 0 <= self.config.error_threshold <= 1:
            raise ValueError(f"error_threshold must be between 0 and 1, got {self.config.error_threshold}")
            
        self.log_info("ModelCalibrationConfig validation succeeded.")
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor."""
        env_vars = super()._get_environment_variables()
        
        # Add calibration-specific environment variables
        env_vars.update({
            "CALIBRATION_METHOD": self.config.calibration_method.lower(),
            "LABEL_FIELD": self.config.label_field,
            "SCORE_FIELD": self.config.score_field,
            "MONOTONIC_CONSTRAINT": str(self.config.monotonic_constraint).lower(),
            "GAM_SPLINES": str(self.config.gam_splines),
            "ERROR_THRESHOLD": str(self.config.error_threshold)
        })
        
        return env_vars
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract."""
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
            
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract."""
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
            
        return self._get_spec_driven_processor_outputs(outputs)
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the model calibration processing step.
        
        Args:
            **kwargs: Additional keyword arguments for step creation.
                     Should include 'dependencies' list if step has dependencies.
                     
        Returns:
            ProcessingStep: The configured model calibration processing step.
        """
        # Extract inputs from dependencies using the resolver
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Get processor inputs and outputs
        inputs = self._get_inputs(extracted_inputs)
        outputs = self._get_outputs({})
        
        # Create processor
        processor = self._get_processor()
        
        # Set environment variables
        env_vars = self._get_environment_variables()
        
        # Create and return the step
        step_name = kwargs.get('step_name', 'ModelCalibration')
        step = processor.run(
            inputs=inputs,
            outputs=outputs,
            container_arguments=[],
            container_entrypoint=["python", self.config.get_script_path()],
            job_name=self._generate_job_name(step_name),
            wait=False,
            environment=env_vars
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

### 6. Processing Script Implementation (Core Functions)

```python
# src/pipeline_scripts/model_calibration.py
#!/usr/bin/env python

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Import pygam for GAM implementation if available
try:
    from pygam import LogisticGAM, s
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
INPUT_DATA_PATH = "/opt/ml/processing/input/eval_data"
INPUT_MODEL_PATH = "/opt/ml/processing/input/model"
INPUT_CODE_PATH = "/opt/ml/processing/input/code"
OUTPUT_CALIBRATION_PATH = "/opt/ml/processing/output/calibration"
OUTPUT_METRICS_PATH = "/opt/ml/processing/output/metrics"
OUTPUT_CALIBRATED_DATA_PATH = "/opt/ml/processing/output/calibrated_data"

# Get environment variables
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "gam").lower()
LABEL_FIELD = os.environ.get("LABEL_FIELD", "label")
SCORE_FIELD = os.environ.get("SCORE_FIELD", "prob_class_1")
MONOTONIC_CONSTRAINT = os.environ.get("MONOTONIC_CONSTRAINT", "True").lower() == "true"
GAM_SPLINES = int(os.environ.get("GAM_SPLINES", "10"))
ERROR_THRESHOLD = float(os.environ.get("ERROR_THRESHOLD", "0.05"))

def create_directories():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUT_CALIBRATION_PATH, exist_ok=True)
    os.makedirs(OUTPUT_METRICS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_CALIBRATED_DATA_PATH, exist_ok=True)

def find_first_data_file(data_dir: str) -> str:
    """Find the first supported data file in directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return os.path.join(data_dir, fname)
    return None

def load_data():
    """Load evaluation data with predictions."""
    data_file = find_first_data_file(INPUT_DATA_PATH)
    if not data_file:
        raise FileNotFoundError(f"No data file found in {INPUT_DATA_PATH}")
    
    logger.info(f"Loading data from {data_file}")
    if data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    # Validate required columns
    if LABEL_FIELD not in df.columns:
        raise ValueError(f"Label field '{LABEL_FIELD}' not found in data")
    if SCORE_FIELD not in df.columns:
        raise ValueError(f"Score field '{SCORE_FIELD}' not found in data")
    
    logger.info(f"Loaded data with shape {df.shape}")
    return df

def load_model_info():
    """Load model information if available."""
    hyperparams_path = os.path.join(INPUT_MODEL_PATH, "hyperparameters.json")
    model_info = {}
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, "r") as f:
            model_info = json.load(f)
            logger.info(f"Loaded model hyperparameters from {hyperparams_path}")
    return model_info

def train_gam_calibration(scores, labels):
    """Train a GAM calibration model with optional monotonicity constraints."""
    if not HAS_PYGAM:
        raise ImportError("pygam package is required for GAM calibration but not installed")
    
    scores = scores.reshape(-1, 1)  # Reshape for GAM
    
    # Configure GAM with monotonic constraint if specified
    if MONOTONIC_CONSTRAINT:
        gam = LogisticGAM(s(0, n_splines=GAM_SPLINES, constraints='monotonic_inc'))
        logger.info(f"Training GAM with monotonic constraint, {GAM_SPLINES} splines")
    else:
        gam = LogisticGAM(s(0, n_splines=GAM_SPLINES))
        logger.info(f"Training GAM without monotonic constraint, {GAM_SPLINES} splines")
    
    gam.fit(scores, labels)
    logger.info(f"GAM training complete, deviance: {gam.statistics_['deviance']}")
    return gam

def train_isotonic_calibration(scores, labels):
    """Train an isotonic regression calibration model."""
    logger.info("Training isotonic regression calibration model")
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, labels)
    logger.info("Isotonic regression training complete")
    return ir

def train_platt_scaling(scores, labels):
    """Train a Platt scaling (logistic regression) calibration model."""
    logger.info("Training Platt scaling (logistic regression) calibration model")
    scores = scores.reshape(-1, 1)  # Reshape for LogisticRegression
    lr = LogisticRegression(C=1e5)  # High C for minimal regularization
    lr.fit(scores, labels)
    logger.info("Platt scaling training complete")
    return lr

def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Compute calibration metrics including:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Calibration curve data points
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Compute ECE and MCE
    # Get bin assignments and counts
    bin_indices = np.minimum(n_bins - 1, (y_prob * n_bins).astype(int))
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = bin_counts.astype(np.float64)
    
    # Compute mean predicted probability in each bin
    bin_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins) / np.maximum(bin_counts, 1)
    
    # Compute mean true label in each bin
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins) / np.maximum(bin_counts, 1)
    
    # Compute calibration errors per bin
    abs_errors = np.abs(bin_probs - bin_true)
    
    # Expected Calibration Error (weighted average of absolute errors)
    ece = np.sum(bin_counts / len(y_true) * abs_errors)
    
    # Maximum Calibration Error
    mce = np.max(abs_errors)
    
    metrics = {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "reliability_diagram": {
            "true_probs": prob_true.tolist(),
            "pred_probs": prob_pred.tolist()
        },
        "bin_statistics": {
            "bin_counts": bin_counts.
