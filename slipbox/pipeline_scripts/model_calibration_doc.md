# Model Calibration Pipeline Script

## Overview

The `model_calibration.py` script is designed to calibrate model prediction scores into accurate probabilities, which is essential for risk-based decision-making and threshold setting. Model calibration addresses the problem of overconfidence or underconfidence in probabilistic predictions, ensuring that a predicted probability of 0.8 truly corresponds to an 80% likelihood of the predicted event occurring.

The script supports multiple calibration methods including GAM (Generalized Additive Models), Isotonic Regression, and Platt Scaling, with options for monotonicity constraints. It handles both binary and multi-class classification scenarios from XGBoost models.

## Why Calibration Matters

Uncalibrated models may provide scores that don't reflect true probabilities. For example:
- A model might predict a 90% probability for samples that actually belong to the positive class only 70% of the time (overconfidence)
- Or it might predict a 60% probability for samples that actually belong to the positive class 80% of the time (underconfidence)

Proper calibration ensures that:
- Risk assessments are accurate
- Decision thresholds can be meaningfully set
- Probabilistic predictions can be trusted for downstream tasks

## Implementation Details

### Data Processing

1. **Input Data Format**: The script expects evaluation data with:
   - A ground truth label column
   - For binary classification: a single probability score column
   - For multi-class: probability columns for each class (e.g., prob_class_0, prob_class_1, prob_class_2)

2. **Data Loading Process**:
   - Locates the first supported file (.csv, .parquet, .json) in the input directory
   - Validates required columns based on binary/multi-class mode
   - Extracts target labels and uncalibrated probability scores

### Calibration Methods

#### 1. GAM (Generalized Additive Model)

- **Implementation**: Uses LogisticGAM from the pygam package
- **Configuration**:
  - Number of splines: Controls flexibility of the calibration curve
  - Monotonic constraint: Ensures calibrated probabilities increase monotonically with uncalibrated scores
- **Advantages**: Flexible, can capture complex non-linear relationships
- **Fallback**: If pygam is not installed, falls back to Platt scaling

#### 2. Isotonic Regression

- **Implementation**: Uses IsotonicRegression from scikit-learn
- **Configuration**: 
  - out_of_bounds='clip': Handles predictions outside the training range
- **Advantages**: Non-parametric, preserves rank ordering of predictions

#### 3. Platt Scaling

- **Implementation**: Uses LogisticRegression from scikit-learn
- **Configuration**:
  - C=1e5: Minimal regularization to focus on calibration
- **Advantages**: Simple, robust, works well when the uncalibrated scores are already somewhat calibrated

### Multi-Class Approach

For multi-class calibration, the script implements a one-vs-rest approach:

1. **One-Hot Encoding**: Converts multi-class labels to one-hot encoded format
2. **Per-Class Calibration**: Trains a separate calibration model for each class
3. **Probability Normalization**: Ensures the sum of calibrated probabilities equals 1.0

### Metrics Calculation

#### Binary Classification Metrics

1. **Expected Calibration Error (ECE)**:
   - Divides predictions into bins
   - Computes the weighted average of absolute differences between:
     - Mean predicted probability in each bin
     - Fraction of positive samples in each bin

2. **Maximum Calibration Error (MCE)**:
   - Maximum absolute calibration error across all bins

3. **Brier Score**:
   - Mean squared difference between:
     - Predicted probabilities
     - Actual outcomes (0 or 1)

4. **Reliability Diagram Data**:
   - Points for plotting calibration curve
   - Bin statistics with detailed information

#### Multi-Class Metrics

1. **Per-Class Metrics**:
   - Computes binary metrics for each class independently

2. **Aggregate Metrics**:
   - **Macro-averaged ECE**: Average of per-class ECEs
   - **Macro-averaged MCE**: Average of per-class MCEs
   - **Multi-class Brier Score**: Specialized version for multi-class

### Visualization

#### Binary Classification

- **Reliability Diagram**: Plots calibration curves showing:
  - Perfect calibration (diagonal line)
  - Uncalibrated predictions
  - Calibrated predictions
- **Prediction Distribution**: Histogram showing distribution of probabilities before and after calibration

#### Multi-Class Classification

- **Per-Class Reliability Diagrams**: Creates a grid of reliability diagrams, one for each class
- **Dynamic Grid Sizing**: Adjusts the layout based on the number of classes

## Step-by-Step Workflow

### Binary Classification

1. **Environment Setup**:
   - Load configuration from environment variables
   - Create output directories

2. **Data Preparation**:
   - Load evaluation data with predictions
   - Extract ground truth labels and uncalibrated probabilities

3. **Model Training**:
   - Select calibration method (GAM, Isotonic, or Platt)
   - Train calibration model on uncalibrated scores

4. **Score Calibration**:
   - Apply calibration model to uncalibrated scores
   - Generate calibrated probabilities

5. **Metrics Computation**:
   - Calculate calibration metrics before calibration
   - Calculate calibration metrics after calibration
   - Compute improvement metrics

6. **Visualization**:
   - Create reliability diagram comparing uncalibrated vs. calibrated probabilities
   - Generate histograms of probability distributions

7. **Output Generation**:
   - Save calibration model
   - Save metrics report with detailed statistics
   - Save calibrated data with both original and calibrated probabilities
   - Generate summary report with key findings

### Multi-Class Classification

1. **Environment Setup**:
   - Same as binary, plus multi-class specific configuration

2. **Data Preparation**:
   - Load evaluation data
   - Extract ground truth labels
   - Locate and extract probability columns for each class

3. **Model Training**:
   - For each class:
     - Convert ground truth to one-hot encoding
     - Train a separate calibration model
   - Store all calibration models

4. **Score Calibration**:
   - Apply each calibration model to corresponding class probabilities
   - Normalize resulting probabilities to sum to 1.0

5. **Metrics Computation**:
   - Compute per-class calibration metrics
   - Calculate aggregate metrics across classes

6. **Visualization**:
   - Create reliability diagram for each class
   - Arrange in a grid layout

7. **Output Generation**:
   - Save all calibration models (one per class)
   - Save metrics report with per-class and aggregate metrics
   - Save calibrated data with both original and calibrated probabilities
   - Generate summary report

## Input and Output Format

### Input Files

- **Evaluation Data**: CSV, Parquet, or JSON file containing:
  - Label column (specified by LABEL_FIELD)
  - For binary: score column (specified by SCORE_FIELD)
  - For multi-class: multiple score columns (following SCORE_FIELD_PREFIX pattern)

### Output Files

1. **Calibration Model(s)**:
   - Binary: Single joblib file containing the calibration model
   - Multi-class: Multiple joblib files, one for each class

2. **Metrics Report**: JSON file containing:
   - Calibration method used
   - Detailed metrics before and after calibration
   - Improvement statistics
   - Configuration parameters

3. **Calibrated Data**: Parquet file containing:
   - Original input data
   - Added columns with calibrated probabilities

4. **Visualizations**: PNG images of reliability diagrams

5. **Summary Report**: JSON file with high-level results and file paths

## Common Use Cases

1. **Risk Model Calibration**: Ensuring risk scores reflect true probabilities
2. **Multi-class Classification**: Calibrating probabilities across multiple classes
3. **Model Post-Processing**: Improving model outputs without retraining
4. **A/B Testing**: Comparing different calibration methods
5. **Threshold Setting**: Finding optimal decision thresholds based on calibrated probabilities

## Environment Variables

The script uses the following environment variables for configuration:

- **CALIBRATION_METHOD**: Calibration method to use ("gam", "isotonic", "platt")
- **LABEL_FIELD**: Column name for ground truth labels
- **SCORE_FIELD**: Column name for prediction scores (binary classification)
- **MONOTONIC_CONSTRAINT**: Whether to enforce monotonicity in calibration
- **GAM_SPLINES**: Number of splines for GAM calibration
- **ERROR_THRESHOLD**: Threshold for improvement warning
- **IS_BINARY**: Whether the task is binary classification
- **NUM_CLASSES**: Number of classes for multi-class classification
- **SCORE_FIELD_PREFIX**: Prefix for probability columns in multi-class scenario
- **MULTICLASS_CATEGORIES**: Optional class names for better reporting

## Testability Considerations

While the current implementation works well for its intended purpose, there are several aspects that could be improved to enhance testability:

### Current Testability Challenges

1. **Global Environment Variables**: Heavy reliance on environment variables for configuration
2. **Hard-coded I/O Paths**: File paths defined at the module level
3. **Function Dependencies**: Functions depending on global state
4. **Limited Parameterization**: Some functions use globals instead of parameters

### Recommendations for Better Testability

#### 1. Create a Configuration Class

```python
class CalibrationConfig:
    """Configuration class for model calibration."""
    
    def __init__(
        self,
        calibration_method="gam",
        label_field="label",
        score_field="prob_class_1",
        monotonic_constraint=True,
        gam_splines=10,
        error_threshold=0.05,
        is_binary=True,
        num_classes=2,
        score_field_prefix="prob_class_",
        multiclass_categories=None,
        input_path="/opt/ml/processing/input/eval_data",
        output_calibration_path="/opt/ml/processing/output/calibration",
        output_metrics_path="/opt/ml/processing/output/metrics",
        output_calibrated_data_path="/opt/ml/processing/output/calibrated_data"
    ):
        # Initialize configuration properties
        ...
        
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables."""
        # Load configuration from environment
        ...
```

#### 2. Refactor Core Functions to Accept Configuration

```python
def load_data(config, data_file=None):
    """Load evaluation data with predictions."""
    # Implementation using explicit config parameter
    ...

def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """Pure function for computing metrics."""
    # Implementation unchanged but without global dependencies
    ...

def train_calibration_model(scores, labels, method, monotonic_constraint=True, gam_splines=10):
    """Train calibration model based on method."""
    # Unified interface for all calibration methods
    ...
```

#### 3. Create a CalibrationProcessor Class

```python
class CalibrationProcessor:
    """Class for handling model calibration."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        ...
        
    def process(self):
        """Main processing function."""
        ...
            
    def _process_binary(self):
        """Process binary classification data."""
        ...
        
    def _process_multiclass(self):
        """Process multi-class classification data."""
        ...
```

#### 4. Create Separate IO and Processing Functions

```python
def save_metrics(metrics, path):
    """Save metrics to a file."""
    ...

def save_model(model, path):
    """Save calibration model."""
    ...
```

### Unit Testing Approach

With these changes, unit testing would be significantly easier:

1. **Test Configuration**: Create test configs with specific parameters
2. **Mock Data**: Use pre-defined DataFrames instead of loading from files
3. **Test Processing Logic**: Verify calibration logic in isolation
4. **Test Metrics Computation**: Ensure metrics are calculated correctly

Example test:

```python
def test_binary_calibration():
    """Test binary classification calibration."""
    # Create test configuration
    config = CalibrationConfig(
        is_binary=True,
        calibration_method="platt",
        input_path="test_data",
        output_calibration_path="test_output/calibration",
        output_metrics_path="test_output/metrics",
        output_calibrated_data_path="test_output/data"
    )
    
    # Create processor with mocked data
    processor = CalibrationProcessor(config)
    
    # Mock data
    df = pd.DataFrame({
        "label": [0, 1, 0, 1, 0],
        "prob_class_1": [0.3, 0.8, 0.4, 0.7, 0.2]
    })
    
    # Test process function with mocked data
    result = processor.process_with_data(df)
    
    # Assertions
    assert "calibrated" in result
    assert result["improvement_percentage"] > 0
    # etc.
```

## Conclusion

The `model_calibration.py` script provides robust functionality for calibrating both binary and multi-class classification models. With the suggested refactoring for testability, it would be easier to maintain, extend, and verify through automated testing.
