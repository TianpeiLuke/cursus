# TSAModelCalibration

**TSA (Temporal Self-Attention) model calibration step using monotone B-spline calibration for converting raw prediction scores to well-calibrated probabilities for fraud detection**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `tsa_model_calibration.py` |
| **Interface file** | `steps/interfaces/tsa_model_calibration.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Contract for TSA model calibration processing step.

The TSA model calibration step implements monotone B-spline calibration specifically
for Temporal Self-Attention fraud detection models. It converts raw model prediction scores
into well-calibrated probabilities using constrained optimization, which is essential for
risk-based decision-making and reliable fraud detection thresholds.

This calibration method is based on the generic_rfuge.r approach and uses:
- B-spline basis functions with adaptive knot placement
- Monotonicity constraints to ensure score ordering preservation
- Iterative reweighted least squares (IRLS) with quadratic programming
- Emphasis on high-score regions (90th-100th percentile) for fraud detection

Input Structure:
- /opt/ml/processing/input/eval_data: Evaluation dataset with ground truth labels and model predictions
  * Supports multiple formats: CSV, TSV, Parquet
  * Can handle nested tarballs from SageMaker training job outputs
  * Expected columns: label field (ground truth) and score field (raw predictions)

Output Structure:
- /opt/ml/processing/output/calibration: Calibration model artifacts
  * calibration_model.pkl: Pickled B-spline calibrator (for backward compatibility)
  * tsa_bspline_calibrator.json: JSON format calibrator (for inspection)
  * tsa_calibration_summary.json: Summary of calibration results
- /opt/ml/processing/output/metrics: Calibration quality metrics and visualizations
  * tsa_calibration_metrics.json: Comprehensive metrics (ECE, MCE, Brier score, AUC)
  * tsa_reliability_diagram.png: Visual comparison of uncalibrated vs calibrated
- /opt/ml/processing/output/calibrated_data: Dataset with calibrated probabilities
  * Original format preserved (CSV/TSV/Parquet)
  * New column: calibrated_{SCORE_FIELD}
  * All original columns retained

Command-Line Arguments:
- job-type: Determines data loading strategy
  * "training": Uses nested tarball extraction for training job outputs
  * "calibration"/"validation"/"testing": Uses standard data loading

Environment Variables (Required):
- CALIBRATION_METHOD: Calibration method to use (currently only "bspline" supported)
- LABEL_FIELD: Name of the ground truth label column (e.g., "is_abusive_mdr")
- SCORE_FIELD: Name of the raw prediction score column (e.g., "prob_class_1")

Environment Variables (Optional - B-spline Configuration):
- BSPLINE_DEGREE: Degree of B-spline basis functions (default: 3 for cubic splines)
- ADAPTIVE_KNOTS: Whether to use adaptive knot placement based on data size (default: True)
- BASE_KNOTS: Fixed number of knots to use (overrides adaptive if set)

Environment Variables (Optional - Quality Thresholds):
- MIN_RECORDS: Minimum number of records required for calibration (default: 1000)
- MIN_FRAUD: Minimum number of fraud cases required (default: 10)
- MAX_COEF_THRESHOLD: Maximum acceptable coefficient magnitude (default: 1e12)
- MIN_UNIQUE_VALUES: Minimum unique calibrated predictions required (default: 10)

Environment Variables (Optional - Optimization Parameters):
- LAMBDA_SMOOTH: Smoothness penalty for P-spline regularization (default: 1e-10)
- MAX_ITER: Maximum iterations for IRLS optimization (default: 1000)
- TOLERANCE: Convergence tolerance for coefficient updates (default: 1e-6)

Infrastructure:
- USE_SECURE_PYPI: Whether to use secure CodeArtifact PyPI for package installation (default: false)

Key Features:
- Monotone B-spline calibration preserves score ordering
- Adaptive knot placement with emphasis on high-score regions
- Format preservation for input/output data (CSV/TSV/Parquet)
- Nested tarball support for SageMaker training job outputs
- Comprehensive metrics: ECE, MCE, Brier score, AUC
- Visual reliability diagrams for calibration quality assessment
- Quality validation with automatic status determination

Calibration Quality Metrics:
- Expected Calibration Error (ECE): Average calibration error across bins
- Maximum Calibration Error (MCE): Worst-case calibration error
- Brier Score: Mean squared difference between predictions and outcomes
- AUC-ROC: Area under receiver operating characteristic curve
- Model MSE: Mean squared error of fitted B-spline
- Coefficient magnitude: Maximum absolute coefficient value
- Unique predictions: Number of distinct calibrated probabilities

Success Criteria:
- Convergence: IRLS optimization converges within MAX_ITER iterations
- No NaN coefficients: All B-spline coefficients are finite
- Sufficient unique values: At least MIN_UNIQUE_VALUES distinct predictions
- MSE improvement: Model MSE better than baseline (mean prediction)
- Coefficient stability: Maximum coefficient below MAX_COEF_THRESHOLD

Supported Job Types:
- training: Extracts data from nested tarballs (output.tar.gz -> val.tar.gz/test.tar.gz)
- calibration: Standard data loading for dedicated calibration datasets
- validation: Standard data loading for validation datasets
- testing: Standard data loading for test datasets

Performance Optimizations:
The script includes transparent I/O optimizations that automatically improve performance
without requiring any configuration changes:

- PyArrow-based Parquet I/O:
  * Uses PyArrow engine for Parquet files when available (30-50% faster loading)
  * Writes Parquet with Snappy compression (40-60% smaller files, 20-30% faster)
  * Automatic fallback to default pandas engine if PyArrow unavailable
  * No configuration required - optimizations are transparent

- Enhanced Logging:
  * Tracks file sizes, row counts, and column counts during I/O operations
  * Reports detected file formats and compression ratios
  * Provides visibility into I/O performance

- Format Preservation:
  * Input format automatically detected (CSV, TSV, or Parquet)
  * Output saved in same format as input for consistency
  * Parquet format recommended for best performance (2x faster, 60% smaller)

Expected Performance:
- Small datasets (<100K rows): 20-40% faster I/O
- Medium datasets (1M rows): 30-50% faster I/O, 40-60% smaller files
- Large datasets (>10M rows): 40-60% faster I/O, significant memory savings

The optimizations primarily benefit data I/O operations. Calibration optimization itself
(B-spline fitting) remains CPU-bound and represents the main processing time. Overall
speedup for typical calibration workflows: ~30-40% faster end-to-end.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `preprocessor_input` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [TSAPreprocessing](tsa_preprocessing.md), [TSATabularPreprocessing](tsa_tabular_preprocessing.md), DataPreprocessing |
| `evaluation_data` | `processing_output` | yes | [TSATraining](tsa_training.md), [TSAModelEval](tsa_model_eval.md), [PyTorchTraining](pytorch_training.md), [PyTorchModelEval](pytorch_model_eval.md), [XGBoostTraining](xgboost_training.md), [XGBoostModelEval](xgboost_model_eval.md), [LightGBMTraining](lightgbm_training.md), [LightGBMModelEval](lightgbm_model_eval.md), ModelEvaluation, TrainingEvaluation |

## Outputs

| Output | Type |
|--------|------|
| `calibration_output` | `processing_output` |
| `metrics_output` | `processing_output` |
| `calibrated_data` | `processing_output` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `numpy` | `>=1.21.0` |
| `scipy` | `>=1.7.0` |
| `pandas` | `>=1.3.0` |
| `scikit-learn` | `>=1.0.0` |
| `matplotlib` | `>=3.3.0` |

---

← [Back to the Step Catalog](index.md)
