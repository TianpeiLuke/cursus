#!/usr/bin/env python
import os
import json
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
import xgboost as xgb
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

# Embedded processor classes to remove external dependencies

class RiskTableMappingProcessor:
    """
    A processor that performs risk-table-based mapping on a specified categorical variable.
    The 'process' method (called via __call__) handles single values.
    The 'transform' method handles pandas Series or DataFrames.
    """

    def __init__(
        self,
        column_name: str,
        label_name: str,
        smooth_factor: float = 0.0,
        count_threshold: int = 0,
        risk_tables: Optional[Dict] = None,
    ):
        """
        Initialize RiskTableMappingProcessor.

        Args:
            column_name: Name of the categorical column to be binned.
            label_name: Name of label/target variable (expected to be binary 0 or 1).
            smooth_factor: Smoothing factor for risk calculation (0 to 1).
            count_threshold: Minimum count for considering a category's calculated risk.
            risk_tables: Optional pre-computed risk tables.
        """
        self.processor_name = "risk_table_mapping_processor"
        self.function_name_list = ["process", "transform", "fit"]

        if not isinstance(column_name, str) or not column_name:
            raise ValueError("column_name must be a non-empty string.")
        self.column_name = column_name
        self.label_name = label_name
        self.smooth_factor = smooth_factor
        self.count_threshold = count_threshold

        self.is_fitted = False
        if risk_tables:
            self._validate_risk_tables(risk_tables)
            self.risk_tables = risk_tables
            self.is_fitted = True
        else:
            self.risk_tables = {}

    def get_name(self) -> str:
        return self.processor_name

    def _validate_risk_tables(self, risk_tables: Dict) -> None:
        if not isinstance(risk_tables, dict):
            raise ValueError("Risk tables must be a dictionary.")
        if "bins" not in risk_tables or "default_bin" not in risk_tables:
            raise ValueError("Risk tables must contain 'bins' and 'default_bin' keys.")
        if not isinstance(risk_tables["bins"], dict):
            raise ValueError("Risk tables 'bins' must be a dictionary.")
        if not isinstance(
            risk_tables["default_bin"], (int, float, np.floating, np.integer)
        ):
            raise ValueError(
                f"Risk tables 'default_bin' must be a number, got {type(risk_tables['default_bin'])}."
            )

    def set_risk_tables(self, risk_tables: Dict) -> None:
        self._validate_risk_tables(risk_tables)
        self.risk_tables = risk_tables
        self.is_fitted = True

    def process(self, input_value: Any) -> float:
        """
        Process a single input value (for the configured 'column_name'),
        mapping it to its binned risk value.
        This method is called when the processor instance is called as a function.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor must be fitted or initialized with risk tables before processing."
            )
        str_value = str(input_value)
        return self.risk_tables["bins"].get(str_value, self.risk_tables["default_bin"])

    def transform(
        self, data: Union[pd.DataFrame, pd.Series, Any]
    ) -> Union[pd.DataFrame, pd.Series, float]:
        """
        Transform data using the computed risk tables.
        - If data is a DataFrame, transforms the 'column_name' Series within it.
        - If data is a Series, transforms the Series (assumed to be the target column).
        - If data is a single value, uses the 'process' method.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor must be fitted or initialized with risk tables before transforming."
            )

        if isinstance(data, pd.DataFrame):
            if self.column_name not in data.columns:
                raise ValueError(
                    f"Column '{self.column_name}' not found in input DataFrame for transform operation."
                )
            output_data = data.copy()
            output_data[self.column_name] = (
                data[self.column_name]
                .astype(str)
                .map(self.risk_tables["bins"])
                .fillna(self.risk_tables["default_bin"])
            )
            return output_data
        elif isinstance(data, pd.Series):
            return (
                data.astype(str)
                .map(self.risk_tables["bins"])
                .fillna(self.risk_tables["default_bin"])
            )
        else:
            return self.process(data)  # Consistent with __call__

    def get_risk_tables(self) -> Dict:
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor has not been fitted or initialized with risk tables."
            )
        return self.risk_tables


class NumericalVariableImputationProcessor:
    """
    A processor that performs imputation on numerical variables using predefined or computed values.
    Supports mean, median, and mode imputation strategies.
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        imputation_dict: Optional[Dict[str, Union[int, float]]] = None,
        strategy: str = "mean",
    ):
        self.processor_name = "numerical_variable_imputation_processor"
        self.function_name_list = ["fit", "process", "transform"]

        self.variables = variables
        self.strategy = strategy
        self.is_fitted = False

        if imputation_dict:
            self._validate_imputation_dict(imputation_dict)
            self.imputation_dict = imputation_dict
            self.is_fitted = True
        else:
            self.imputation_dict = None

    def get_name(self) -> str:
        return self.processor_name

    def __call__(self, input_data):
        return self.process(input_data)

    def _validate_imputation_dict(self, imputation_dict: Dict[str, Any]) -> None:
        if not isinstance(imputation_dict, dict):
            raise ValueError("imputation_dict must be a dictionary")
        if not imputation_dict:
            raise ValueError("imputation_dict cannot be empty")
        for k, v in imputation_dict.items():
            if not isinstance(k, str):
                raise ValueError(f"All keys must be strings, got {type(k)} for key {k}")
            if not isinstance(v, (int, float, np.number)):
                raise ValueError(
                    f"All values must be numeric, got {type(v)} for key {k}"
                )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError(
                "Processor is not fitted. Call 'fit' with appropriate arguments before using this method."
            )

        output_data = input_data.copy()
        for var, value in input_data.items():
            if var in self.imputation_dict and pd.isna(value):
                output_data[var] = self.imputation_dict[var]
        return output_data

    def transform(
        self, X: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform input data by imputing missing values.

        Args:
            X: Input DataFrame or Series

        Returns:
            Transformed DataFrame or Series with imputed values
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Processor is not fitted. Call 'fit' with appropriate arguments before using this method."
            )

        # Handle Series input
        if isinstance(X, pd.Series):
            if X.name not in self.imputation_dict:
                raise ValueError(f"No imputation value found for series name: {X.name}")
            return X.fillna(self.imputation_dict[X.name])

        # Handle DataFrame input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be pandas Series or DataFrame")

        # Make a copy to avoid modifying the input
        df = X.copy()

        # Apply imputation only to variables in imputation_dict and only to NaN values
        for var, impute_value in self.imputation_dict.items():
            if var in df.columns:
                # Create mask for NaN values
                nan_mask = df[var].isna()
                # Only replace NaN values
                df.loc[nan_mask, var] = impute_value

        return df

    def get_params(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "imputation_dict": self.imputation_dict,
            "strategy": self.strategy,
        }

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "MODEL_DIR": "/opt/ml/processing/input/model",
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_EVAL_DIR": "/opt/ml/processing/output/eval",
}


def load_model_artifacts(model_dir: str) -> Tuple[xgb.Booster, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Load the trained XGBoost model and all preprocessing artifacts from the specified directory.
    Returns model, risk_tables, impute_dict, feature_columns, and hyperparameters.
    """
    logger.info(f"Loading model artifacts from {model_dir}")
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost_model.bst"))
    logger.info("Loaded xgboost_model.bst")
    with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
        risk_tables = pkl.load(f)
    logger.info("Loaded risk_table_map.pkl")
    with open(os.path.join(model_dir, "impute_dict.pkl"), "rb") as f:
        impute_dict = pkl.load(f)
    logger.info("Loaded impute_dict.pkl")
    with open(os.path.join(model_dir, "feature_columns.txt"), "r") as f:
        feature_columns = [
            line.strip().split(",")[1] for line in f if not line.startswith("#")
        ]
    logger.info(f"Loaded feature_columns.txt: {feature_columns}")
    with open(os.path.join(model_dir, "hyperparameters.json"), "r") as f:
        hyperparams = json.load(f)
    logger.info("Loaded hyperparameters.json")
    return model, risk_tables, impute_dict, feature_columns, hyperparams


def preprocess_inference_data(
    df: pd.DataFrame, 
    feature_columns: List[str], 
    risk_tables: Dict[str, Any], 
    impute_dict: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply risk table mapping and numerical imputation to inference data.
    Preserves all original columns while ensuring features are model-ready.
    """
    # Preserve original dataframe structure
    result_df = df.copy()
    
    # Get available feature columns
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Found {len(available_features)} out of {len(feature_columns)} expected feature columns")
    
    # Apply risk table mapping for categorical features
    logger.info("Starting risk table mapping for categorical features")
    for feature, risk_table in risk_tables.items():
        if feature in available_features:
            logger.info(f"Applying risk table mapping for feature: {feature}")
            processor = RiskTableMappingProcessor(
                column_name=feature, 
                label_name="label", 
                risk_tables=risk_table
            )
            result_df[feature] = processor.transform(df[feature])
    logger.info("Risk table mapping complete")
    
    # Apply numerical imputation
    logger.info("Starting numerical imputation")
    feature_df = result_df[available_features].copy()
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    imputed_df = imputer.transform(feature_df)
    
    # Update feature columns in result dataframe
    for col in available_features:
        if col in imputed_df:
            result_df[col] = imputed_df[col]
    logger.info("Numerical imputation complete")
    
    # Ensure feature columns are numeric
    logger.info("Ensuring feature columns are numeric")
    result_df[available_features] = (
        result_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    
    logger.info(f"Preprocessed data shape: {result_df.shape} (preserving all original columns)")
    return result_df


def generate_predictions(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_columns: List[str],
    hyperparams: Dict[str, Any],
) -> np.ndarray:
    """
    Generate predictions using the XGBoost model.
    Handles both binary and multiclass scenarios.
    """
    # Get available features for prediction
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using {len(available_features)} features for prediction")
    X = df[available_features].values
    
    # Create XGBoost DMatrix
    dmatrix = xgb.DMatrix(X)
    
    # Generate predictions
    y_prob = model.predict(dmatrix)
    logger.info(f"Model prediction shape: {y_prob.shape}")
    
    # Handle binary vs multiclass output format
    if len(y_prob.shape) == 1:
        # Binary classification - convert to two-column probabilities
        y_prob = np.column_stack([1 - y_prob, y_prob])
        logger.info("Converted binary prediction to two-column probabilities")
    
    return y_prob


def load_eval_data(eval_data_dir: str) -> pd.DataFrame:
    """
    Load the first .csv or .parquet file found in the evaluation data directory.
    Returns a pandas DataFrame.
    """
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".parquet"]
        ]
    )
    if not eval_files:
        logger.error("No eval data file found in eval_data input.")
        raise RuntimeError("No eval data file found in eval_data input.")
    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")
    if eval_file.suffix == ".parquet":
        df = pd.read_parquet(eval_file)
    else:
        df = pd.read_csv(eval_file)
    logger.info(f"Loaded eval data shape: {df.shape}")
    return df


def get_id_label_columns(df: pd.DataFrame, id_field: str, label_field: str) -> Tuple[str, str]:
    """
    Determine the ID and label columns in the DataFrame.
    Falls back to the first and second columns if not found.
    """
    id_col = id_field if id_field in df.columns else df.columns[0]
    label_col = label_field if label_field in df.columns else df.columns[1]
    logger.info(f"Using id_col: {id_col}, label_col: {label_col}")
    return id_col, label_col


def save_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    output_dir: str,
    format: str = "csv",
    id_col: str = "id",
    label_col: str = "label",
    json_orient: str = "records"
) -> str:
    """
    Save predictions with original data in specified format.
    Supports CSV, Parquet, and JSON formats.
    """
    logger.info(f"Saving predictions to {output_dir} in {format} format")
    
    # Create output dataframe with original data
    output_df = df.copy()
    
    # Add prediction columns
    n_classes = predictions.shape[1]
    for i in range(n_classes):
        output_df[f"prob_class_{i}"] = predictions[:, i]
    
    # Save in specified format
    os.makedirs(output_dir, exist_ok=True)
    
    if format.lower() == "parquet":
        output_path = os.path.join(output_dir, "predictions.parquet")
        output_df.to_parquet(output_path, index=False)
    elif format.lower() == "json":
        output_path = os.path.join(output_dir, "predictions.json")
        # Convert numpy types to native Python types for JSON serialization
        output_df_json = output_df.copy()
        for col in output_df_json.columns:
            if output_df_json[col].dtype == 'object':
                continue
            elif 'int' in str(output_df_json[col].dtype):
                output_df_json[col] = output_df_json[col].astype(int)
            elif 'float' in str(output_df_json[col].dtype):
                output_df_json[col] = output_df_json[col].astype(float)
        
        # Save as JSON with specified orientation
        output_df_json.to_json(output_path, orient=json_orient, indent=2)
    else:  # Default to CSV
        output_path = os.path.join(output_dir, "predictions.csv")
        output_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved predictions to {output_path}")
    return output_path


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main entry point for XGBoost model inference script.
    Loads model and data, runs inference, and saves predictions.

    Args:
        input_paths (Dict[str, str]): Dictionary of input paths
        output_paths (Dict[str, str]): Dictionary of output paths
        environ_vars (Dict[str, str]): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
    """
    # Extract paths from parameters - using contract-defined logical names
    model_dir = input_paths.get("model_input", input_paths.get("model_dir"))
    eval_data_dir = input_paths.get("processed_data", input_paths.get("eval_data_dir"))
    output_eval_dir = output_paths.get(
        "eval_output", output_paths.get("output_eval_dir")
    )

    # Extract environment variables
    id_field = environ_vars.get("ID_FIELD", "id")
    label_field = environ_vars.get("LABEL_FIELD", "label")
    output_format = environ_vars.get("OUTPUT_FORMAT", "csv")
    json_orient = environ_vars.get("JSON_ORIENT", "records")

    # Log job info
    job_type = job_args.job_type
    logger.info(f"Running model inference with job_type: {job_type}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)

    logger.info("Starting model inference script")

    # Load model artifacts
    model, risk_tables, impute_dict, feature_columns, hyperparams = (
        load_model_artifacts(model_dir)
    )

    # Load and preprocess data
    df = load_eval_data(eval_data_dir)
    
    # Get ID and label columns before preprocessing
    id_col, label_col = get_id_label_columns(df, id_field, label_field)
    
    # Process the data - preserves all columns including id and label
    df = preprocess_inference_data(df, feature_columns, risk_tables, impute_dict)
    
    logger.info(f"Final inference DataFrame shape: {df.shape}")

    # Get the available features (those that exist in the DataFrame)
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Generate predictions
    predictions = generate_predictions(model, df, available_features, hyperparams)
    
    # Save predictions with original data
    output_path = save_predictions(
        df, 
        predictions, 
        output_eval_dir, 
        format=output_format,
        id_col=id_col,
        label_col=label_col,
        json_orient=json_orient
    )

    logger.info("Model inference script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths using contract-defined paths only
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
    }

    # Collect environment variables - ID_FIELD and LABEL_FIELD are required per contract
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),  # Fallback for testing
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),  # Fallback for testing
        "OUTPUT_FORMAT": os.environ.get("OUTPUT_FORMAT", "csv"),  # csv, parquet, or json
        "JSON_ORIENT": os.environ.get("JSON_ORIENT", "records"),  # JSON orientation
    }

    try:
        # Call main function with testability parameters
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["eval_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["eval_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("eval_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)
