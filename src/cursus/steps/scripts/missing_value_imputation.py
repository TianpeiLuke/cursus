#!/usr/bin/env python
"""
Missing Value Imputation Processing Script

This script handles missing value imputation for tabular data using simple statistical methods.
It supports both training mode (fit and transform) and inference mode (transform only).
Follows the same pattern as risk_table_mapping.py for consistency.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import pickle as pkl
import traceback
from pathlib import Path
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime

# Default paths (will be overridden by parameters in main function)
DEFAULT_INPUT_DIR = "/opt/ml/processing/input/data"
DEFAULT_OUTPUT_DIR = "/opt/ml/processing/output"
DEFAULT_IMPUTATION_PARAMS_DIR = "/opt/ml/processing/input/imputation_params"

# Constants for file paths to ensure consistency between training and inference
IMPUTATION_PARAMS_FILENAME = "imputation_parameters.pkl"
IMPUTATION_SUMMARY_FILENAME = "imputation_summary.json"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_split_data(job_type: str, input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load data according to job_type, following risk_table_mapping pattern.
    
    For 'training': Loads data from train, test, and val subdirectories
    For others: Loads single job_type split
    """
    input_path = Path(input_dir)

    if job_type == "training":
        # For training, we expect data in train/test/val subdirectories
        train_df = pd.read_csv(input_path / "train" / "train_processed_data.csv")
        test_df = pd.read_csv(input_path / "test" / "test_processed_data.csv")
        val_df = pd.read_csv(input_path / "val" / "val_processed_data.csv")
        logger.info(
            f"Loaded training data splits: train={train_df.shape}, test={test_df.shape}, val={val_df.shape}"
        )
        return {"train": train_df, "test": test_df, "val": val_df}
    else:
        # For other job types, we expect data in a single directory named after job_type
        df = pd.read_csv(input_path / job_type / f"{job_type}_processed_data.csv")
        logger.info(f"Loaded {job_type} data: {df.shape}")
        return {job_type: df}


def save_output_data(
    job_type: str, output_dir: str, data_dict: Dict[str, pd.DataFrame]
) -> None:
    """
    Save processed data according to job_type, following risk_table_mapping pattern.

    For 'training': Saves data to train, test, and val subdirectories
    For others: Saves to single job_type directory
    """
    output_path = Path(output_dir)

    for split_name, df in data_dict.items():
        split_output_dir = output_path / split_name
        split_output_dir.mkdir(exist_ok=True, parents=True)

        output_file = split_output_dir / f"{split_name}_processed_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {split_name} data to {output_file}, shape: {df.shape}")


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive missing value analysis for imputation planning.
    """
    missing_analysis = {
        "total_records": len(df),
        "columns_with_missing": {},
        "missing_patterns": {},
        "data_types": {},
        "imputation_recommendations": {}
    }
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        if missing_count > 0:
            missing_analysis["columns_with_missing"][col] = {
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_percentage),
                "data_type": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(5).tolist()
            }
            
            # Recommend imputation strategy based on data type and distribution
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    skewness = df[col].skew()
                    if abs(skewness) > 1:  # Highly skewed
                        missing_analysis["imputation_recommendations"][col] = "median"
                    else:
                        missing_analysis["imputation_recommendations"][col] = "mean"
                except:
                    missing_analysis["imputation_recommendations"][col] = "mean"
            else:
                missing_analysis["imputation_recommendations"][col] = "mode"
        
        missing_analysis["data_types"][col] = str(df[col].dtype)
    
    # Analyze missing patterns
    missing_pattern = df.isnull().sum(axis=1)
    missing_analysis["missing_patterns"] = {
        "records_with_no_missing": int((missing_pattern == 0).sum()),
        "records_with_missing": int((missing_pattern > 0).sum()),
        "max_missing_per_record": int(missing_pattern.max()),
        "avg_missing_per_record": float(missing_pattern.mean())
    }
    
    return missing_analysis


def validate_imputation_data(
    df: pd.DataFrame, 
    label_field: str,
    exclude_columns: List[str] = None
) -> Dict[str, Any]:
    """
    Validate data for imputation processing.
    """
    exclude_columns = exclude_columns or []
    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "imputable_columns": [],
        "excluded_columns": exclude_columns.copy()
    }
    
    # Check if label field exists and exclude it from imputation
    if label_field in df.columns:
        validation_report["excluded_columns"].append(label_field)
    else:
        validation_report["warnings"].append(f"Label field '{label_field}' not found in data")
    
    # Identify columns suitable for imputation
    for col in df.columns:
        if col not in validation_report["excluded_columns"]:
            if df[col].isnull().any():
                validation_report["imputable_columns"].append(col)
    
    if not validation_report["imputable_columns"]:
        validation_report["warnings"].append("No columns with missing values found for imputation")
    
    return validation_report


def load_imputation_config(environ_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    Load imputation configuration from environment variables.
    """
    config = {
        "default_numerical_strategy": environ_vars.get("DEFAULT_NUMERICAL_STRATEGY", "mean"),
        "default_categorical_strategy": environ_vars.get("DEFAULT_CATEGORICAL_STRATEGY", "mode"),
        "default_text_strategy": environ_vars.get("DEFAULT_TEXT_STRATEGY", "mode"),
        "numerical_constant_value": float(environ_vars.get("NUMERICAL_CONSTANT_VALUE", "0")),
        "categorical_constant_value": environ_vars.get("CATEGORICAL_CONSTANT_VALUE", "Unknown"),
        "text_constant_value": environ_vars.get("TEXT_CONSTANT_VALUE", "Unknown"),
        "categorical_preserve_dtype": environ_vars.get("CATEGORICAL_PRESERVE_DTYPE", "true").lower() == "true",
        "auto_detect_categorical": environ_vars.get("AUTO_DETECT_CATEGORICAL", "true").lower() == "true",
        "categorical_unique_ratio_threshold": float(environ_vars.get("CATEGORICAL_UNIQUE_RATIO_THRESHOLD", "0.1")),
        "validate_fill_values": environ_vars.get("VALIDATE_FILL_VALUES", "true").lower() == "true",
        "column_strategies": {},
        "exclude_columns": environ_vars.get("EXCLUDE_COLUMNS", "").split(",") if environ_vars.get("EXCLUDE_COLUMNS") else []
    }
    
    # Parse column-specific strategies from environment variables
    # Format: COLUMN_STRATEGY_<column_name>=<strategy>
    for key, value in environ_vars.items():
        if key.startswith("COLUMN_STRATEGY_"):
            column_name = key.replace("COLUMN_STRATEGY_", "").lower()
            config["column_strategies"][column_name] = value.lower()
    
    return config


def get_pandas_na_values() -> set:
    """
    Get set of values that pandas interprets as NA/NULL.
    """
    # Common pandas NA values to avoid
    return {
        "N/A", "NA", "NULL", "NaN", "nan", "NAN",
        "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", 
        "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
        "null", "Null", "none", "None", "NONE"
    }


def validate_text_fill_value(value: str) -> bool:
    """
    Validate that a text fill value won't be interpreted as NA by pandas.
    """
    pandas_na_values = get_pandas_na_values()
    return value not in pandas_na_values


def detect_column_type(df: pd.DataFrame, column: str, config: Dict[str, Any]) -> str:
    """
    Enhanced data type detection for imputation strategy selection.
    """
    if pd.api.types.is_numeric_dtype(df[column]):
        return "numerical"
    elif pd.api.types.is_categorical_dtype(df[column]):
        return "categorical"
    elif df[column].dtype == 'object':
        if config.get("auto_detect_categorical", True):
            # Distinguish between text and categorical based on unique values
            non_null_count = df[column].dropna().shape[0]
            if non_null_count > 0:
                unique_ratio = df[column].nunique() / non_null_count
                threshold = config.get("categorical_unique_ratio_threshold", 0.1)
                if unique_ratio < threshold:
                    return "categorical"
        return "text"
    else:
        return "text"  # Default for other types


class ImputationStrategyManager:
    """
    Enhanced strategy manager supporting numerical, text, and categorical data types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pandas_na_values = get_pandas_na_values()
        
    def get_strategy_for_column(self, df: pd.DataFrame, column: str) -> SimpleImputer:
        """
        Enhanced strategy selection supporting text and categorical types.
        """
        # Detect column type using enhanced detection
        column_type = detect_column_type(df, column, self.config)
        
        # Check if strategy is explicitly configured
        if column in self.config.get("column_strategies", {}):
            strategy_name = self.config["column_strategies"][column]
            return self._create_strategy_from_name(df, column, column_type, strategy_name)
        
        # Auto-select based on detected type
        if column_type == "numerical":
            default_strategy = self.config.get("default_numerical_strategy", "mean")
        elif column_type == "categorical":
            default_strategy = self.config.get("default_categorical_strategy", "mode")
        else:  # text
            default_strategy = self.config.get("default_text_strategy", "mode")
        
        return self._create_strategy_from_name(df, column, column_type, default_strategy)
    
    def _create_strategy_from_name(self, df: pd.DataFrame, column: str, column_type: str, strategy_name: str) -> SimpleImputer:
        """
        Create appropriate imputation strategy based on column type and strategy name.
        """
        if column_type == "numerical":
            return self._create_numerical_strategy(strategy_name)
        elif column_type == "categorical":
            return self._create_categorical_strategy(df, column, strategy_name)
        else:  # text
            return self._create_text_strategy(strategy_name)
    
    def _create_numerical_strategy(self, strategy_name: str) -> SimpleImputer:
        """
        Create numerical imputation strategy.
        """
        if strategy_name == "mean":
            return SimpleImputer(strategy="mean")
        elif strategy_name == "median":
            return SimpleImputer(strategy="median")
        elif strategy_name == "constant":
            fill_value = self.config.get("numerical_constant_value", 0)
            return SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            logger.warning(f"Unknown numerical strategy '{strategy_name}', using mean")
            return SimpleImputer(strategy="mean")
    
    def _create_categorical_strategy(self, df: pd.DataFrame, column: str, strategy_name: str) -> SimpleImputer:
        """
        Create categorical imputation strategy with dtype preservation.
        """
        if strategy_name == "mode":
            return SimpleImputer(strategy="most_frequent")
        elif strategy_name == "constant":
            fill_value = self.config.get("categorical_constant_value", "Unknown")
            # Validate fill value is pandas-safe
            if self.config.get("validate_fill_values", True) and fill_value in self.pandas_na_values:
                logger.warning(f"Categorical fill value '{fill_value}' may be interpreted as NA by pandas. Using 'Missing' instead.")
                fill_value = "Missing"
            return SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            logger.warning(f"Unknown categorical strategy '{strategy_name}', using mode")
            return SimpleImputer(strategy="most_frequent")
    
    def _create_text_strategy(self, strategy_name: str) -> SimpleImputer:
        """
        Create text-specific imputation strategy with pandas-safe values.
        """
        if strategy_name == "mode":
            return SimpleImputer(strategy="most_frequent")
        elif strategy_name == "constant":
            fill_value = self.config.get("text_constant_value", "Unknown")
            # Validate fill value is pandas-safe
            if self.config.get("validate_fill_values", True) and fill_value in self.pandas_na_values:
                logger.warning(f"Text fill value '{fill_value}' may be interpreted as NA by pandas. Using 'Unknown' instead.")
                fill_value = "Unknown"
            return SimpleImputer(strategy="constant", fill_value=fill_value)
        elif strategy_name == "empty":
            return SimpleImputer(strategy="constant", fill_value="")
        else:
            logger.warning(f"Unknown text strategy '{strategy_name}', using mode")
            return SimpleImputer(strategy="most_frequent")


class SimpleImputationEngine:
    """
    Core engine for simple statistical imputation methods.
    """
    
    def __init__(self, strategy_manager: ImputationStrategyManager, label_field: str):
        self.strategy_manager = strategy_manager
        self.label_field = label_field
        self.fitted_imputers = {}
        self.imputation_statistics = {}
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit imputation parameters on training data.
        """
        logger.info("Fitting imputation parameters on training data")
        
        # Get columns to impute (exclude label and other specified columns)
        exclude_cols = [self.label_field] + self.strategy_manager.config.get("exclude_columns", [])
        imputable_columns = [col for col in df.columns 
                           if col not in exclude_cols and df[col].isnull().any()]
        
        logger.info(f"Columns to impute: {imputable_columns}")
        
        for column in imputable_columns:
            # Get appropriate strategy for this column
            imputer = self.strategy_manager.get_strategy_for_column(df, column)
            
            # Fit the imputer on non-null values
            column_data = df[[column]]
            imputer.fit(column_data)
            
            # Store fitted imputer
            self.fitted_imputers[column] = imputer
            
            # Store imputation statistics
            self.imputation_statistics[column] = {
                "strategy": imputer.strategy,
                "fill_value": getattr(imputer, "fill_value", None),
                "statistics": getattr(imputer, "statistics_", None),
                "missing_count_training": int(df[column].isnull().sum()),
                "missing_percentage_training": float((df[column].isnull().sum() / len(df)) * 100),
                "data_type": str(df[column].dtype)
            }
            
            logger.info(f"Fitted imputer for column '{column}': {imputer.strategy}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted imputation to data.
        """
        logger.info("Applying imputation to data")
        df_imputed = df.copy()
        
        transformation_log = {}
        
        for column, imputer in self.fitted_imputers.items():
            if column in df_imputed.columns:
                # Count missing values before imputation
                missing_before = df_imputed[column].isnull().sum()
                
                if missing_before > 0:
                    # Apply imputation
                    column_data = df_imputed[[column]]
                    imputed_data = imputer.transform(column_data)
                    df_imputed[column] = imputed_data[:, 0]
                    
                    # Count missing values after imputation
                    missing_after = df_imputed[column].isnull().sum()
                    
                    transformation_log[column] = {
                        "missing_before": int(missing_before),
                        "missing_after": int(missing_after),
                        "imputed_count": int(missing_before - missing_after),
                        "strategy_used": imputer.strategy
                    }
                    
                    logger.info(f"Imputed {missing_before - missing_after} values in column '{column}'")
                else:
                    transformation_log[column] = {
                        "missing_before": 0,
                        "missing_after": 0,
                        "imputed_count": 0,
                        "strategy_used": imputer.strategy
                    }
        
        self.last_transformation_log = transformation_log
        return df_imputed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit imputation parameters and transform data in one step.
        """
        self.fit(df)
        return self.transform(df)
    
    def get_imputation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of imputation process.
        """
        return {
            "fitted_columns": list(self.fitted_imputers.keys()),
            "imputation_statistics": self.imputation_statistics,
            "last_transformation_log": getattr(self, "last_transformation_log", {}),
            "total_imputers": len(self.fitted_imputers)
        }


def save_imputation_artifacts(
    imputation_engine: SimpleImputationEngine,
    imputation_config: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save imputation artifacts to the specified output path.
    
    Args:
        imputation_engine: SimpleImputationEngine instance with fitted parameters
        imputation_config: Imputation configuration dictionary
        output_path: Path to save artifacts to
    """
    # Prepare imputation parameters for serialization
    imputation_parameters = {
        "fitted_imputers": imputation_engine.fitted_imputers,
        "imputation_statistics": imputation_engine.imputation_statistics,
        "config": imputation_config,
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Save imputation parameters - using consistent filename
    params_output_path = output_path / IMPUTATION_PARAMS_FILENAME
    with open(params_output_path, "wb") as f:
        pkl.dump(imputation_parameters, f)
    logger.info(f"Saved imputation parameters to {params_output_path}")
    logger.info(f"This file can be used as input for non-training jobs")
    
    # Save human-readable summary
    summary = imputation_engine.get_imputation_summary()
    summary_output_path = output_path / IMPUTATION_SUMMARY_FILENAME
    with open(summary_output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved imputation summary to {summary_output_path}")


def load_imputation_parameters(imputation_params_path: Path) -> Dict:
    """
    Load imputation parameters from a pickle file.
    
    Args:
        imputation_params_path: Path to the imputation parameters file
    
    Returns:
        Dictionary of imputation parameters
    """
    if not imputation_params_path.exists():
        raise FileNotFoundError(f"Imputation parameters file not found: {imputation_params_path}")
    
    logger.info(f"Loading imputation parameters from {imputation_params_path}")
    with open(imputation_params_path, "rb") as f:
        imputation_parameters = pkl.load(f)
    
    logger.info(f"Successfully loaded imputation parameters with {len(imputation_parameters['fitted_imputers'])} imputers")
    return imputation_parameters


def process_data(
    data_dict: Dict[str, pd.DataFrame],
    label_field: str,
    job_type: str,
    imputation_config: Dict[str, Any],
    imputation_parameters: Optional[Dict] = None
) -> Tuple[Dict[str, pd.DataFrame], SimpleImputationEngine]:
    """
    Core data processing logic for missing value imputation.
    
    Args:
        data_dict: Dictionary of dataframes keyed by split name
        label_field: Target column name
        job_type: Type of job (training, validation, testing, calibration)
        imputation_config: Imputation configuration dictionary
        imputation_parameters: Pre-fitted imputation parameters (for non-training jobs)
    
    Returns:
        Tuple containing:
        - Dictionary of imputed dataframes
        - SimpleImputationEngine instance with fitted parameters
    """
    strategy_manager = ImputationStrategyManager(imputation_config)
    imputation_engine = SimpleImputationEngine(strategy_manager, label_field)
    
    if job_type == "training":
        logger.info("Running in 'training' mode: fitting on train data, transforming all splits")
        
        # Fit imputation parameters on training data only
        imputation_engine.fit(data_dict["train"])
        
        # Transform all splits
        transformed_data = {}
        for split_name, df in data_dict.items():
            df_imputed = imputation_engine.transform(df)
            transformed_data[split_name] = df_imputed
            logger.info(f"Imputed {split_name} data, shape: {df_imputed.shape}")
    
    else:
        # Non-training mode: load pre-fitted parameters
        if not imputation_parameters:
            raise ValueError("For non-training job types, imputation_parameters must be provided")
        
        # Load pre-fitted imputation parameters
        imputation_engine.fitted_imputers = imputation_parameters["fitted_imputers"]
        imputation_engine.imputation_statistics = imputation_parameters["imputation_statistics"]
        
        logger.info(f"Using pre-fitted imputation parameters with {len(imputation_engine.fitted_imputers)} imputers")
        
        # Transform the data
        transformed_data = {}
        for split_name, df in data_dict.items():
            df_imputed = imputation_engine.transform(df)
            transformed_data[split_name] = df_imputed
            logger.info(f"Imputed {split_name} data, shape: {df_imputed.shape}")
    
    return transformed_data, imputation_engine


def generate_imputation_report(
    imputation_engine: SimpleImputationEngine,
    missing_analysis: Dict[str, Any],
    validation_report: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Generate comprehensive imputation report with statistics and insights.
    """
    # Get imputation summary
    imputation_summary = imputation_engine.get_imputation_summary()
    
    # Generate comprehensive report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "missing_value_analysis": missing_analysis,
        "validation_report": validation_report,
        "imputation_summary": imputation_summary,
        "quality_metrics": calculate_imputation_quality_metrics(imputation_summary),
        "recommendations": generate_imputation_recommendations(imputation_summary, missing_analysis)
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, "imputation_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate text summary
    text_summary = generate_imputation_text_summary(report)
    text_path = os.path.join(output_dir, "imputation_summary.txt")
    with open(text_path, "w") as f:
        f.write(text_summary)
    
    return {
        "json_report": json_path,
        "text_summary": text_path
    }


def calculate_imputation_quality_metrics(imputation_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate quality metrics for imputation process.
    """
    quality_metrics = {
        "total_columns_imputed": len(imputation_summary["fitted_columns"]),
        "imputation_coverage": {},
        "strategy_distribution": {},
        "data_type_coverage": {}
    }
    
    # Calculate imputation coverage by column
    for column, stats in imputation_summary["imputation_statistics"].items():
        quality_metrics["imputation_coverage"][column] = {
            "missing_percentage": stats["missing_percentage_training"],
            "strategy_used": stats["strategy"],
            "data_type": stats["data_type"]
        }
    
    # Calculate strategy distribution
    strategies = [stats["strategy"] for stats in imputation_summary["imputation_statistics"].values()]
    strategy_counts = {}
    for strategy in strategies:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    quality_metrics["strategy_distribution"] = strategy_counts
    
    # Calculate data type coverage
    data_types = [stats["data_type"] for stats in imputation_summary["imputation_statistics"].values()]
    type_counts = {}
    for dtype in data_types:
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
    quality_metrics["data_type_coverage"] = type_counts
    
    return quality_metrics


def generate_imputation_recommendations(
    imputation_summary: Dict[str, Any], 
    missing_analysis: Dict[str, Any]
) -> List[str]:
    """
    Generate actionable recommendations based on imputation analysis.
    """
    recommendations = []
    
    # Check for high missing value percentages
    high_missing_columns = []
    for column, stats in imputation_summary["imputation_statistics"].items():
        if stats["missing_percentage_training"] > 50:
            high_missing_columns.append(column)
    
    if high_missing_columns:
        recommendations.append(
            f"Columns with >50% missing values detected: {high_missing_columns}. "
            "Consider investigating data collection issues or using advanced imputation methods."
        )
    
    # Check strategy appropriateness
    numerical_mode_columns = []
    for column, stats in imputation_summary["imputation_statistics"].items():
        if "int" in stats["data_type"] or "float" in stats["data_type"]:
            if stats["strategy"] == "most_frequent":
                numerical_mode_columns.append(column)
    
    if numerical_mode_columns:
        recommendations.append(
            f"Numerical columns using mode imputation: {numerical_mode_columns}. "
            "Consider using mean or median imputation for better statistical properties."
        )
    
    # Check for potential data quality issues
    total_missing_patterns = missing_analysis["missing_patterns"]["records_with_missing"]
    total_records = missing_analysis["total_records"]
    missing_record_percentage = (total_missing_patterns / total_records) * 100
    
    if missing_record_percentage > 30:
        recommendations.append(
            f"{missing_record_percentage:.1f}% of records have missing values. "
            "Consider investigating systematic data collection issues."
        )
    
    # General recommendations
    if len(imputation_summary["fitted_columns"]) > 10:
        recommendations.append(
            "Large number of columns require imputation. Consider feature selection "
            "or advanced imputation methods like MICE for better performance."
        )
    
    return recommendations


def generate_imputation_text_summary(report: Dict[str, Any]) -> str:
    """
    Generate human-readable text summary of imputation process.
    """
    summary_lines = [
        "=" * 60,
        "MISSING VALUE IMPUTATION SUMMARY",
        "=" * 60,
        f"Generated: {report['timestamp']}",
        "",
        "DATA OVERVIEW:",
        f"  Total Records: {report['missing_value_analysis']['total_records']:,}",
        f"  Columns with Missing Values: {len(report['missing_value_analysis']['columns_with_missing'])}",
        f"  Records with Missing Values: {report['missing_value_analysis']['missing_patterns']['records_with_missing']:,}",
        "",
        "IMPUTATION RESULTS:",
        f"  Columns Imputed: {report['quality_metrics']['total_columns_imputed']}",
        f"  Strategy Distribution: {report['quality_metrics']['strategy_distribution']}",
        ""
    ]
    
    # Add column-specific details
    if report['imputation_summary']['imputation_statistics']:
        summary_lines.append("COLUMN DETAILS:")
        for column, stats in report['imputation_summary']['imputation_statistics'].items():
            summary_lines.append(
                f"  {column}: {stats['strategy']} imputation, "
                f"{stats['missing_percentage_training']:.1f}% missing"
            )
        summary_lines.append("")
    
    # Add recommendations
    if report['recommendations']:
        summary_lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            summary_lines.append(f"  {i}. {rec}")
        summary_lines.append("")
    
    summary_lines.append("=" * 60)
    
    return "\n".join(summary_lines)


def internal_main(
    job_type: str,
    input_dir: str,
    output_dir: str,
    imputation_config: Dict[str, Any],
    label_field: str,
    imputation_params_input_dir: Optional[str] = None,
    load_data_func: Callable = load_split_data,
    save_data_func: Callable = save_output_data,
) -> Tuple[Dict[str, pd.DataFrame], SimpleImputationEngine]:
    """
    Main logic for missing value imputation, handling both training and inference modes.

    Args:
        job_type: Type of job (training, validation, testing, calibration)
        input_dir: Input directory for data
        output_dir: Output directory for processed data
        imputation_config: Imputation configuration dictionary
        label_field: Target column name
        imputation_params_input_dir: Directory containing pre-trained imputation parameters (for non-training jobs)
        load_data_func: Function to load data (for dependency injection in tests)
        save_data_func: Function to save data (for dependency injection in tests)

    Returns:
        Tuple containing:
        - Dictionary of imputed dataframes
        - SimpleImputationEngine instance with fitted parameters
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using imputation configuration: {imputation_config}")
    logger.info(f"Label field: {label_field}")

    # Load data according to job type
    data_dict = load_data_func(job_type, input_dir)

    # Load imputation parameters if needed (non-training modes)
    imputation_parameters = None
    if job_type != "training" and imputation_params_input_dir:
        # Use the consistent filename for loading imputation parameters
        imputation_params_path = Path(imputation_params_input_dir) / IMPUTATION_PARAMS_FILENAME
        imputation_parameters = load_imputation_parameters(imputation_params_path)
        logger.info(f"Loaded pre-trained imputation parameters from {imputation_params_path}")

    # Process the data
    transformed_data, imputation_engine = process_data(
        data_dict=data_dict,
        label_field=label_field,
        job_type=job_type,
        imputation_config=imputation_config,
        imputation_parameters=imputation_parameters,
    )

    # Save processed data
    save_data_func(job_type, output_dir, transformed_data)

    # Save fitted artifacts (only for training jobs)
    if job_type == "training":
        save_imputation_artifacts(imputation_engine, imputation_config, output_path)

    # Generate comprehensive report
    if transformed_data:
        sample_df = next(iter(transformed_data.values()))
        missing_analysis = analyze_missing_values(sample_df)
        validation_report = validate_imputation_data(sample_df, label_field)
        generate_imputation_report(imputation_engine, missing_analysis, validation_report, output_dir)

    logger.info("Missing value imputation complete.")
    return transformed_data, imputation_engine


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Tuple[Dict[str, pd.DataFrame], SimpleImputationEngine]:
    """
    Standardized main entry point for missing value imputation script.

    Args:
        input_paths: Dictionary of input paths with logical names
            - "data_input": Input data directory (from tabular_preprocessing)
            - "imputation_params_input": Imputation parameters directory (for non-training jobs)
        output_paths: Dictionary of output paths with logical names
            - "data_output": Output directory for imputed data
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments containing job_type

    Returns:
        Tuple containing:
        - Dictionary of imputed dataframes
        - SimpleImputationEngine instance with fitted parameters
    """
    try:
        # Extract paths from input parameters - required keys must be present
        if "data_input" not in input_paths:
            raise ValueError("Missing required input path: data_input")
        if "data_output" not in output_paths:
            raise ValueError("Missing required output path: data_output")

        # Extract job_type from args
        if job_args is None or not hasattr(job_args, "job_type"):
            raise ValueError("job_args must contain job_type parameter")

        job_type = job_args.job_type
        input_dir = input_paths["data_input"]
        output_dir = output_paths["data_output"]

        # For non-training jobs, check if imputation parameters input path is provided
        imputation_params_input_dir = None
        if job_type != "training":
            imputation_params_input_dir = input_paths.get("imputation_params_input")
            if not imputation_params_input_dir:
                logger.warning(
                    f"No imputation_params_input path provided for non-training job {job_type}. "
                    + "Imputation may fail."
                )

        # Log input/output paths for clarity
        logger.info(f"Input data directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        if imputation_params_input_dir:
            logger.info(f"Imputation parameters input directory: {imputation_params_input_dir}")
            logger.info(
                f"Expected imputation parameters path: {Path(imputation_params_input_dir) / IMPUTATION_PARAMS_FILENAME}"
            )

        # Load imputation configuration from environment variables
        imputation_config = load_imputation_config(environ_vars)
        label_field = environ_vars.get("LABEL_FIELD", "target")

        # Execute the internal main logic
        return internal_main(
            job_type=job_type,
            input_dir=input_dir,
            output_dir=output_dir,
            imputation_config=imputation_config,
            label_field=label_field,
            imputation_params_input_dir=imputation_params_input_dir,
        )

    except Exception as e:
        logger.error(f"Error in missing value imputation: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="Type of job to perform",
        )
        args = parser.parse_args()

        # Define standard SageMaker paths based on contract
        input_paths = {
            "data_input": DEFAULT_INPUT_DIR,
        }

        output_paths = {"data_output": DEFAULT_OUTPUT_DIR}

        # For non-training jobs, add imputation parameters input path
        if args.job_type != "training":
            input_paths["imputation_params_input"] = DEFAULT_IMPUTATION_PARAMS_DIR

        # Environment variables dictionary
        environ_vars = {
            "LABEL_FIELD": os.environ.get("LABEL_FIELD", "target"),
            "DEFAULT_NUMERICAL_STRATEGY": os.environ.get("DEFAULT_NUMERICAL_STRATEGY", "mean"),
            "DEFAULT_CATEGORICAL_STRATEGY": os.environ.get("DEFAULT_CATEGORICAL_STRATEGY", "mode"),
            "DEFAULT_TEXT_STRATEGY": os.environ.get("DEFAULT_TEXT_STRATEGY", "mode"),
            "NUMERICAL_CONSTANT_VALUE": os.environ.get("NUMERICAL_CONSTANT_VALUE", "0"),
            "CATEGORICAL_CONSTANT_VALUE": os.environ.get("CATEGORICAL_CONSTANT_VALUE", "Unknown"),
            "TEXT_CONSTANT_VALUE": os.environ.get("TEXT_CONSTANT_VALUE", "Unknown"),
            "CATEGORICAL_PRESERVE_DTYPE": os.environ.get("CATEGORICAL_PRESERVE_DTYPE", "true"),
            "AUTO_DETECT_CATEGORICAL": os.environ.get("AUTO_DETECT_CATEGORICAL", "true"),
            "CATEGORICAL_UNIQUE_RATIO_THRESHOLD": os.environ.get("CATEGORICAL_UNIQUE_RATIO_THRESHOLD", "0.1"),
            "VALIDATE_FILL_VALUES": os.environ.get("VALIDATE_FILL_VALUES", "true"),
            "EXCLUDE_COLUMNS": os.environ.get("EXCLUDE_COLUMNS", ""),
        }

        # Add column-specific strategies from environment variables
        for key, value in os.environ.items():
            if key.startswith("COLUMN_STRATEGY_"):
                environ_vars[key] = value

        # Execute the main function with standardized inputs
        result, _ = main(input_paths, output_paths, environ_vars, args)

        logger.info(f"Missing value imputation completed successfully")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Error in missing value imputation script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(3)
