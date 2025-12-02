#!/usr/bin/env python
"""Percentile Model Calibration Script for SageMaker Processing.

This script performs percentile score mapping calibration to convert raw model scores
to calibrated percentile values using ROC curve analysis. It replicates the functionality
of percentile_score_mapping.py but follows the cursus framework patterns with
environment variable configuration and standardized I/O channels.
"""

import os
import sys
import json
import logging
import traceback
import argparse
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import roc_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# ============================================================================


def _detect_file_format(file_path: str) -> str:
    """
    Detect the format of a data file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'tsv', or 'parquet'
    """
    from pathlib import Path

    suffix = Path(file_path).suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load DataFrame and detect its format.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (DataFrame, format_string)
    """
    detected_format = _detect_file_format(file_path)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame, output_path: str, format_str: str
) -> str:
    """
    Save DataFrame in specified format.

    Args:
        df: DataFrame to save
        output_path: Base output path (without extension)
        format_str: Format to save in ('csv', 'tsv', or 'parquet')

    Returns:
        Path to saved file
    """
    from pathlib import Path

    output_path = Path(output_path)

    if format_str == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format_str == "tsv":
        file_path = output_path.with_suffix(".tsv")
        df.to_csv(file_path, sep="\t", index=False)
    elif format_str == "parquet":
        file_path = output_path.with_suffix(".parquet")
        df.to_parquet(file_path, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {format_str}")

    return str(file_path)


def parse_score_fields(environ_vars: Dict[str, str]) -> List[str]:
    """
    Parse score fields from environment variables with backward compatibility.

    Priority:
    1. SCORE_FIELDS (multi-task) - comma-separated list
    2. SCORE_FIELD (single-task) - single field fallback

    Args:
        environ_vars: Dictionary of environment variables

    Returns:
        List of score field names to calibrate

    Examples:
        >>> parse_score_fields({"SCORE_FIELDS": "task_0_prob,task_1_prob"})
        ["task_0_prob", "task_1_prob"]

        >>> parse_score_fields({"SCORE_FIELD": "prob_class_1"})
        ["prob_class_1"]
    """
    # Priority 1: Check for multi-task SCORE_FIELDS
    score_fields = environ_vars.get("SCORE_FIELDS", "").strip()

    if score_fields:
        # Parse comma-separated list
        field_list = [f.strip() for f in score_fields.split(",") if f.strip()]
        if field_list:
            logger.info(
                f"Multi-task mode: Parsed {len(field_list)} score fields from SCORE_FIELDS"
            )
            logger.info(f"Score fields: {field_list}")
            return field_list

    # Priority 2: Fallback to single-task SCORE_FIELD
    single_field = environ_vars.get("SCORE_FIELD", "prob_class_1")
    logger.info(f"Single-task mode: Using SCORE_FIELD={single_field}")
    return [single_field]


def validate_score_fields(
    df: pd.DataFrame, score_fields: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Validate that score fields exist in dataframe.

    Args:
        df: Input dataframe
        score_fields: List of score field names to validate

    Returns:
        Tuple of (valid_fields, missing_fields)
    """
    available_columns = set(df.columns)
    valid_fields = []
    missing_fields = []

    for field in score_fields:
        if field in available_columns:
            valid_fields.append(field)
        else:
            missing_fields.append(field)

    if missing_fields:
        logger.warning(f"Score fields not found in data: {missing_fields}")
        logger.warning(f"Available columns: {list(df.columns)}")

    if valid_fields:
        logger.info(f"Valid score fields: {valid_fields}")

    return valid_fields, missing_fields


def get_calibrated_score_map(
    df: pd.DataFrame,
    score_field: str,
    calibration_dictionary: Dict[float, float],
    weight_field: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """
    Calculate the calibrated score map based on the input data frame and calibration dictionary.

    Args:
        df (pd.DataFrame): The input data frame containing the score and optional weight fields.
        score_field (str): The name of the column in the data frame that contains the score values.
        calibration_dictionary (Dict[float, float]): A dictionary mapping from calibrated scores
                                                                         to the corresponding percentiles. If None,
                                                                         no calibration is applied. Defaults to None.
        weight_field (Optional[str], optional): The name of the column in the data frame that contains the weight values.
                                                If None, no weights are used. Defaults to None.

    Returns:
        List[Tuple[float, float]]: A list of tuples, each containing a pair of (score, calibrated_score).
                                   The list is sorted by the score values.

    Note:
        - The function adds an 'all' column to the input data frame, set to 1 for all rows, to assist in the
          calculation of the ROC curve.
        - The calibrated scores are calculated by finding the appropriate position in the ROC curve for each
          percentile defined in the calibration dictionary, and interpolating between the neighboring points
          on the ROC curve to find the exact score corresponding to the calibrated percentile.
        - The function returns a list of (score, calibrated_score) pairs, including (0, 0) and (1, 1) to
          define the mapping at the boundaries.
    """
    """
    Parameters:
        df - Dataframe that includes the data and score to use for the calibration (required)
        score_field – name of the column in the df dataframe that contains the score to calibrate (required)
        weight_field – name of the column in the df dataframe that contains the weight to correct for any down sampling.  Default is no weighting of the records.
        calibration_dictionary – a Python dictionary that contains the calibration table with key = score threshold, value = target volume above threshold ratio.  Default is to use the standard calibration at https://w.amazon.com/bin/view/AbusePrevention/Abuse_ML/ModelCalibration/
    """

    df["all"] = 1
    if not weight_field:
        temp, pct, thresholds = roc_curve(df["all"], df[score_field])
    else:
        temp, pct, thresholds = roc_curve(
            df["all"], df[score_field], sample_weight=df[weight_field].values
        )
    pct = np.concatenate([[0.0], pct, [1.0]])
    thresholds = np.concatenate([[1.0], thresholds, [0.0]])
    score_map = []
    score_map.append((0.0, 0.0))
    for s in sorted(calibration_dictionary.keys()):
        for p in range(len(pct) - 1):
            if (
                pct[p] <= calibration_dictionary[s]
                and pct[p + 1] >= calibration_dictionary[s]
            ):
                scr = thresholds[p] + (thresholds[p + 1] - thresholds[p]) * (
                    calibration_dictionary[s] - pct[p]
                ) / (pct[p + 1] - pct[p])
                score_map.append((scr, s))
    score_map.append((1.0, 1.0))
    return score_map


def find_first_data_file(data_dir: str) -> str:
    """Find the most appropriate data file in directory, handling multiple XGBoost output files.

    Args:
        data_dir: Directory to search for data files

    Returns:
        str: Path to the most appropriate data file found

    Raises:
        FileNotFoundError: If no supported data file is found
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    # Get all files in directory
    all_files = os.listdir(data_dir)
    logger.info(f"Found {len(all_files)} files in {data_dir}: {all_files}")

    # Priority order for XGBoost evaluation/inference outputs
    priority_files = [
        "eval_predictions.csv",  # XGBoost model evaluation output (standard)
        "eval_predictions_with_comparison.csv",  # XGBoost model evaluation output (comparison)
        "predictions.csv",  # XGBoost model inference output (CSV)
        "predictions.parquet",  # XGBoost model inference output (Parquet)
        "predictions.json",  # XGBoost model inference output (JSON)
        "processed_data.csv",  # Legacy format
    ]

    # First, check for priority files in order
    found_priority_files = []
    for priority_file in priority_files:
        potential_path = os.path.join(data_dir, priority_file)
        if os.path.exists(potential_path):
            found_priority_files.append((priority_file, potential_path))

    if found_priority_files:
        # If multiple priority files exist, log them and use the first one
        if len(found_priority_files) > 1:
            file_list = [f[0] for f in found_priority_files]
            logger.warning(f"Multiple XGBoost output files found: {file_list}")
            logger.warning(f"Using highest priority file: {found_priority_files[0][0]}")
        else:
            logger.info(f"Found XGBoost output file: {found_priority_files[0][0]}")

        return found_priority_files[0][1]

    # Fallback to any supported file, but warn about multiple files
    supported_files = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            supported_files.append(fname)

    if not supported_files:
        raise FileNotFoundError(
            f"No supported data file (.csv, .parquet, .json) found in {data_dir}"
        )

    if len(supported_files) > 1:
        logger.warning(f"Multiple supported files found: {supported_files}")
        logger.warning(f"Using first file alphabetically: {supported_files[0]}")
    else:
        logger.info(f"Using data file: {supported_files[0]}")

    return os.path.join(data_dir, supported_files[0])


def load_calibration_dictionary(input_paths: Dict[str, str]) -> Dict[float, float]:
    """Load calibration dictionary from input path or use built-in default.

    Args:
        input_paths: Dictionary of input paths with logical names

    Returns:
        Dict[float, float]: Calibration dictionary mapping scores to percentiles
    """
    # First, try to load from explicit input path (following XGBoost pattern)
    calibration_config_path = input_paths.get("calibration_config")
    if calibration_config_path and os.path.exists(calibration_config_path):
        config_file_path = os.path.join(
            calibration_config_path, "standard_calibration_dictionary.json"
        )
        if os.path.exists(config_file_path):
            try:
                logger.info(
                    f"Loading calibration dictionary from external config: {config_file_path}"
                )
                with open(config_file_path, "r") as f:
                    raw_dict = json.load(f)
                standard_calibration_dict = {float(k): v for k, v in raw_dict.items()}
                logger.info(
                    f"Successfully loaded calibration dictionary from external config with {len(standard_calibration_dict)} entries"
                )
                return standard_calibration_dict
            except (json.JSONDecodeError, ValueError, IOError) as e:
                logger.warning(
                    f"Failed to load calibration dictionary from {config_file_path}: {e}"
                )
                logger.info("Falling back to built-in default")
        else:
            logger.info(
                f"Calibration config file not found at {config_file_path}, falling back to built-in default"
            )
    else:
        logger.info(
            "No calibration_config input path provided or path does not exist, falling back to built-in default"
        )

    # Default calibration dictionary from atoz_regional_xgboost/config
    default_calibration_dict = {
        0.001: 0.995014,
        0.002: 0.990047,
        0.003: 0.985101,
        0.004: 0.980174,
        0.005: 0.975267,
        0.006: 0.97038,
        0.007: 0.965514,
        0.008: 0.960665,
        0.009: 0.955838,
        0.01: 0.951028,
        0.011: 0.94624,
        0.012: 0.941469,
        0.013: 0.936717,
        0.014: 0.931987,
        0.015: 0.927273,
        0.016: 0.922581,
        0.017: 0.917906,
        0.018: 0.91325,
        0.019: 0.908613,
        0.02: 0.903996,
        0.021: 0.899397,
        0.022: 0.894818,
        0.023: 0.890255,
        0.024: 0.885714,
        0.025: 0.881189,
        0.026: 0.876683,
        0.027: 0.872197,
        0.028: 0.867727,
        0.029: 0.863277,
        0.03: 0.858845,
        0.031: 0.85443,
        0.032: 0.850035,
        0.033: 0.845657,
        0.034: 0.841298,
        0.035: 0.836957,
        0.036: 0.832633,
        0.037: 0.828326,
        0.038: 0.82404,
        0.039: 0.81977,
        0.04: 0.815516,
        0.041: 0.811283,
        0.042: 0.807064,
        0.043: 0.802866,
        0.044: 0.798684,
        0.045: 0.79452,
        0.046: 0.790372,
        0.047: 0.786242,
        0.048: 0.78213,
        0.049: 0.778034,
        0.05: 0.773957,
        0.051: 0.769895,
        0.052: 0.765852,
        0.053: 0.761825,
        0.054: 0.757815,
        0.055: 0.753823,
        0.056: 0.749846,
        0.057: 0.745887,
        0.058: 0.741945,
        0.059: 0.738019,
        0.06: 0.734109,
        0.061: 0.730217,
        0.062: 0.726341,
        0.063: 0.722482,
        0.064: 0.718638,
        0.065: 0.714812,
        0.066: 0.711002,
        0.067: 0.707208,
        0.068: 0.70343,
        0.069: 0.699668,
        0.07: 0.695921,
        0.071: 0.692193,
        0.072: 0.688479,
        0.073: 0.684782,
        0.074: 0.681099,
        0.075: 0.677435,
        0.076: 0.673785,
        0.077: 0.67015,
        0.078: 0.666532,
        0.079: 0.662929,
        0.08: 0.659342,
        0.081: 0.655771,
        0.082: 0.652214,
        0.083: 0.648674,
        0.084: 0.645149,
        0.085: 0.641639,
        0.086: 0.638143,
        0.087: 0.634664,
        0.088: 0.631201,
        0.089: 0.627752,
        0.09: 0.624317,
        0.091: 0.620899,
        0.092: 0.617495,
        0.093: 0.614105,
        0.094: 0.610731,
        0.095: 0.607372,
        0.096: 0.604028,
        0.097: 0.6007,
        0.098: 0.597384,
        0.099: 0.594085,
        0.1: 0.590799,
        0.101: 0.587527,
        0.102: 0.584272,
        0.103: 0.581029,
        0.104: 0.577801,
        0.105: 0.574589,
        0.106: 0.57139,
        0.107: 0.568206,
        0.108: 0.565035,
        0.109: 0.561879,
        0.11: 0.558736,
        0.111: 0.555609,
        0.112: 0.552495,
        0.113: 0.549395,
        0.114: 0.546308,
        0.115: 0.543237,
        0.116: 0.540179,
        0.117: 0.537135,
        0.118: 0.534104,
        0.119: 0.531087,
        0.12: 0.528084,
        0.121: 0.525094,
        0.122: 0.522118,
        0.123: 0.519154,
        0.124: 0.516205,
        0.125: 0.51327,
        0.126: 0.510348,
        0.127: 0.507439,
        0.128: 0.504544,
        0.129: 0.501661,
        0.13: 0.498791,
        0.131: 0.495936,
        0.132: 0.493093,
        0.133: 0.490263,
        0.134: 0.487446,
        0.135: 0.484643,
        0.136: 0.481852,
        0.137: 0.479074,
        0.138: 0.476307,
        0.139: 0.473556,
        0.14: 0.470815,
        0.141: 0.468089,
        0.142: 0.465375,
        0.143: 0.462673,
        0.144: 0.459984,
        0.145: 0.457308,
        0.146: 0.454644,
        0.147: 0.451992,
        0.148: 0.449353,
        0.149: 0.446725,
        0.15: 0.444111,
        0.151: 0.441509,
        0.152: 0.438919,
        0.153: 0.436341,
        0.154: 0.433774,
        0.155: 0.431222,
        0.156: 0.42868,
        0.157: 0.42615,
        0.158: 0.423633,
        0.159: 0.421127,
        0.16: 0.418633,
        0.161: 0.416151,
        0.162: 0.41368,
        0.163: 0.411223,
        0.164: 0.408776,
        0.165: 0.406341,
        0.166: 0.403918,
        0.167: 0.401506,
        0.168: 0.399105,
        0.169: 0.396717,
        0.17: 0.394339,
        0.171: 0.391974,
        0.172: 0.389619,
        0.173: 0.387276,
        0.174: 0.384945,
        0.175: 0.382623,
        0.176: 0.380314,
        0.177: 0.378015,
        0.178: 0.375729,
        0.179: 0.373454,
        0.18: 0.371189,
        0.181: 0.368934,
        0.182: 0.366691,
        0.183: 0.36446,
        0.184: 0.362239,
        0.185: 0.360029,
        0.186: 0.357829,
        0.187: 0.355641,
        0.188: 0.353462,
        0.189: 0.351296,
        0.19: 0.349139,
        0.191: 0.346993,
        0.192: 0.344858,
        0.193: 0.342733,
        0.194: 0.340619,
        0.195: 0.338515,
        0.196: 0.336422,
        0.197: 0.334339,
        0.198: 0.332266,
        0.199: 0.330204,
        0.2: 0.328152,
        0.201: 0.32611,
        0.202: 0.324078,
        0.203: 0.322057,
        0.204: 0.320044,
        0.205: 0.318044,
        0.206: 0.316053,
        0.207: 0.314071,
        0.208: 0.3121,
        0.209: 0.310139,
        0.21: 0.308187,
        0.211: 0.306245,
        0.212: 0.304314,
        0.213: 0.302392,
        0.214: 0.300479,
        0.215: 0.298577,
        0.216: 0.296684,
        0.217: 0.294801,
        0.218: 0.292927,
        0.219: 0.291062,
        0.22: 0.289207,
        0.221: 0.287363,
        0.222: 0.285527,
        0.223: 0.283701,
        0.224: 0.281883,
        0.225: 0.280076,
        0.226: 0.278277,
        0.227: 0.276489,
        0.228: 0.274709,
        0.229: 0.272939,
        0.23: 0.271177,
        0.231: 0.269424,
        0.232: 0.267682,
        0.233: 0.265947,
        0.234: 0.264222,
        0.235: 0.262505,
        0.236: 0.260799,
        0.237: 0.2591,
        0.238: 0.257411,
        0.239: 0.25573,
        0.24: 0.254058,
        0.241: 0.252395,
        0.242: 0.250741,
        0.243: 0.249094,
        0.244: 0.247458,
        0.245: 0.24583,
        0.246: 0.244209,
        0.247: 0.242598,
        0.248: 0.240997,
        0.249: 0.239402,
        0.25: 0.237817,
        0.251: 0.236239,
        0.252: 0.234671,
        0.253: 0.233109,
        0.254: 0.231557,
        0.255: 0.230014,
        0.256: 0.228479,
        0.257: 0.22695,
        0.258: 0.225432,
        0.259: 0.22392,
        0.26: 0.222417,
        0.261: 0.220922,
        0.262: 0.219437,
        0.263: 0.217957,
        0.264: 0.216486,
        0.265: 0.215025,
        0.266: 0.213569,
        0.267: 0.212122,
        0.268: 0.210683,
        0.269: 0.209253,
        0.27: 0.207828,
        0.271: 0.206414,
        0.272: 0.205006,
        0.273: 0.203605,
        0.274: 0.202213,
        0.275: 0.200827,
        0.276: 0.199451,
        0.277: 0.19808,
        0.278: 0.196719,
        0.279: 0.195364,
        0.28: 0.194018,
        0.281: 0.192678,
        0.282: 0.191346,
        0.283: 0.190021,
        0.284: 0.188703,
        0.285: 0.187394,
        0.286: 0.18609,
        0.287: 0.184795,
        0.288: 0.183507,
        0.289: 0.182226,
        0.29: 0.180952,
        0.291: 0.179684,
        0.292: 0.178426,
        0.293: 0.177173,
        0.294: 0.175928,
        0.295: 0.174688,
        0.296: 0.173458,
        0.297: 0.172233,
        0.298: 0.171016,
        0.299: 0.169804,
        0.3: 0.168601,
        0.301: 0.167405,
        0.302: 0.166214,
        0.303: 0.16503,
        0.304: 0.163855,
        0.305: 0.162685,
        0.306: 0.161522,
        0.307: 0.160365,
        0.308: 0.159216,
        0.309: 0.158073,
        0.31: 0.156936,
        0.311: 0.155806,
        0.312: 0.154683,
        0.313: 0.153565,
        0.314: 0.152455,
        0.315: 0.151352,
        0.316: 0.150254,
        0.317: 0.149163,
        0.318: 0.148078,
        0.319: 0.147,
        0.32: 0.145927,
        0.321: 0.144862,
        0.322: 0.143801,
        0.323: 0.142749,
        0.324: 0.141701,
        0.325: 0.14066,
        0.326: 0.139626,
        0.327: 0.138597,
        0.328: 0.137574,
        0.329: 0.136557,
        0.33: 0.135547,
        0.331: 0.134542,
        0.332: 0.133544,
        0.333: 0.132551,
        0.334: 0.131565,
        0.335: 0.130584,
        0.336: 0.129609,
        0.337: 0.12864,
        0.338: 0.127676,
        0.339: 0.126719,
        0.34: 0.125767,
        0.341: 0.124821,
        0.342: 0.12388,
        0.343: 0.122947,
        0.344: 0.122017,
        0.345: 0.121095,
        0.346: 0.120176,
        0.347: 0.119265,
        0.348: 0.118358,
        0.349: 0.117457,
        0.35: 0.116562,
        0.351: 0.115673,
        0.352: 0.114788,
        0.353: 0.113909,
        0.354: 0.113036,
        0.355: 0.112166,
        0.356: 0.111305,
        0.357: 0.110446,
        0.358: 0.109595,
        0.359: 0.108748,
        0.36: 0.107906,
        0.361: 0.10707,
        0.362: 0.106239,
        0.363: 0.105412,
        0.364: 0.104591,
        0.365: 0.103776,
        0.366: 0.102966,
        0.367: 0.10216,
        0.368: 0.101359,
        0.369: 0.100564,
        0.37: 0.099774,
        0.371: 0.098989,
        0.372: 0.098207,
        0.373: 0.097432,
        0.374: 0.096662,
        0.375: 0.095897,
        0.376: 0.095136,
        0.377: 0.094379,
        0.378: 0.093628,
        0.379: 0.092883,
        0.38: 0.092141,
        0.381: 0.091404,
        0.382: 0.090672,
        0.383: 0.089945,
        0.384: 0.089223,
        0.385: 0.088505,
        0.386: 0.087792,
        0.387: 0.087083,
        0.388: 0.086379,
        0.389: 0.085679,
        0.39: 0.084984,
        0.391: 0.084294,
        0.392: 0.083608,
        0.393: 0.082927,
        0.394: 0.082249,
        0.395: 0.081578,
        0.396: 0.08091,
        0.397: 0.080246,
        0.398: 0.079586,
        0.399: 0.078932,
        0.4: 0.078281,
        0.401: 0.077636,
        0.402: 0.076993,
        0.403: 0.076356,
        0.404: 0.075723,
        0.405: 0.075093,
        0.406: 0.074469,
        0.407: 0.073848,
        0.408: 0.07323,
        0.409: 0.072619,
        0.41: 0.072009,
        0.411: 0.071406,
        0.412: 0.070806,
        0.413: 0.07021,
        0.414: 0.069618,
        0.415: 0.06903,
        0.416: 0.068445,
        0.417: 0.067866,
        0.418: 0.06729,
        0.419: 0.066718,
        0.42: 0.06615,
        0.421: 0.065584,
        0.422: 0.065025,
        0.423: 0.064468,
        0.424: 0.063915,
        0.425: 0.063366,
        0.426: 0.062821,
        0.427: 0.062279,
        0.428: 0.061742,
        0.429: 0.061209,
        0.43: 0.060679,
        0.431: 0.060152,
        0.432: 0.059628,
        0.433: 0.059109,
        0.434: 0.058594,
        0.435: 0.058082,
        0.436: 0.057574,
        0.437: 0.057069,
        0.438: 0.056568,
        0.439: 0.056072,
        0.44: 0.055577,
        0.441: 0.055087,
        0.442: 0.054599,
        0.443: 0.054117,
        0.444: 0.053637,
        0.445: 0.05316,
        0.446: 0.052687,
        0.447: 0.052217,
        0.448: 0.05175,
        0.449: 0.051287,
        0.45: 0.050828,
        0.451: 0.050371,
        0.452: 0.049918,
        0.453: 0.049468,
        0.454: 0.049021,
        0.455: 0.048577,
        0.456: 0.048138,
        0.457: 0.047701,
        0.458: 0.047268,
        0.459: 0.046837,
        0.46: 0.04641,
        0.461: 0.045986,
        0.462: 0.045565,
        0.463: 0.045147,
        0.464: 0.044732,
        0.465: 0.04432,
        0.466: 0.043912,
        0.467: 0.043505,
        0.468: 0.043102,
        0.469: 0.042704,
        0.47: 0.042307,
        0.471: 0.041914,
        0.472: 0.041523,
        0.473: 0.041135,
        0.474: 0.04075,
        0.475: 0.040368,
        0.476: 0.039989,
        0.477: 0.039612,
        0.478: 0.03924,
        0.479: 0.038869,
        0.48: 0.038502,
        0.481: 0.038137,
        0.482: 0.037774,
        0.483: 0.037416,
        0.484: 0.037059,
        0.485: 0.036705,
        0.486: 0.036355,
        0.487: 0.036005,
        0.488: 0.035661,
        0.489: 0.035317,
        0.49: 0.034977,
        0.491: 0.03464,
        0.492: 0.034304,
        0.493: 0.033972,
        0.494: 0.033643,
        0.495: 0.033316,
        0.496: 0.032991,
        0.497: 0.032669,
        0.498: 0.03235,
        0.499: 0.032033,
        0.5: 0.031717,
        0.501: 0.031405,
        0.502: 0.031097,
        0.503: 0.030789,
        0.504: 0.030484,
        0.505: 0.030182,
        0.506: 0.029883,
        0.507: 0.029585,
        0.508: 0.02929,
        0.509: 0.028998,
        0.51: 0.028709,
        0.511: 0.028421,
        0.512: 0.028135,
        0.513: 0.027852,
        0.514: 0.027572,
        0.515: 0.027292,
        0.516: 0.027017,
        0.517: 0.026743,
        0.518: 0.02647,
        0.519: 0.026202,
        0.52: 0.025934,
        0.521: 0.025668,
        0.522: 0.025407,
        0.523: 0.025145,
        0.524: 0.024887,
        0.525: 0.024631,
        0.526: 0.024377,
        0.527: 0.024125,
        0.528: 0.023875,
        0.529: 0.023627,
        0.53: 0.023381,
        0.531: 0.023136,
        0.532: 0.022896,
        0.533: 0.022656,
        0.534: 0.022418,
        0.535: 0.022182,
        0.536: 0.021949,
        0.537: 0.021717,
        0.538: 0.021488,
        0.539: 0.021261,
        0.54: 0.021034,
        0.541: 0.020812,
        0.542: 0.02059,
        0.543: 0.02037,
        0.544: 0.020152,
        0.545: 0.019936,
        0.546: 0.019722,
        0.547: 0.01951,
        0.548: 0.019299,
        0.549: 0.019091,
        0.55: 0.018883,
        0.551: 0.018678,
        0.552: 0.018476,
        0.553: 0.018275,
        0.554: 0.018075,
        0.555: 0.017877,
        0.556: 0.017681,
        0.557: 0.017487,
        0.558: 0.017294,
        0.559: 0.017103,
        0.56: 0.016915,
        0.561: 0.016727,
        0.562: 0.016541,
        0.563: 0.016357,
        0.564: 0.016175,
        0.565: 0.015995,
        0.566: 0.015816,
        0.567: 0.015638,
        0.568: 0.015462,
        0.569: 0.015288,
        0.57: 0.015116,
        0.571: 0.014945,
        0.572: 0.014775,
        0.573: 0.014608,
        0.574: 0.01444,
        0.575: 0.014277,
        0.576: 0.014112,
        0.577: 0.013951,
        0.578: 0.013791,
        0.579: 0.013633,
        0.58: 0.013476,
        0.581: 0.01332,
        0.582: 0.013165,
        0.583: 0.013013,
        0.584: 0.012862,
        0.585: 0.012712,
        0.586: 0.012563,
        0.587: 0.012416,
        0.588: 0.01227,
        0.589: 0.012126,
        0.59: 0.011984,
        0.591: 0.011842,
        0.592: 0.011701,
        0.593: 0.011562,
        0.594: 0.011425,
        0.595: 0.01129,
        0.596: 0.011155,
        0.597: 0.011021,
        0.598: 0.01089,
        0.599: 0.010759,
        0.6: 0.010629,
        0.601: 0.010501,
        0.602: 0.010374,
        0.603: 0.010248,
        0.604: 0.010124,
        0.605: 0.010001,
        0.606: 0.009878,
        0.607: 0.009758,
        0.608: 0.009639,
        0.609: 0.009519,
        0.61: 0.009403,
        0.611: 0.009287,
        0.612: 0.009172,
        0.613: 0.009059,
        0.614: 0.008945,
        0.615: 0.008834,
        0.616: 0.008725,
        0.617: 0.008616,
        0.618: 0.008508,
        0.619: 0.008401,
        0.62: 0.008295,
        0.621: 0.008191,
        0.622: 0.008087,
        0.623: 0.007984,
        0.624: 0.007883,
        0.625: 0.007783,
        0.626: 0.007684,
        0.627: 0.007586,
        0.628: 0.007488,
        0.629: 0.007391,
        0.63: 0.007297,
        0.631: 0.007202,
        0.632: 0.00711,
        0.633: 0.007018,
        0.634: 0.006927,
        0.635: 0.006836,
        0.636: 0.006746,
        0.637: 0.006659,
        0.638: 0.006572,
        0.639: 0.006486,
        0.64: 0.006399,
        0.641: 0.006315,
        0.642: 0.006232,
        0.643: 0.00615,
        0.644: 0.006068,
        0.645: 0.005987,
        0.646: 0.005907,
        0.647: 0.005828,
        0.648: 0.00575,
        0.649: 0.005673,
        0.65: 0.005596,
        0.651: 0.005521,
        0.652: 0.005445,
        0.653: 0.005373,
        0.654: 0.0053,
        0.655: 0.005227,
        0.656: 0.005156,
        0.657: 0.005085,
        0.658: 0.005016,
        0.659: 0.004947,
        0.66: 0.004879,
        0.661: 0.004811,
        0.662: 0.004745,
        0.663: 0.004679,
        0.664: 0.004614,
        0.665: 0.00455,
        0.666: 0.004485,
        0.667: 0.004423,
        0.668: 0.00436,
        0.669: 0.0043,
        0.67: 0.004239,
        0.671: 0.004178,
        0.672: 0.004119,
        0.673: 0.004062,
        0.674: 0.004004,
        0.675: 0.003947,
        0.676: 0.003891,
        0.677: 0.003835,
        0.678: 0.00378,
        0.679: 0.003725,
        0.68: 0.003672,
        0.681: 0.003619,
        0.682: 0.003565,
        0.683: 0.003515,
        0.684: 0.003463,
        0.685: 0.003413,
        0.686: 0.003362,
        0.687: 0.003313,
        0.688: 0.003264,
        0.689: 0.003217,
        0.69: 0.003169,
        0.691: 0.003123,
        0.692: 0.003077,
        0.693: 0.003031,
        0.694: 0.002986,
        0.695: 0.002941,
        0.696: 0.002897,
        0.697: 0.002853,
        0.698: 0.00281,
        0.699: 0.002768,
        0.7: 0.002727,
        0.701: 0.002686,
        0.702: 0.002644,
        0.703: 0.002604,
        0.704: 0.002565,
        0.705: 0.002526,
        0.706: 0.002487,
        0.707: 0.00245,
        0.708: 0.002412,
        0.709: 0.002375,
        0.71: 0.002339,
        0.711: 0.002303,
        0.712: 0.002267,
        0.713: 0.002232,
        0.714: 0.002196,
        0.715: 0.002163,
        0.716: 0.002129,
        0.717: 0.002095,
        0.718: 0.002062,
        0.719: 0.002031,
        0.72: 0.001999,
        0.721: 0.001967,
        0.722: 0.001937,
        0.723: 0.001906,
        0.724: 0.001875,
        0.725: 0.001846,
        0.726: 0.001816,
        0.727: 0.001787,
        0.728: 0.001759,
        0.729: 0.001731,
        0.73: 0.001702,
        0.731: 0.001676,
        0.732: 0.001648,
        0.733: 0.001622,
        0.734: 0.001596,
        0.735: 0.00157,
        0.736: 0.001545,
        0.737: 0.001519,
        0.738: 0.001495,
        0.739: 0.00147,
        0.74: 0.001447,
        0.741: 0.001423,
        0.742: 0.0014,
        0.743: 0.001377,
        0.744: 0.001354,
        0.745: 0.001331,
        0.746: 0.00131,
        0.747: 0.001287,
        0.748: 0.001267,
        0.749: 0.001246,
        0.75: 0.001225,
        0.751: 0.001205,
        0.752: 0.001184,
        0.753: 0.001165,
        0.754: 0.001145,
        0.755: 0.001127,
        0.756: 0.001108,
        0.757: 0.001089,
        0.758: 0.001071,
        0.759: 0.001053,
        0.76: 0.001035,
        0.761: 0.001018,
        0.762: 0.001001,
        0.763: 0.000983,
        0.764: 0.000967,
        0.765: 0.00095,
        0.766: 0.000935,
        0.767: 0.000919,
        0.768: 0.000902,
        0.769: 0.000888,
        0.77: 0.000873,
        0.771: 0.000857,
        0.772: 0.000843,
        0.773: 0.000829,
        0.774: 0.000815,
        0.775: 0.000801,
        0.776: 0.000786,
        0.777: 0.000774,
        0.778: 0.000759,
        0.779: 0.000746,
        0.78: 0.000735,
        0.781: 0.000722,
        0.782: 0.00071,
        0.783: 0.000696,
        0.784: 0.000685,
        0.785: 0.000674,
        0.786: 0.000661,
        0.787: 0.000651,
        0.788: 0.000639,
        0.789: 0.000629,
        0.79: 0.000618,
        0.791: 0.000607,
        0.792: 0.000597,
        0.793: 0.000585,
        0.794: 0.000576,
        0.795: 0.000566,
        0.796: 0.000557,
        0.797: 0.000547,
        0.798: 0.000538,
        0.799: 0.000529,
        0.8: 0.000518,
        0.801: 0.000511,
        0.802: 0.000502,
        0.803: 0.000493,
        0.804: 0.000484,
        0.805: 0.000475,
        0.806: 0.000468,
        0.807: 0.000459,
        0.808: 0.000452,
        0.809: 0.000445,
        0.81: 0.000437,
        0.811: 0.00043,
        0.812: 0.000421,
        0.813: 0.000415,
        0.814: 0.000408,
        0.815: 0.000401,
        0.816: 0.000395,
        0.817: 0.000388,
        0.818: 0.000381,
        0.819: 0.000375,
        0.82: 0.000368,
        0.821: 0.000361,
        0.822: 0.000356,
        0.823: 0.00035,
        0.824: 0.000345,
        0.825: 0.000339,
        0.826: 0.000332,
        0.827: 0.000328,
        0.828: 0.000321,
        0.829: 0.000317,
        0.83: 0.000312,
        0.831: 0.000307,
        0.832: 0.000301,
        0.833: 0.000297,
        0.834: 0.000292,
        0.835: 0.000287,
        0.836: 0.000281,
        0.837: 0.000278,
        0.838: 0.000272,
        0.839: 0.000269,
        0.84: 0.000265,
        0.841: 0.00026,
        0.842: 0.000256,
        0.843: 0.000252,
        0.844: 0.000248,
        0.845: 0.000244,
        0.846: 0.000239,
        0.847: 0.000237,
        0.848: 0.000233,
        0.849: 0.000228,
        0.85: 0.000226,
        0.851: 0.000222,
        0.852: 0.000218,
        0.853: 0.000215,
        0.854: 0.000212,
        0.855: 0.000208,
        0.856: 0.000206,
        0.857: 0.000203,
        0.858: 0.000199,
        0.859: 0.000197,
        0.86: 0.000194,
        0.861: 0.00019,
        0.862: 0.000188,
        0.863: 0.000185,
        0.864: 0.000181,
        0.865: 0.000179,
        0.866: 0.000177,
        0.867: 0.000174,
        0.868: 0.000171,
        0.869: 0.000169,
        0.87: 0.000167,
        0.871: 0.000165,
        0.872: 0.000161,
        0.873: 0.000159,
        0.874: 0.000158,
        0.875: 0.000155,
        0.876: 0.000153,
        0.877: 0.00015,
        0.878: 0.000149,
        0.879: 0.000147,
        0.88: 0.000145,
        0.881: 0.000143,
        0.882: 0.00014,
        0.883: 0.000139,
        0.884: 0.000137,
        0.885: 0.000135,
        0.886: 0.000133,
        0.887: 0.00013,
        0.888: 0.000129,
        0.889: 0.000128,
        0.89: 0.000126,
        0.891: 0.000124,
        0.892: 0.000122,
        0.893: 0.000121,
        0.894: 0.000118,
        0.895: 0.000118,
        0.896: 0.000116,
        0.897: 0.000113,
        0.898: 0.000113,
        0.899: 0.000111,
        0.9: 0.000109,
        0.901: 0.000108,
        0.902: 0.000107,
        0.903: 0.000104,
        0.904: 0.000104,
        0.905: 0.000103,
        0.906: 0.000101,
        0.907: 9.999999e-05,
        0.908: 9.8e-05,
        0.909: 9.7e-05,
        0.91: 9.6e-05,
        0.911: 9.4e-05,
        0.912: 9.3e-05,
        0.913: 9.2e-05,
        0.914: 9.099999e-05,
        0.915: 8.9e-05,
        0.916: 8.8e-05,
        0.917: 8.7e-05,
        0.918: 8.599999e-05,
        0.919: 8.4e-05,
        0.92: 8.3e-05,
        0.921: 8.2e-05,
        0.922: 8.099999e-05,
        0.923: 7.999999e-05,
        0.924: 7.8e-05,
        0.925: 7.7e-05,
        0.926: 7.599999e-05,
        0.927: 7.5e-05,
        0.928: 7.4e-05,
        0.929: 7.3e-05,
        0.93: 7.2e-05,
        0.931: 7e-05,
        0.932: 6.9e-05,
        0.933: 6.8e-05,
        0.934: 6.7e-05,
        0.935: 6.599999e-05,
        0.936: 6.5e-05,
        0.937: 6.4e-05,
        0.938: 6.3e-05,
        0.939: 6.2e-05,
        0.94: 6.1e-05,
        0.941: 5.999999e-05,
        0.942: 5.9e-05,
        0.943: 5.8e-05,
        0.944: 5.6e-05,
        0.945: 5.499999e-05,
        0.946: 5.4e-05,
        0.947: 5.3e-05,
        0.948: 5.2e-05,
        0.949: 5.1e-05,
        0.95: 4.999999e-05,
        0.951: 4.9e-05,
        0.952: 4.8e-05,
        0.953: 4.7e-05,
        0.954: 4.6e-05,
        0.955: 4.499999e-05,
        0.956: 4.4e-05,
        0.957: 4.299999e-05,
        0.958: 4.2e-05,
        0.959: 4.1e-05,
        0.96: 3.999999e-05,
        0.961: 3.9e-05,
        0.962: 3.799999e-05,
        0.963: 3.7e-05,
        0.964: 3.6e-05,
        0.965: 3.5e-05,
        0.966: 3.4e-05,
        0.967: 3.299999e-05,
        0.968: 3.2e-05,
        0.969: 3.1e-05,
        0.97: 2.999999e-05,
        0.971: 2.9e-05,
        0.972: 2.8e-05,
        0.973: 2.7e-05,
        0.974: 2.6e-05,
        0.975: 2.499999e-05,
        0.976: 2.4e-05,
        0.977: 2.3e-05,
        0.978: 2.2e-05,
        0.979: 2.1e-05,
        0.98: 1.999999e-05,
        0.981: 1.899999e-05,
        0.982: 1.8e-05,
        0.983: 1.7e-05,
        0.984: 1.6e-05,
        0.985: 1.499999e-05,
        0.986: 1.4e-05,
        0.987: 1.3e-05,
        0.988: 1.2e-05,
        0.989: 1.1e-05,
        0.99: 9.999999e-06,
        0.991: 9e-06,
        0.992: 8e-06,
        0.993: 7e-06,
        0.994: 6e-06,
        0.995: 4.999999e-06,
        0.996: 4e-06,
        0.997: 3e-06,
        0.998: 2e-06,
        0.999: 1e-06,
    }

    return default_calibration_dict


def main(
    input_paths: dict,
    output_paths: dict,
    environ_vars: dict,
    job_args: argparse.Namespace = None,
) -> dict:
    """Main entry point for the percentile model calibration script.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Dictionary with metrics and results
    """
    try:
        logger.info("Starting percentile model calibration")

        # Get job_type from command line arguments if available
        job_type = "calibration"  # default
        if job_args and hasattr(job_args, "job_type"):
            job_type = job_args.job_type
            logger.info(f"Using job_type from command line: {job_type}")

        # Parse environment variables
        n_bins = int(environ_vars.get("N_BINS", "1000"))
        accuracy = float(environ_vars.get("ACCURACY", "1e-5"))

        # Parse score fields (multi-task support with backward compatibility)
        score_fields = parse_score_fields(environ_vars)
        is_multitask = len(score_fields) > 1

        logger.info(
            f"Configuration: n_bins={n_bins}, accuracy={accuracy}, job_type={job_type}"
        )
        logger.info(
            f"Mode: {'Multi-task' if is_multitask else 'Single-task'} with {len(score_fields)} score field(s)"
        )

        # Create output directories
        calibration_output_dir = output_paths.get("calibration_output")
        metrics_output_dir = output_paths.get("metrics_output")
        calibrated_data_dir = output_paths.get("calibrated_data")

        if not calibration_output_dir:
            raise ValueError("calibration_output path not provided in output_paths")
        if not metrics_output_dir:
            raise ValueError("metrics_output path not provided in output_paths")
        if not calibrated_data_dir:
            raise ValueError("calibrated_data path not provided in output_paths")

        os.makedirs(calibration_output_dir, exist_ok=True)
        os.makedirs(metrics_output_dir, exist_ok=True)
        os.makedirs(calibrated_data_dir, exist_ok=True)

        # Load calibration dictionary
        standard_calibration_dict = load_calibration_dictionary(input_paths)
        logger.info(
            f"Loaded calibration dictionary with {len(standard_calibration_dict)} entries"
        )

        # Load input data
        evaluation_data_dir = input_paths.get("evaluation_data")
        if not evaluation_data_dir:
            raise ValueError("evaluation_data path not provided in input_paths")

        # Use the same flexible approach as model_calibration.py - find any supported data file
        calibration_scores_path = find_first_data_file(evaluation_data_dir)

        logger.info(f"Loading calibration data from {calibration_scores_path}")

        # Load the data with format preservation
        try:
            df_calibration_scores, input_format = load_dataframe_with_format(
                calibration_scores_path
            )
            logger.info(f"Detected input format: {input_format}")
        except Exception as e:
            raise ValueError(
                f"Failed to load data from {calibration_scores_path}: {str(e)}"
            )

        logger.info(
            f"Loaded dataframe with shape {df_calibration_scores.shape} and columns: {list(df_calibration_scores.columns)}"
        )

        # Validate score fields exist in dataframe
        valid_fields, missing_fields = validate_score_fields(
            df_calibration_scores, score_fields
        )

        if missing_fields:
            raise ValueError(
                f"Score fields not found in data: {missing_fields}. "
                f"Available columns: {list(df_calibration_scores.columns)}"
            )

        if not valid_fields:
            raise ValueError("No valid score fields found for calibration")

        logger.info(
            f"Will calibrate {len(valid_fields)} score field(s): {valid_fields}"
        )

        # Helper function to apply percentile mapping
        def apply_percentile_mapping(score, score_map):
            """Apply percentile mapping to a single score."""
            if score <= score_map[0][0]:
                return score_map[0][1]
            if score >= score_map[-1][0]:
                return score_map[-1][1]

            # Find the appropriate range and interpolate
            for i in range(len(score_map) - 1):
                if score_map[i][0] <= score <= score_map[i + 1][0]:
                    # Linear interpolation
                    x1, y1 = score_map[i]
                    x2, y2 = score_map[i + 1]
                    if x2 == x1:
                        return y1
                    return y1 + (y2 - y1) * (score - x1) / (x2 - x1)

            return score_map[-1][1]  # fallback

        # Calibrate each task
        task_calibrations = {}
        task_metrics = {}
        df_calibrated = df_calibration_scores.copy()

        for score_field in valid_fields:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Calibrating score field: {score_field}")
            logger.info(f"{'=' * 60}")

            # Extract and validate scores for this task
            raw_scores = df_calibration_scores[score_field].values

            # Handle missing values
            missing_count = pd.isna(raw_scores).sum()
            if missing_count > 0:
                logger.warning(
                    f"Found {missing_count} missing values in '{score_field}', they will be excluded from calibration"
                )

            # Get valid scores (non-NaN)
            valid_mask = ~pd.isna(raw_scores)
            calibration_scores = raw_scores[valid_mask].reshape(-1)

            if len(calibration_scores) == 0:
                logger.error(f"No valid scores for field '{score_field}', skipping")
                continue

            # Basic data quality checks
            min_score = float(np.min(calibration_scores))
            max_score = float(np.max(calibration_scores))
            mean_score = float(np.mean(calibration_scores))
            std_score = float(np.std(calibration_scores))
            unique_scores = len(np.unique(calibration_scores))

            logger.info(
                f"Score statistics for '{score_field}': "
                f"min={min_score:.6f}, max={max_score:.6f}, "
                f"mean={mean_score:.6f}, std={std_score:.6f}"
            )
            logger.info(
                f"Loaded {len(calibration_scores)} calibration scores "
                f"with {unique_scores} unique values"
            )

            # Validate score range (should be probabilities between 0 and 1)
            if min_score < 0 or max_score > 1:
                logger.warning(
                    f"Scores outside [0,1] range for '{score_field}': "
                    f"min={min_score:.6f}, max={max_score:.6f}"
                )
                if min_score < -0.1 or max_score > 1.1:
                    logger.error(
                        f"Scores significantly outside [0,1] range for '{score_field}', skipping"
                    )
                    continue

                # Clip to valid range
                calibration_scores = np.clip(calibration_scores, 0.0, 1.0)
                logger.info(f"Clipped scores for '{score_field}' to [0,1] range")

            # Check for constant scores
            if std_score < 1e-10:
                logger.error(
                    f"All scores are constant (std={std_score:.2e}) for '{score_field}', skipping"
                )
                continue

            # Warn about low diversity
            if unique_scores < 10:
                logger.warning(
                    f"Only {unique_scores} unique score values for '{score_field}', "
                    f"calibration may be less effective"
                )

            # Create dataframe for calibration
            raw_score_df = pd.DataFrame({"raw_scores": calibration_scores})

            # Perform calibration
            logger.info(
                f"Performing percentile score mapping calibration for '{score_field}'"
            )
            calibrated_score_map = get_calibrated_score_map(
                df=raw_score_df,
                score_field="raw_scores",
                calibration_dictionary=standard_calibration_dict,
                weight_field=None,
            )

            logger.info(
                f"Generated calibrated score map for '{score_field}' "
                f"with {len(calibrated_score_map)} entries"
            )

            # Store calibration map
            task_calibrations[score_field] = calibrated_score_map

            # Save per-task calibration output
            task_percentile_filename = f"percentile_score_{score_field}.pkl"
            task_percentile_path = os.path.join(
                calibration_output_dir, task_percentile_filename
            )
            with open(task_percentile_path, "wb") as f:
                pkl.dump(calibrated_score_map, f)

            logger.info(
                f"Saved percentile score mapping for '{score_field}' to {task_percentile_path}"
            )

            # Store task metrics
            task_metrics[score_field] = {
                "num_calibration_points": len(calibrated_score_map),
                "num_input_scores": len(calibration_scores),
                "score_statistics": {
                    "min_score": min_score,
                    "max_score": max_score,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "unique_scores": unique_scores,
                },
                "calibration_range": {
                    "min_percentile": min(calibrated_score_map, key=lambda x: x[1])[1],
                    "max_percentile": max(calibrated_score_map, key=lambda x: x[1])[1],
                    "min_score_threshold": min(
                        calibrated_score_map, key=lambda x: x[0]
                    )[0],
                    "max_score_threshold": max(
                        calibrated_score_map, key=lambda x: x[0]
                    )[0],
                },
            }

            # Apply calibration to all scores (including NaN)
            calibrated_field_values = []
            for score in raw_scores:
                if pd.isna(score):
                    calibrated_field_values.append(np.nan)
                else:
                    calibrated_field_values.append(
                        apply_percentile_mapping(score, calibrated_score_map)
                    )

            # Add calibrated column to output dataframe
            df_calibrated[f"{score_field}_percentile"] = calibrated_field_values

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Calibration complete for all tasks")
        logger.info(f"{'=' * 60}\n")

        # Save aggregated metrics
        metrics = {
            "calibration_method": "percentile_score_mapping",
            "mode": "multi-task" if is_multitask else "single-task",
            "num_tasks": len(task_calibrations),
            "score_fields": list(task_calibrations.keys()),
            "per_task_metrics": task_metrics,
            "config": {
                "n_bins": n_bins,
                "accuracy": accuracy,
                "calibration_dict_size": len(standard_calibration_dict),
                "job_type": job_type,
            },
        }

        # Add aggregate statistics if multi-task
        if is_multitask and task_metrics:
            aggregate_stats = {
                "total_calibration_points": sum(
                    m["num_calibration_points"] for m in task_metrics.values()
                ),
                "total_input_scores": sum(
                    m["num_input_scores"] for m in task_metrics.values()
                ),
                "avg_unique_scores": np.mean(
                    [
                        m["score_statistics"]["unique_scores"]
                        for m in task_metrics.values()
                    ]
                ),
            }
            metrics["aggregate_statistics"] = aggregate_stats

        metrics_path = os.path.join(metrics_output_dir, "calibration_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved calibration metrics to {metrics_path}")

        # Save calibrated data with format preservation
        from pathlib import Path

        calibrated_data_base = os.path.join(calibrated_data_dir, "calibrated_data")
        calibrated_data_path = save_dataframe_with_format(
            df_calibrated, calibrated_data_base, input_format
        )

        logger.info(
            f"Saved calibrated data (format={input_format}): {calibrated_data_path}"
        )

        # Build output files dict
        output_files = {
            "metrics": metrics_path,
            "calibrated_data": calibrated_data_path,
        }

        # Add per-task percentile score files
        for score_field in task_calibrations.keys():
            output_files[f"percentile_score_{score_field}"] = os.path.join(
                calibration_output_dir, f"percentile_score_{score_field}.pkl"
            )

        # Backward compatibility: save first task as default percentile_score.pkl
        if task_calibrations:
            first_field = list(task_calibrations.keys())[0]
            default_percentile_path = os.path.join(
                calibration_output_dir, "percentile_score.pkl"
            )
            with open(default_percentile_path, "wb") as f:
                pkl.dump(task_calibrations[first_field], f)
            output_files["percentile_score"] = default_percentile_path
            logger.info(
                f"Saved default percentile_score.pkl (from '{first_field}') for backward compatibility"
            )

        # Return results
        results = {
            "status": "success",
            "calibration_method": "percentile_score_mapping",
            "mode": "multi-task" if is_multitask else "single-task",
            "num_tasks": len(task_calibrations),
            "score_fields": list(task_calibrations.keys()),
            "output_files": output_files,
            "config": {
                "n_bins": n_bins,
                "accuracy": accuracy,
                "calibration_dict_size": len(standard_calibration_dict),
                "job_type": job_type,
            },
        }

        logger.info("Percentile model calibration completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error in percentile model calibration: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Percentile Model Calibration Script for SageMaker Processing"
    )
    parser.add_argument(
        "--job_type",
        type=str,
        default="calibration",
        help="Job type - one of: training, calibration, validation, testing",
    )
    args = parser.parse_args()

    logger.info(
        f"Starting percentile model calibration from command line with job_type: {args.job_type}"
    )

    # Define standard SageMaker paths
    INPUT_DATA_PATH = "/opt/ml/processing/input/eval_data"
    INPUT_CALIBRATION_CONFIG_PATH = "/opt/ml/code/calibration"
    OUTPUT_CALIBRATION_PATH = "/opt/ml/processing/output/calibration"
    OUTPUT_METRICS_PATH = "/opt/ml/processing/output/metrics"
    OUTPUT_CALIBRATED_DATA_PATH = "/opt/ml/processing/output/calibrated_data"

    # Parse environment variables (multi-task support via SCORE_FIELDS)
    environ_vars = {
        "N_BINS": os.environ.get("N_BINS", "1000"),
        "SCORE_FIELD": os.environ.get(
            "SCORE_FIELD", "prob_class_1"
        ),  # Single-task fallback
        "SCORE_FIELDS": os.environ.get(
            "SCORE_FIELDS", ""
        ),  # Multi-task: comma-separated list
        "ACCURACY": os.environ.get("ACCURACY", "1e-5"),
    }

    # Set up input and output paths
    input_paths = {
        "evaluation_data": INPUT_DATA_PATH,
        "calibration_config": INPUT_CALIBRATION_CONFIG_PATH,  # Optional calibration config path
    }
    output_paths = {
        "calibration_output": OUTPUT_CALIBRATION_PATH,
        "metrics_output": OUTPUT_METRICS_PATH,
        "calibrated_data": OUTPUT_CALIBRATED_DATA_PATH,
    }

    # Call the main function
    try:
        results = main(input_paths, output_paths, environ_vars, args)
        if results["status"] == "success":
            logger.info("Calibration completed successfully")
            sys.exit(0)
        else:
            logger.error(
                f"Calibration failed: {results.get('error_message', 'Unknown error')}"
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
