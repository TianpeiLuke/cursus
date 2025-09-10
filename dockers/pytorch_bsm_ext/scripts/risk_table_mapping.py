#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
import json
import pickle as pkl
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json_config(config_path):
    """Loads a JSON configuration file."""
    with open(config_path, "r") as file:
        return json.load(file)


class OfflineBinning:
    """A class to create risk tables for categorical features."""
    def __init__(self, metadata, tag):
        self.risk_tables: dict[str, dict] = {}
        self.metadata = metadata
        self.target = tag
        # Only consider categorical columns (except for a specific one if needed)
        self.variables = [
            v for v in metadata["varname"]
            if v in set(self.metadata.loc[self.metadata.iscategory, "varname"])
        ]

    def fit(self, df: pd.DataFrame, smooth_factor: float = 0, count_threshold: int = 0):
        """Fits the risk tables based on the provided dataframe."""
        # Drop any -1 or NaN target rows for fitting
        fit_df = df.loc[(df[self.target] != -1) & (~df[self.target].isnull())].copy()
        default_risk = float(fit_df[self.target].mean())
        smooth_samples = int(len(fit_df) * smooth_factor)

        for var in self.variables:
            if var not in fit_df.columns:
                continue

            self.risk_tables[var] = {
                "varName": var, "type": "categorical",
                "mode": self.metadata.loc[self.metadata.varname == var, "datatype"].iat[0],
                "default_bin": default_risk
            }
            if fit_df[var].isnull().all():
                self.risk_tables[var]["bins"] = {}
                continue

            risk_table = self._create_risk_table(fit_df, var, default_risk, smooth_samples, count_threshold)
            self.risk_tables[var]["bins"] = risk_table

    def _create_risk_table(self, df, variable, default_risk, samples, count_threshold):
        """Helper to calculate the risk table for a single variable."""
        cross_tab = pd.crosstab(df[variable], df[self.target].astype(object), margins=True, margins_name="_count_", dropna=False)
        cross_tab["risk"] = cross_tab.apply(lambda x: x.get(1, 0.0) / (x.get(1, 0) + x.get(0, 0)), axis=1)
        cross_tab["smooth_risk"] = cross_tab.apply(
            lambda x: ((x["_count_"] * x["risk"] + samples * default_risk) / (x["_count_"] + samples))
            if x["_count_"] >= count_threshold else default_risk, axis=1
        )
        cross_tab = cross_tab.loc[cross_tab.index != "_count_"]
        return dict(zip(cross_tab.index, cross_tab["smooth_risk"]))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe using the fitted risk tables."""
        df_transformed = df.copy()
        for var, risk_table_info in self.risk_tables.items():
            if var in df_transformed.columns:
                bins = risk_table_info["bins"]
                default_bin = risk_table_info["default_bin"]
                df_transformed[var] = df_transformed[var].map(bins).fillna(default_bin)
        return df_transformed


class MissingValueImputation:
    """A class to handle missing value imputation for numeric features."""
    def __init__(self, metadata):
        self.metadata = metadata
        self.imputers: dict[str, SimpleImputer] = {}
        self.numeric_variables = [
            v for v in metadata["varname"]
            if metadata.loc[metadata["varname"] == v, "datatype"].iat[0] == "numeric"
        ]

    def fit(self, df: pd.DataFrame):
        """Fits imputers for numeric variables based on the provided dataframe."""
        for var in self.numeric_variables:
            if var in df.columns:
                impute_strategy = self.metadata.loc[self.metadata["varname"] == var, "impute_strategy"].iat[0]
                if impute_strategy and impute_strategy != "none":
                    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
                    imputer.fit(df[var].values.reshape(-1, 1))
                    self.imputers[var] = imputer

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe using the fitted imputers."""
        df_transformed = df.copy()
        for var, imputer in self.imputers.items():
            if var in df_transformed.columns:
                df_transformed[var] = imputer.transform(df_transformed[var].values.reshape(-1, 1))
        return df_transformed


def main(job_type: str, input_dir: str, output_dir: str, config: dict, train_ratio: float, test_val_ratio: float):
    """
    Main logic for fitting, transforming, and saving risk tables and imputed data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = pd.DataFrame(config["metadata"])
    tag = config["tag"]

    # Initialize transformers
    binner = OfflineBinning(metadata, tag)
    imputer = MissingValueImputation(metadata)
    
    # --- Logic for Training mode ---
    if job_type == 'training':
        logger.info("Running in 'training' mode: fitting, transforming, and splitting.")
        # Load the single, unsplit dataset
        df = pd.read_csv(Path(input_dir) / "data.csv")
        
        # Fit on the entire dataset
        binner.fit(df, **config["model_training_config"]["category_risk_params"])
        imputer.fit(df)
        
        # Transform the entire dataset before splitting
        df_imputed = imputer.transform(df)
        df_transformed = binner.transform(df_imputed)
        
        # Split the transformed data
        train_df, holdout_df = train_test_split(df_transformed, train_size=train_ratio, random_state=42, stratify=df_transformed[tag])
        test_df, val_df = train_test_split(holdout_df, test_size=test_val_ratio, random_state=42, stratify=holdout_df[tag])
        
        # Save splits
        for split_name, split_df in [("train", train_df), ("test", test_df), ("val", val_df)]:
            split_output_dir = output_path / split_name
            split_output_dir.mkdir(exist_ok=True)
            split_df.to_csv(split_output_dir / f"{split_name}_processed_data.csv", index=False)
            logger.info(f"Saved transformed {split_name} data to {split_output_dir}")

    # --- Logic for Inference/Validation modes ---
    else:
        logger.info(f"Running in '{job_type}' mode: fitting on train, transforming all splits.")
        # Load all splits: fit on train, transform all
        train_df = pd.read_csv(Path(input_dir) / "train" / "train_processed_data.csv")
        val_df = pd.read_csv(Path(input_dir) / "validation" / "validation_processed_data.csv")
        test_df = pd.read_csv(Path(input_dir) / "test" / "test_processed_data.csv")
        
        # Fit only on the training data
        binner.fit(train_df, **config["model_training_config"]["category_risk_params"])
        imputer.fit(train_df)
        
        # Transform all three datasets
        datasets_to_transform = {"train": train_df, "validation": val_df, "test": test_df}
        for split_name, df_to_transform in datasets_to_transform.items():
            df_imputed = imputer.transform(df_to_transform)
            df_transformed = binner.transform(df_imputed)
            
            # Save transformed data to corresponding output subfolder
            split_output_dir = output_path / split_name
            split_output_dir.mkdir(exist_ok=True)
            df_transformed.to_csv(split_output_dir / f"{split_name}_processed_data.csv", index=False)
            logger.info(f"Saved transformed {split_name} data to {split_output_dir}")
            
    # --- Save fitted artifacts ---
    bin_output_path = output_path / "bin_mapping.pkl"
    with open(bin_output_path, "wb") as f:
        pkl.dump(binner.risk_tables, f)
    logger.info(f"Saved binning mapping to {bin_output_path}")

    impute_output_path = output_path / "missing_value_imputation.pkl"
    imputation_dict = {var: imp.statistics_[0] for var, imp in imputer.imputers.items()}
    with open(impute_output_path, "wb") as f:
        pkl.dump(imputation_dict, f)
    logger.info(f"Saved imputation model to {impute_output_path}")
    
    config_output_path = output_path / "config.pkl"
    with open(config_output_path, "wb") as f:
        config['metadata'] = metadata.to_dict() # Serialize DataFrame for pkl
        pkl.dump(config, f)
    logger.info(f"Saved final config to {config_output_path}")

    print("Risk-table mapping complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True, choices=["training", "validation", "testing"])
    args = parser.parse_args()
    
    # Define standard SageMaker paths
    input_dir = "/opt/ml/processing/input/data"
    config_dir = "/opt/ml/processing/input/config"
    output_dir = "/opt/ml/processing/output"

    # Load master config and metadata
    config = load_json_config(os.path.join(config_dir, "config.json"))
    metadata_df = pd.read_csv(os.path.join(config_dir, "metadata.csv"))
    config["metadata"] = metadata_df
    
    # Load environment variables for split ratios
    train_ratio = float(os.environ.get("TRAIN_RATIO", 0.7))
    test_val_ratio = float(os.environ.get("TEST_VAL_RATIO", 0.5))

    # Execute the main logic
    main(
        job_type=args.job_type,
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
        train_ratio=train_ratio,
        test_val_ratio=test_val_ratio
    )