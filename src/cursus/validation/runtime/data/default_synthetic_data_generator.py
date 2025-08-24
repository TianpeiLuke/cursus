"""Default synthetic data generator for testing pipeline scripts."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .base_synthetic_data_generator import BaseSyntheticDataGenerator

class DefaultSyntheticDataGenerator(BaseSyntheticDataGenerator):
    """
    Default implementation of synthetic data generator.
    
    Generates synthetic data for testing pipeline scripts without requiring access to production data.
    Provides basic data generation for common script types.
    
    Users can either:
    1. Use this class directly for basic data generation
    2. Inherit from BaseSyntheticDataGenerator to create custom generators
    3. Inherit from this class to extend the default behavior
    """
    
    def get_supported_scripts(self) -> List[str]:
        """Return list of supported script patterns."""
        return [
            "currency*", "conversion*",
            "tabular*", "preprocessing*", 
            "xgboost*", "training*",
            "calibration*", "model_calibration*",
            "*"  # Fallback for any script
        ]
    
    def generate_for_script(self, script_name: str, 
                           data_size: str = "small") -> Dict[str, str]:
        """Generate synthetic data for specific script based on name patterns."""
        
        # Basic data generation based on script name patterns
        if "currency" in script_name.lower() or "conversion" in script_name.lower():
            return self._generate_currency_data(data_size)
        elif "tabular" in script_name.lower() or "preprocessing" in script_name.lower():
            return self._generate_tabular_data(data_size)
        elif "xgboost" in script_name.lower() or "training" in script_name.lower():
            return self._generate_training_data(data_size)
        elif "calibration" in script_name.lower() or "model_calibration" in script_name.lower():
            return self._generate_calibration_data(data_size)
        else:
            return self._generate_generic_data(data_size)
    
    def _generate_currency_data(self, data_size: str) -> Dict[str, str]:
        """Generate currency conversion test data"""
        
        size_map = {"small": 100, "medium": 1000, "large": 10000}
        num_records = size_map.get(data_size, 100)
        
        # Generate currency data
        currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CNY", "INR"]
        
        data = []
        for _ in range(num_records):
            from_currency = np.random.choice(currencies)
            to_currency = np.random.choice([c for c in currencies if c != from_currency])
            
            data.append({
                "from_currency": from_currency,
                "to_currency": to_currency,
                "amount": np.random.uniform(1, 10000),
                "date": (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d")
            })
        
        df = pd.DataFrame(data)
        
        # Save to temporary file
        output_path = Path("./temp_currency_data.csv")
        df.to_csv(output_path, index=False)
        
        return {"input": str(output_path)}
    
    def _generate_tabular_data(self, data_size: str) -> Dict[str, str]:
        """Generate tabular preprocessing test data"""
        
        size_map = {"small": 500, "medium": 5000, "large": 50000}
        num_records = size_map.get(data_size, 500)
        
        # Generate tabular data with various data types
        data = {
            "id": range(1, num_records + 1),
            "feature_1": np.random.normal(0, 1, num_records),
            "feature_2": np.random.uniform(-10, 10, num_records),
            "feature_3": np.random.choice(["A", "B", "C"], num_records),
            "feature_4": np.random.exponential(1, num_records),
            "feature_5": np.random.binomial(10, 0.5, num_records),
            "target": np.random.choice([0, 1], num_records)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        missing_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False)
        df.loc[missing_indices, "feature_1"] = np.nan
        
        # Save to temporary file
        output_path = Path("./temp_tabular_data.csv")
        df.to_csv(output_path, index=False)
        
        return {"input": str(output_path)}
    
    def _generate_training_data(self, data_size: str) -> Dict[str, str]:
        """Generate training data for ML scripts"""
        
        size_map = {"small": 1000, "medium": 10000, "large": 100000}
        num_records = size_map.get(data_size, 1000)
        
        # Generate training dataset with XGBoost-friendly features
        num_features = 10
        X = np.random.normal(0, 1, (num_records, num_features))
        
        # Create a non-linear relationship for the target
        y = (X[:, 0] + X[:, 1]**2 + np.sin(X[:, 2]) + 
             np.exp(X[:, 3]/10) + np.random.normal(0, 0.1, num_records) > 1).astype(int)
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(num_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data["target"] = y
        
        # Add categorical feature
        data["category"] = np.random.choice(["cat_A", "cat_B", "cat_C", "cat_D"], num_records)
        
        # Save to temporary file
        output_path = Path("./temp_training_data.csv")
        data.to_csv(output_path, index=False)
        
        return {"input": str(output_path)}
    
    def _generate_calibration_data(self, data_size: str) -> Dict[str, str]:
        """Generate model calibration data"""
        
        size_map = {"small": 500, "medium": 5000, "large": 50000}
        num_records = size_map.get(data_size, 500)
        
        # Generate predictions and actual values for calibration
        predictions = np.random.uniform(0, 1, num_records)
        
        # Create some bias in the predictions for calibration to correct
        adjusted_predictions = np.power(predictions, 1.5)  # Biased predictions
        
        # Generate actual values with correlation to adjusted predictions
        actuals = (adjusted_predictions > np.random.uniform(0, 1, num_records)).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            "prediction": predictions,
            "actual": actuals
        })
        
        # Add some metadata columns
        data["segment"] = np.random.choice(["seg_A", "seg_B", "seg_C"], num_records)
        data["timestamp"] = [(datetime.now() - timedelta(days=i % 30)).strftime("%Y-%m-%d") 
                             for i in range(num_records)]
        
        # Save to temporary file
        output_path = Path("./temp_calibration_data.csv")
        data.to_csv(output_path, index=False)
        
        return {"input": str(output_path)}
    
    def _generate_generic_data(self, data_size: str) -> Dict[str, str]:
        """Generate generic test data for unknown script types"""
        
        size_map = {"small": 100, "medium": 1000, "large": 10000}
        num_records = size_map.get(data_size, 100)
        
        # Create a generic dataset with various data types
        data = {
            "id": range(1, num_records + 1),
            "numeric_value": np.random.normal(0, 1, num_records),
            "integer_value": np.random.randint(1, 100, num_records),
            "category": np.random.choice(["X", "Y", "Z"], num_records),
            "boolean": np.random.choice([True, False], num_records),
            "timestamp": [(datetime.now() - timedelta(days=i % 30)).strftime("%Y-%m-%d") 
                         for i in range(num_records)]
        }
        
        df = pd.DataFrame(data)
        
        # Save to temporary file
        output_path = Path("./temp_generic_data.csv")
        df.to_csv(output_path, index=False)
        
        return {"input": str(output_path)}
