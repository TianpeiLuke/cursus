"""Base synthetic data generator for testing pipeline scripts."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

class BaseSyntheticDataGenerator(ABC):
    """
    Base class for generating synthetic data for testing pipeline scripts.
    
    Users should inherit from this class and implement the abstract methods
    to provide custom data generation logic for their specific use cases.
    
    Example:
        class MyDataGenerator(BaseSyntheticDataGenerator):
            def get_supported_scripts(self) -> List[str]:
                return ["my_script", "another_script*"]
            
            def generate_for_script(self, script_name: str, data_size: str = "small", **kwargs) -> Dict[str, str]:
                if script_name == "my_script":
                    # Generate custom data for my_script
                    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
                    path = self.save_dataframe(df, "my_data.csv")
                    return {"input": path}
                # ... handle other scripts
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize data generator with optional random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
    
    @abstractmethod
    def generate_for_script(self, script_name: str, 
                           data_size: str = "small", 
                           **kwargs) -> Dict[str, str]:
        """
        Generate synthetic data for specific script based on name patterns.
        
        Args:
            script_name: Name of the script to generate data for
            data_size: Size of data to generate ("small", "medium", "large")
            **kwargs: Additional parameters for data generation
            
        Returns:
            Dictionary mapping logical names to file paths of generated data
            
        Example:
            return {"input": "/path/to/input.csv", "config": "/path/to/config.json"}
        """
        pass
    
    @abstractmethod
    def get_supported_scripts(self) -> List[str]:
        """
        Return list of script names or patterns that this generator supports.
        
        Returns:
            List of script names or patterns (e.g., ["xgboost_training", "tabular_*"])
            
        Note:
            Patterns can use simple wildcards (*) for prefix matching.
            For example, "tabular_*" will match "tabular_preprocessing", "tabular_training", etc.
        """
        pass
    
    def supports_script(self, script_name: str) -> bool:
        """
        Check if this generator supports the given script.
        
        Args:
            script_name: Name of the script to check
            
        Returns:
            True if the script is supported, False otherwise
        """
        supported = self.get_supported_scripts()
        
        # Check exact matches first
        if script_name in supported:
            return True
        
        # Check pattern matches (simple wildcard support)
        for pattern in supported:
            if "*" in pattern:
                # Simple wildcard matching
                prefix = pattern.split("*")[0]
                if script_name.startswith(prefix):
                    return True
            elif pattern.lower() in script_name.lower():
                return True
        
        return False
    
    def get_data_size_config(self, data_size: str) -> Dict[str, int]:
        """
        Get configuration for different data sizes.
        
        Args:
            data_size: Size specification ("small", "medium", "large")
            
        Returns:
            Dictionary with size configuration
        """
        size_configs = {
            "small": {"num_records": 100, "num_features": 5},
            "medium": {"num_records": 1000, "num_features": 10},
            "large": {"num_records": 10000, "num_features": 20}
        }
        return size_configs.get(data_size, size_configs["small"])
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, 
                      output_dir: Optional[str] = None) -> str:
        """
        Save DataFrame to file and return the path.
        
        Args:
            df: DataFrame to save
            filename: Name of the file (with extension)
            output_dir: Directory to save to (defaults to current directory)
            
        Returns:
            Path to the saved file
        """
        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(filename)
        
        # Determine file format from extension
        if filename.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif filename.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        elif filename.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def save_json(self, data: Dict[str, Any], filename: str,
                 output_dir: Optional[str] = None) -> str:
        """
        Save dictionary data to JSON file and return the path.
        
        Args:
            data: Dictionary to save
            filename: Name of the file (with .json extension)
            output_dir: Directory to save to (defaults to current directory)
            
        Returns:
            Path to the saved file
        """
        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(filename)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(output_path)
    
    def generate_random_dataframe(self, num_records: int, columns: Dict[str, str],
                                 output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Helper method to generate random DataFrame with specified column types.
        
        Args:
            num_records: Number of records to generate
            columns: Dictionary mapping column names to types
                    Supported types: "int", "float", "category", "bool", "datetime"
            output_dir: Optional output directory (not used, for consistency)
            
        Returns:
            Generated DataFrame
            
        Example:
            df = self.generate_random_dataframe(100, {
                "id": "int",
                "value": "float", 
                "category": "category",
                "flag": "bool",
                "date": "datetime"
            })
        """
        data = {}
        
        for col_name, col_type in columns.items():
            if col_type == "int":
                data[col_name] = np.random.randint(1, 1000, num_records)
            elif col_type == "float":
                data[col_name] = np.random.normal(0, 1, num_records)
            elif col_type == "category":
                categories = ["A", "B", "C", "D", "E"]
                data[col_name] = np.random.choice(categories, num_records)
            elif col_type == "bool":
                data[col_name] = np.random.choice([True, False], num_records)
            elif col_type == "datetime":
                base_date = datetime.now()
                data[col_name] = [
                    (base_date - timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d")
                    for _ in range(num_records)
                ]
            else:
                # Default to string
                data[col_name] = [f"value_{i}" for i in range(num_records)]
        
        return pd.DataFrame(data)
