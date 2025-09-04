"""Default configuration for the pipeline runtime testing system."""

from pathlib import Path
import os
from typing import Dict, Any, List

# Workspace Configuration
DEFAULT_WORKSPACE_DIR = "./development/projects/project_alpha"
DEFAULT_LOG_LEVEL = "INFO"

# Data Configuration
DEFAULT_DATA_SOURCE = "synthetic"
DEFAULT_DATA_SIZE = "small"
DATA_SIZE_MAPPING = {
    "small": {"records": 100, "features": 5},
    "medium": {"records": 1000, "features": 10},
    "large": {"records": 10000, "features": 20}
}

# Script Discovery Configuration
SCRIPT_SEARCH_PATHS = [
    "src/cursus/steps/scripts",
    "cursus/steps/scripts",
    "scripts",
    "dockers/xgboost_atoz/scripts",
    "dockers/pytorch_bsm_ext/scripts",
    "dockers/xgboost_pda/scripts"
]

# System Configuration
DEFAULT_TIMEOUT = 300  # seconds
MEMORY_MONITORING_ENABLED = True
DEFAULT_RANDOM_SEED = 42

class DefaultConfig:
    """Default configuration for the pipeline runtime testing system."""
    
    def __init__(self):
        """Initialize default configuration."""
        self.workspace_dir = Path(DEFAULT_WORKSPACE_DIR)
        self.log_level = DEFAULT_LOG_LEVEL
        self.data_source = DEFAULT_DATA_SOURCE
        self.data_size = DEFAULT_DATA_SIZE
        self.script_search_paths = SCRIPT_SEARCH_PATHS
        self.timeout = DEFAULT_TIMEOUT
        self.memory_monitoring_enabled = MEMORY_MONITORING_ENABLED
        self.random_seed = DEFAULT_RANDOM_SEED
        
        # Initialize workspace directories
        self._initialize_workspace()
    
    def _initialize_workspace(self):
        """Initialize workspace directories."""
        if not self.workspace_dir.exists():
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            
        for subdir in ["inputs", "outputs", "metadata", "logs"]:
            subdir_path = self.workspace_dir / subdir
            if not subdir_path.exists():
                subdir_path.mkdir(exist_ok=True)
    
    def get_script_discovery_paths(self) -> List[str]:
        """Get list of paths for script discovery."""
        return [str(Path(p)) for p in self.script_search_paths]
    
    def get_data_size_config(self) -> Dict[str, Any]:
        """Get configuration for current data size."""
        return DATA_SIZE_MAPPING.get(self.data_size, DATA_SIZE_MAPPING["small"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "workspace_dir": str(self.workspace_dir),
            "log_level": self.log_level,
            "data_source": self.data_source,
            "data_size": self.data_size,
            "script_search_paths": self.script_search_paths,
            "timeout": self.timeout,
            "memory_monitoring_enabled": self.memory_monitoring_enabled,
            "random_seed": self.random_seed,
            "data_size_config": self.get_data_size_config()
        }

# Default instance for easy importing
default_config = DefaultConfig()
