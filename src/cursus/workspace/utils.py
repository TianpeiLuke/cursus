"""
Workspace Utilities - Phase 4 High-Level API Creation

This module provides utility functions for workspace operations,
including path management, configuration helpers, and common operations.
"""

import os
import shutil
import hashlib
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging

from pydantic import BaseModel, Field, ConfigDict


class WorkspaceConfig(BaseModel):
    """Workspace configuration model."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    workspace_id: str = Field(..., min_length=1, description="Workspace identifier")
    base_path: Path = Field(..., description="Base workspace path")
    isolation_mode: str = Field("strict", description="Isolation mode")
    auto_cleanup: bool = Field(True, description="Enable automatic cleanup")
    cleanup_threshold_days: int = Field(30, ge=1, description="Days before cleanup")
    max_workspace_size_mb: Optional[int] = Field(None, ge=1, description="Max workspace size in MB")
    allowed_extensions: List[str] = Field(default_factory=lambda: [".py", ".yaml", ".json", ".md", ".txt"])
    excluded_patterns: List[str] = Field(default_factory=lambda: ["__pycache__", "*.pyc", ".git", ".DS_Store"])


class PathUtils:
    """Utilities for path operations."""
    
    @staticmethod
    def normalize_workspace_path(path: Union[str, Path]) -> Path:
        """
        Normalize workspace path to absolute path.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized absolute path
        """
        return Path(path).resolve()
    
    @staticmethod
    def is_safe_path(path: Union[str, Path], base_path: Union[str, Path]) -> bool:
        """
        Check if path is safe (within base path).
        
        Args:
            path: Path to check
            base_path: Base path for safety check
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            path = Path(path).resolve()
            base_path = Path(base_path).resolve()
            return str(path).startswith(str(base_path))
        except Exception:
            return False
    
    @staticmethod
    def get_relative_path(path: Union[str, Path], base_path: Union[str, Path]) -> Optional[Path]:
        """
        Get relative path from base path.
        
        Args:
            path: Target path
            base_path: Base path
            
        Returns:
            Relative path if possible, None otherwise
        """
        try:
            path = Path(path).resolve()
            base_path = Path(base_path).resolve()
            return path.relative_to(base_path)
        except Exception:
            return None
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> bool:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_directory_size(path: Union[str, Path]) -> int:
        """
        Get total size of directory in bytes.
        
        Args:
            path: Directory path
            
        Returns:
            Size in bytes
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
    
    @staticmethod
    def clean_path_patterns(path: Union[str, Path], patterns: List[str]) -> int:
        """
        Clean files/directories matching patterns.
        
        Args:
            path: Base path to clean
            patterns: Patterns to match for cleanup
            
        Returns:
            Number of items cleaned
        """
        cleaned_count = 0
        path = Path(path)
        
        try:
            for pattern in patterns:
                for item in path.rglob(pattern):
                    try:
                        if item.is_file():
                            item.unlink()
                            cleaned_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            cleaned_count += 1
                    except Exception:
                        continue
        except Exception:
            pass
        
        return cleaned_count


class ConfigUtils:
    """Utilities for configuration management."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return None
        except Exception:
            return None
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        config_path = Path(config_path)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
                else:
                    return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Validation schema
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Simple validation - can be extended with jsonschema
        for key, requirements in schema.items():
            if requirements.get('required', False) and key not in config:
                errors.append(f"Required field '{key}' is missing")
            
            if key in config:
                value = config[key]
                expected_type = requirements.get('type')
                
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field '{key}' should be of type {expected_type.__name__}")
        
        return len(errors) == 0, errors


class FileUtils:
    """Utilities for file operations."""
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> Optional[str]:
        """
        Calculate hash of file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use
            
        Returns:
            Hash string or None if failed
        """
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception:
            return None
    
    @staticmethod
    def is_text_file(file_path: Union[str, Path]) -> bool:
        """
        Check if file is a text file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if text file, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' not in chunk
        except Exception:
            return False
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Create backup of file.
        
        Args:
            file_path: Path to file to backup
            backup_dir: Directory for backup (default: same directory)
            
        Returns:
            Path to backup file or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if backup_dir:
                backup_dir = Path(backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / f"{file_path.name}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception:
            return None
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = True) -> List[Path]:
        """
        Find files matching pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        directory = Path(directory)
        
        try:
            if recursive:
                return list(directory.rglob(pattern))
            else:
                return list(directory.glob(pattern))
        except Exception:
            return []


class ValidationUtils:
    """Utilities for validation operations."""
    
    @staticmethod
    def validate_workspace_structure(workspace_path: Union[str, Path], required_dirs: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate workspace directory structure.
        
        Args:
            workspace_path: Path to workspace
            required_dirs: List of required directories
            
        Returns:
            Tuple of (is_valid, missing_directories)
        """
        workspace_path = Path(workspace_path)
        missing_dirs = []
        
        for required_dir in required_dirs:
            dir_path = workspace_path / required_dir
            if not dir_path.exists() or not dir_path.is_dir():
                missing_dirs.append(required_dir)
        
        return len(missing_dirs) == 0, missing_dirs
    
    @staticmethod
    def validate_file_extensions(directory: Union[str, Path], allowed_extensions: List[str]) -> Tuple[bool, List[Path]]:
        """
        Validate file extensions in directory.
        
        Args:
            directory: Directory to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Tuple of (is_valid, invalid_files)
        """
        directory = Path(directory)
        invalid_files = []
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    if file_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                        invalid_files.append(file_path)
        except Exception:
            pass
        
        return len(invalid_files) == 0, invalid_files
    
    @staticmethod
    def validate_workspace_size(workspace_path: Union[str, Path], max_size_mb: int) -> Tuple[bool, int]:
        """
        Validate workspace size.
        
        Args:
            workspace_path: Path to workspace
            max_size_mb: Maximum size in MB
            
        Returns:
            Tuple of (is_valid, current_size_mb)
        """
        current_size_bytes = PathUtils.get_directory_size(workspace_path)
        current_size_mb = current_size_bytes // (1024 * 1024)
        
        return current_size_mb <= max_size_mb, current_size_mb


class TimeUtils:
    """Utilities for time-based operations."""
    
    @staticmethod
    def is_path_older_than(path: Union[str, Path], days: int) -> bool:
        """
        Check if path is older than specified days.
        
        Args:
            path: Path to check
            days: Number of days
            
        Returns:
            True if older than specified days, False otherwise
        """
        try:
            path = Path(path)
            if not path.exists():
                return False
            
            modified_time = datetime.fromtimestamp(path.stat().st_mtime)
            threshold = datetime.now() - timedelta(days=days)
            
            return modified_time < threshold
        except Exception:
            return False
    
    @staticmethod
    def get_path_age_days(path: Union[str, Path]) -> Optional[int]:
        """
        Get age of path in days.
        
        Args:
            path: Path to check
            
        Returns:
            Age in days or None if failed
        """
        try:
            path = Path(path)
            if not path.exists():
                return None
            
            modified_time = datetime.fromtimestamp(path.stat().st_mtime)
            age = datetime.now() - modified_time
            
            return age.days
        except Exception:
            return None
    
    @staticmethod
    def format_timestamp(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format timestamp to string.
        
        Args:
            timestamp: Timestamp to format (default: now)
            format_str: Format string
            
        Returns:
            Formatted timestamp string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        return timestamp.strftime(format_str)


class LoggingUtils:
    """Utilities for logging operations."""
    
    @staticmethod
    def setup_workspace_logger(workspace_id: str, log_level: str = "INFO") -> logging.Logger:
        """
        Set up logger for workspace operations.
        
        Args:
            workspace_id: Workspace identifier
            log_level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f"workspace.{workspace_id}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {workspace_id} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @staticmethod
    def log_workspace_operation(logger: logging.Logger, operation: str, details: Dict[str, Any]):
        """
        Log workspace operation with structured details.
        
        Args:
            logger: Logger instance
            operation: Operation name
            details: Operation details
        """
        logger.info(f"Operation: {operation}", extra={"operation_details": details})


class WorkspaceUtils:
    """High-level workspace utility functions."""
    
    @staticmethod
    def create_workspace_config(workspace_id: str, base_path: Union[str, Path], **kwargs) -> WorkspaceConfig:
        """
        Create workspace configuration.
        
        Args:
            workspace_id: Workspace identifier
            base_path: Base workspace path
            **kwargs: Additional configuration options
            
        Returns:
            WorkspaceConfig instance
        """
        config_data = {
            "workspace_id": workspace_id,
            "base_path": Path(base_path),
            **kwargs
        }
        
        return WorkspaceConfig(**config_data)
    
    @staticmethod
    def initialize_workspace_directory(workspace_path: Union[str, Path], config: WorkspaceConfig) -> bool:
        """
        Initialize workspace directory structure.
        
        Args:
            workspace_path: Path to workspace
            config: Workspace configuration
            
        Returns:
            True if successful, False otherwise
        """
        workspace_path = Path(workspace_path)
        
        try:
            # Create base directory
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create standard directories
            standard_dirs = ["builders", "configs", "contracts", "specs", "scripts", "tests"]
            for dir_name in standard_dirs:
                (workspace_path / dir_name).mkdir(exist_ok=True)
                (workspace_path / dir_name / "__init__.py").touch()
            
            # Save workspace configuration
            config_path = workspace_path / ".workspace_config.yaml"
            ConfigUtils.save_config(config.model_dump(), config_path)
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def cleanup_workspace(workspace_path: Union[str, Path], config: WorkspaceConfig) -> Tuple[bool, int]:
        """
        Clean up workspace according to configuration.
        
        Args:
            workspace_path: Path to workspace
            config: Workspace configuration
            
        Returns:
            Tuple of (success, items_cleaned)
        """
        try:
            items_cleaned = PathUtils.clean_path_patterns(workspace_path, config.excluded_patterns)
            return True, items_cleaned
        except Exception:
            return False, 0
    
    @staticmethod
    def validate_workspace(workspace_path: Union[str, Path], config: WorkspaceConfig) -> Tuple[bool, List[str]]:
        """
        Validate workspace according to configuration.
        
        Args:
            workspace_path: Path to workspace
            config: Workspace configuration
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []
        
        # Check directory structure
        required_dirs = ["builders", "configs", "contracts", "specs", "scripts"]
        is_structure_valid, missing_dirs = ValidationUtils.validate_workspace_structure(workspace_path, required_dirs)
        if not is_structure_valid:
            errors.extend([f"Missing directory: {d}" for d in missing_dirs])
        
        # Check file extensions
        if config.allowed_extensions:
            is_extensions_valid, invalid_files = ValidationUtils.validate_file_extensions(workspace_path, config.allowed_extensions)
            if not is_extensions_valid:
                errors.extend([f"Invalid file extension: {f}" for f in invalid_files[:5]])  # Limit to first 5
        
        # Check workspace size
        if config.max_workspace_size_mb:
            is_size_valid, current_size = ValidationUtils.validate_workspace_size(workspace_path, config.max_workspace_size_mb)
            if not is_size_valid:
                errors.append(f"Workspace size ({current_size}MB) exceeds limit ({config.max_workspace_size_mb}MB)")
        
        return len(errors) == 0, errors
