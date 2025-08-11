"""
Common utilities for alignment validation.

Provides shared data structures, enums, and helper functions used across
all alignment validation components.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import difflib


class SeverityLevel(Enum):
    """Severity levels for alignment issues."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlignmentLevel(Enum):
    """Alignment validation levels."""
    SCRIPT_CONTRACT = 1
    CONTRACT_SPECIFICATION = 2
    SPECIFICATION_DEPENDENCY = 3
    BUILDER_CONFIGURATION = 4


class AlignmentIssue(BaseModel):
    """
    Represents an alignment issue found during validation.
    
    Attributes:
        level: Severity level of the issue
        category: Category of the alignment issue
        message: Human-readable description of the issue
        details: Additional details about the issue
        recommendation: Suggested fix for the issue
        alignment_level: Which alignment level this issue affects
    """
    level: SeverityLevel
    category: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    alignment_level: Optional[AlignmentLevel] = None


class PathReference(BaseModel):
    """
    Represents a path reference found in script analysis.
    
    Attributes:
        path: The path string found
        line_number: Line number where the path was found
        context: Surrounding code context
        is_hardcoded: Whether this is a hardcoded path
        construction_method: How the path is constructed (e.g., 'os.path.join')
    """
    path: str
    line_number: int
    context: str
    is_hardcoded: bool = True
    construction_method: Optional[str] = None


class EnvVarAccess(BaseModel):
    """
    Represents environment variable access found in script analysis.
    
    Attributes:
        variable_name: Name of the environment variable
        line_number: Line number where the access was found
        context: Surrounding code context
        access_method: How the variable is accessed (e.g., 'os.environ', 'os.getenv')
        has_default: Whether a default value is provided
        default_value: The default value if provided
    """
    variable_name: str
    line_number: int
    context: str
    access_method: str
    has_default: bool = False
    default_value: Optional[str] = None


class ImportStatement(BaseModel):
    """
    Represents an import statement found in script analysis.
    
    Attributes:
        module_name: Name of the imported module
        import_alias: Alias used for the import (if any)
        line_number: Line number where the import was found
        is_from_import: Whether this is a 'from X import Y' statement
        imported_items: List of specific items imported (for from imports)
    """
    module_name: str
    import_alias: Optional[str]
    line_number: int
    is_from_import: bool = False
    imported_items: List[str] = Field(default_factory=list)


class ArgumentDefinition(BaseModel):
    """
    Represents a command-line argument definition found in script analysis.
    
    Attributes:
        argument_name: Name of the argument (without dashes)
        line_number: Line number where the argument was defined
        is_required: Whether the argument is required
        has_default: Whether the argument has a default value
        default_value: The default value if provided
        argument_type: Type of the argument (str, int, etc.)
        choices: Valid choices for the argument (if any)
    """
    argument_name: str
    line_number: int
    is_required: bool = False
    has_default: bool = False
    default_value: Optional[Any] = None
    argument_type: Optional[str] = None
    choices: Optional[List[str]] = None


class PathConstruction(BaseModel):
    """
    Represents a dynamic path construction found in script analysis.
    
    Attributes:
        base_path: The base path being constructed from
        construction_parts: Parts used in the construction
        line_number: Line number where the construction was found
        context: Surrounding code context
        method: Method used for construction (e.g., 'os.path.join', 'pathlib')
    """
    base_path: str
    construction_parts: List[str]
    line_number: int
    context: str
    method: str


class FileOperation(BaseModel):
    """
    Represents a file operation found in script analysis.
    
    Attributes:
        file_path: Path to the file being operated on
        operation_type: Type of operation (read, write, append, etc.)
        line_number: Line number where the operation was found
        context: Surrounding code context
        mode: File mode used (if specified)
        method: Method used for the operation (e.g., 'open', 'tarfile.open', 'pandas.read_csv')
    """
    file_path: str
    operation_type: str
    line_number: int
    context: str
    mode: Optional[str] = None
    method: Optional[str] = None


def normalize_path(path: str) -> str:
    """
    Normalize a path for comparison purposes.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path string
    """
    import os
    return os.path.normpath(path).replace('\\', '/')


def extract_logical_name_from_path(path: str) -> Optional[str]:
    """
    Extract logical name from a SageMaker path.
    
    For paths like '/opt/ml/processing/input/data', extracts 'data'.
    
    Args:
        path: SageMaker path
        
    Returns:
        Logical name or None if not extractable
    """
    # Common SageMaker path patterns
    patterns = [
        '/opt/ml/processing/input/',
        '/opt/ml/processing/output/',
        '/opt/ml/input/data/',
        '/opt/ml/model/',
        '/opt/ml/output/'
    ]
    
    normalized_path = normalize_path(path)
    
    for pattern in patterns:
        if normalized_path.startswith(pattern):
            remainder = normalized_path[len(pattern):].strip('/')
            if remainder:
                # Return the first path component as logical name
                return remainder.split('/')[0]
    
    return None


def is_sagemaker_path(path: str) -> bool:
    """
    Check if a path is a SageMaker container path.
    
    Args:
        path: Path to check
        
    Returns:
        True if this is a SageMaker path
    """
    sagemaker_prefixes = [
        '/opt/ml/processing/',
        '/opt/ml/input/',
        '/opt/ml/model',
        '/opt/ml/output'
    ]
    
    normalized_path = normalize_path(path)
    return any(normalized_path.startswith(prefix) for prefix in sagemaker_prefixes)


def format_alignment_issue(issue: AlignmentIssue) -> str:
    """
    Format an alignment issue for display.
    
    Args:
        issue: The alignment issue to format
        
    Returns:
        Formatted string representation
    """
    level_emoji = {
        SeverityLevel.INFO: "ℹ️",
        SeverityLevel.WARNING: "⚠️", 
        SeverityLevel.ERROR: "❌",
        SeverityLevel.CRITICAL: "🚨"
    }
    
    emoji = level_emoji.get(issue.level, "")
    level_name = issue.level.value
    
    result = f"{emoji} {level_name}: {issue.message}"
    
    if issue.recommendation:
        result += f"\n  💡 Recommendation: {issue.recommendation}"
    
    if issue.details:
        result += f"\n  📋 Details: {issue.details}"
    
    return result


def group_issues_by_severity(issues: List[AlignmentIssue]) -> Dict[SeverityLevel, List[AlignmentIssue]]:
    """
    Group alignment issues by severity level.
    
    Args:
        issues: List of alignment issues
        
    Returns:
        Dictionary mapping severity levels to lists of issues
    """
    grouped = {level: [] for level in SeverityLevel}
    
    for issue in issues:
        grouped[issue.level].append(issue)
    
    return grouped


def get_highest_severity(issues: List[AlignmentIssue]) -> Optional[SeverityLevel]:
    """
    Get the highest severity level among a list of issues.
    
    Args:
        issues: List of alignment issues
        
    Returns:
        Highest severity level or None if no issues
    """
    if not issues:
        return None
    
    severity_order = [SeverityLevel.CRITICAL, SeverityLevel.ERROR, 
                     SeverityLevel.WARNING, SeverityLevel.INFO]
    
    for severity in severity_order:
        if any(issue.level == severity for issue in issues):
            return severity
    
    return None


def create_alignment_issue(
    level: SeverityLevel,
    category: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
    alignment_level: Optional[AlignmentLevel] = None
) -> AlignmentIssue:
    """
    Create an alignment issue with proper defaults.
    
    Args:
        level: Severity level
        category: Issue category
        message: Issue message
        details: Additional details
        recommendation: Suggested fix
        alignment_level: Which alignment level this affects
        
    Returns:
        AlignmentIssue instance
    """
    return AlignmentIssue(
        level=level,
        category=category,
        message=message,
        details=details or {},
        recommendation=recommendation,
        alignment_level=alignment_level
    )


class DependencyPattern(Enum):
    """Types of dependency patterns for classification."""
    PIPELINE_DEPENDENCY = "pipeline"
    EXTERNAL_INPUT = "external"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"


class DependencyPatternClassifier:
    """
    Classify dependencies by pattern type for appropriate validation.
    
    This classifier addresses the false positive issue where all dependencies
    are treated as pipeline dependencies, even when they are external inputs
    or configuration dependencies that don't require pipeline resolution.
    """
    
    def __init__(self):
        """Initialize the dependency pattern classifier."""
        self.external_patterns = {
            # Direct S3 upload patterns
            'pretrained_model_path',
            'hyperparameters_s3_uri',
            'model_s3_uri',
            'data_s3_uri',
            'config_s3_uri',
            # User-provided inputs
            'input_data_path',
            'model_input_path',
            'config_input_path',
        }
        
        self.configuration_patterns = {
            'config_',
            'hyperparameters',
            'parameters',
            'settings',
        }
        
        self.environment_patterns = {
            'env_',
            'environment_',
        }
    
    def classify_dependency(self, dependency_info: Dict[str, Any]) -> DependencyPattern:
        """
        Classify dependency pattern for appropriate validation.
        
        Args:
            dependency_info: Dictionary containing dependency information
                           Should have 'logical_name', 'dependency_type', 'compatible_sources', etc.
        
        Returns:
            DependencyPattern enum indicating the type of dependency
        """
        logical_name = dependency_info.get('logical_name', '').lower()
        dependency_type = dependency_info.get('dependency_type', '').lower()
        compatible_sources = dependency_info.get('compatible_sources', [])
        
        # Check for explicit external markers
        if (isinstance(compatible_sources, list) and 
            len(compatible_sources) == 1 and 
            compatible_sources[0] == "EXTERNAL"):
            return DependencyPattern.EXTERNAL_INPUT
        
        # Check for S3 URI patterns (external inputs)
        if (logical_name.endswith('_s3_uri') or 
            logical_name.endswith('_path') or
            logical_name in self.external_patterns):
            return DependencyPattern.EXTERNAL_INPUT
        
        # Check for configuration patterns
        if (logical_name.startswith('config_') or
            dependency_type == 'hyperparameters' or
            any(pattern in logical_name for pattern in self.configuration_patterns)):
            return DependencyPattern.CONFIGURATION
        
        # Check for environment variable patterns
        if (logical_name.startswith('env_') or
            any(pattern in logical_name for pattern in self.environment_patterns)):
            return DependencyPattern.ENVIRONMENT
        
        # Default to pipeline dependency
        return DependencyPattern.PIPELINE_DEPENDENCY
    
    def should_validate_pipeline_resolution(self, pattern: DependencyPattern) -> bool:
        """
        Determine if a dependency pattern requires pipeline resolution validation.
        
        Args:
            pattern: The dependency pattern
            
        Returns:
            True if pipeline resolution validation is required
        """
        return pattern == DependencyPattern.PIPELINE_DEPENDENCY
    
    def get_validation_message(self, pattern: DependencyPattern, logical_name: str) -> str:
        """
        Get appropriate validation message for a dependency pattern.
        
        Args:
            pattern: The dependency pattern
            logical_name: Name of the dependency
            
        Returns:
            Appropriate validation message
        """
        if pattern == DependencyPattern.EXTERNAL_INPUT:
            return f"External dependency '{logical_name}' - no pipeline resolution needed"
        elif pattern == DependencyPattern.CONFIGURATION:
            return f"Configuration dependency '{logical_name}' - validated through config system"
        elif pattern == DependencyPattern.ENVIRONMENT:
            return f"Environment dependency '{logical_name}' - validated through environment variables"
        else:
            return f"Pipeline dependency '{logical_name}' - requires pipeline resolution"


class FlexibleFileResolver:
    """
    Dynamic file resolution with file-system-driven discovery.
    
    This resolver discovers actual files in the filesystem and matches them
    to script names using intelligent pattern matching, eliminating the need
    for hardcoded mappings that become stale.
    """
    
    def __init__(self, base_directories: Dict[str, str]):
        """
        Initialize the file resolver with base directories.
        
        Args:
            base_directories: Dictionary mapping component types to their base directories
                             e.g., {'contracts': 'src/cursus/steps/contracts', ...}
        """
        self.base_dirs = {k: Path(v) for k, v in base_directories.items()}
        self.file_cache = {}  # Cache discovered files
        self._discover_all_files()
    
    def _discover_all_files(self):
        """Discover all actual files in each directory and extract base names."""
        for component_type, directory in self.base_dirs.items():
            self.file_cache[component_type] = self._scan_directory(directory, component_type)
    
    def _scan_directory(self, directory: Path, component_type: str) -> Dict[str, str]:
        """
        Scan directory and extract base names from actual files.
        
        Args:
            directory: Directory path to scan
            component_type: Type of component (contracts, specs, builders, configs)
            
        Returns:
            Dict mapping base_names to actual filenames
        """
        file_map = {}
        
        if not directory.exists():
            return file_map
        
        # Define patterns for each component type
        patterns = {
            'contracts': r'^(.+)_contract\.py$',
            'specs': r'^(.+)_spec\.py$', 
            'builders': r'^builder_(.+)_step\.py$',
            'configs': r'^config_(.+)_step\.py$'
        }
        
        pattern = patterns.get(component_type)
        if not pattern:
            return file_map
        
        import re
        regex = re.compile(pattern)
        
        for file_path in directory.glob('*.py'):
            if file_path.name.startswith('__'):
                continue
                
            match = regex.match(file_path.name)
            if match:
                base_name = match.group(1)
                file_map[base_name] = file_path.name
        
        return file_map
    
    def _find_best_match(self, script_name: str, component_type: str) -> Optional[str]:
        """
        Find best matching file for script name using multiple strategies.
        
        Args:
            script_name: Name of the script to find files for
            component_type: Type of component to search for
            
        Returns:
            Full path to the best matching file or None
        """
        available_files = self.file_cache.get(component_type, {})
        
        if not available_files:
            return None
        
        # Strategy 1: Exact match
        if script_name in available_files:
            return str(self.base_dirs[component_type] / available_files[script_name])
        
        # Strategy 2: Normalized matching
        normalized_script = self._normalize_name(script_name)
        for base_name, filename in available_files.items():
            if self._normalize_name(base_name) == normalized_script:
                return str(self.base_dirs[component_type] / filename)
        
        # Strategy 3: Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for base_name, filename in available_files.items():
            score = self._calculate_similarity(script_name, base_name)
            if score > 0.8 and score > best_score:  # 80% similarity threshold
                best_score = score
                best_match = str(self.base_dirs[component_type] / filename)
        
        return best_match
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize names for better matching.
        
        Handles common variations:
        - preprocess vs preprocessing
        - eval vs evaluation  
        - xgb vs xgboost
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
        """
        # Convert to lowercase
        normalized = name.lower()
        
        # Handle common word variations
        variations = {
            'preprocess': 'preprocessing',
            'eval': 'evaluation',
            'xgb': 'xgboost',
        }
        
        for short, long in variations.items():
            # Handle both directions
            if short in normalized and long not in normalized:
                normalized = normalized.replace(short, long)
        
        return normalized
    
    def refresh_cache(self):
        """Refresh file cache to pick up new files."""
        self._discover_all_files()
    
    def get_available_files_report(self) -> Dict[str, Dict[str, Any]]:
        """Get report of all discovered files for debugging."""
        report = {}
        for component_type, file_map in self.file_cache.items():
            report[component_type] = {
                'directory': str(self.base_dirs[component_type]),
                'discovered_files': list(file_map.values()),
                'base_names': list(file_map.keys()),
                'count': len(file_map)
            }
        return report
    
    def find_contract_file(self, script_name: str) -> Optional[str]:
        """
        Find contract file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the contract file or None if not found
        """
        return self._find_best_match(script_name, 'contracts')
    
    def find_spec_file(self, script_name: str) -> Optional[str]:
        """
        Find specification file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the specification file or None if not found
        """
        return self._find_best_match(script_name, 'specs')
    
    def find_specification_file(self, script_name: str) -> Optional[str]:
        """
        Alias for find_spec_file to maintain compatibility with existing code.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the specification file or None if not found
        """
        return self.find_spec_file(script_name)
    
    def find_builder_file(self, script_name: str) -> Optional[str]:
        """
        Find builder file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the builder file or None if not found
        """
        return self._find_best_match(script_name, 'builders')
    
    def find_config_file(self, script_name: str) -> Optional[str]:
        """
        Find config file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the config file or None if not found
        """
        return self._find_best_match(script_name, 'configs')
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using difflib.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    def find_all_component_files(self, script_name: str) -> Dict[str, Optional[str]]:
        """
        Find all component files for a given script.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Dictionary mapping component types to their file paths
        """
        return {
            'contract': self.find_contract_file(script_name),
            'spec': self.find_spec_file(script_name),
            'builder': self.find_builder_file(script_name),
            'config': self.find_config_file(script_name),
        }
    
    def extract_base_name_from_spec(self, spec_path: Path) -> str:
        """
        Extract the base name from a specification file path.
        
        For job type variant specifications like 'preprocessing_training_spec.py',
        this extracts 'preprocessing'.
        
        Args:
            spec_path: Path to the specification file
            
        Returns:
            Base name for the specification
        """
        stem = spec_path.stem  # Remove .py extension
        
        # Remove '_spec' suffix
        if stem.endswith('_spec'):
            stem = stem[:-5]
        
        # Remove job type suffix if present
        job_types = ['training', 'validation', 'testing', 'calibration']
        for job_type in job_types:
            if stem.endswith(f'_{job_type}'):
                return stem[:-len(job_type)-1]  # Remove _{job_type}
        
        return stem
    
    def find_spec_constant_name(self, script_name: str, job_type: str = 'training') -> Optional[str]:
        """
        Find the expected specification constant name for a script and job type.
        
        Args:
            script_name: Name of the script
            job_type: Job type variant (training, validation, testing, calibration)
            
        Returns:
            Expected constant name or None
        """
        # Generate based on discovered spec file patterns
        spec_file = self.find_spec_file(script_name)
        if spec_file:
            base_name = self.extract_base_name_from_spec(Path(spec_file))
            return f"{base_name.upper()}_{job_type.upper()}_SPEC"
        
        # Fallback to script name
        return f"{script_name.upper()}_{job_type.upper()}_SPEC"
