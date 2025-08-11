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
    Flexible file resolution with multiple naming pattern support.
    
    This resolver addresses the critical false positive issue where the alignment
    testers look for files with incorrect naming patterns. It provides fuzzy
    matching and handles the actual naming conventions used in the codebase.
    """
    
    def __init__(self, base_directories: Dict[str, str]):
        """
        Initialize the file resolver with base directories.
        
        Args:
            base_directories: Dictionary mapping component types to their base directories
                             e.g., {'contracts': 'src/cursus/steps/contracts', ...}
        """
        self.base_dirs = base_directories
        self.naming_patterns = self._load_naming_patterns()
    
    def _load_naming_patterns(self) -> Dict[str, Dict[str, str]]:
        """
        Load known naming pattern mappings for common scripts.
        
        Returns:
            Dictionary mapping script names to their actual file names
        """
        return {
            'contracts': {
                'model_evaluation_xgb': 'model_evaluation_contract.py',
                'dummy_training': 'dummy_training_contract.py',
                'currency_conversion': 'currency_conversion_contract.py',
                'mims_package': 'mims_package_contract.py',
                'mims_payload': 'mims_payload_contract.py',
                'model_calibration': 'model_calibration_contract.py',
                'risk_table_mapping': 'risk_table_mapping_contract.py',
                'tabular_preprocess': 'tabular_preprocess_contract.py',
            },
            'specs': {
                'model_evaluation_xgb': 'model_eval_spec.py',
                'dummy_training': 'dummy_training_spec.py',
                'currency_conversion': 'currency_conversion_training_spec.py',  # Has variants
                'mims_package': 'packaging_spec.py',
                'mims_payload': 'payload_spec.py',
                'model_calibration': 'model_calibration_spec.py',
                'risk_table_mapping': 'risk_table_mapping_training_spec.py',  # Has variants
                'tabular_preprocess': 'preprocessing_training_spec.py',  # Has variants
            },
            'builders': {
                'model_evaluation_xgb': 'builder_model_eval_step_xgboost.py',
                'dummy_training': 'builder_dummy_training_step.py',
                'currency_conversion': 'builder_currency_conversion_step.py',
                'mims_package': 'builder_package_step.py',
                'mims_payload': 'builder_payload_step.py',
                'model_calibration': 'builder_model_calibration_step.py',
                'risk_table_mapping': 'builder_risk_table_mapping_step.py',
                'tabular_preprocess': 'builder_tabular_preprocessing_step.py',
            },
            'configs': {
                'model_evaluation_xgb': 'config_model_eval_step_xgboost.py',
                'dummy_training': 'config_dummy_training_step.py',
                'currency_conversion': 'config_currency_conversion_step.py',
                'mims_package': 'config_package_step.py',
                'mims_payload': 'config_payload_step.py',
                'model_calibration': 'config_model_calibration_step.py',
                'risk_table_mapping': 'config_risk_table_mapping_step.py',
                'tabular_preprocess': 'config_tabular_preprocessing_step.py',
            }
        }
    
    def find_contract_file(self, script_name: str) -> Optional[str]:
        """
        Find contract file using flexible naming patterns.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the contract file or None if not found
        """
        patterns = [
            f"{script_name}_contract.py",
            f"{self._normalize_name(script_name)}_contract.py",
        ]
        
        # Add known pattern if available
        if script_name in self.naming_patterns['contracts']:
            patterns.insert(0, self.naming_patterns['contracts'][script_name])
        
        return self._find_file_by_patterns(self.base_dirs.get('contracts', ''), patterns)
    
    def find_spec_file(self, script_name: str) -> Optional[str]:
        """
        Find specification file using flexible naming patterns.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the specification file or None if not found
        """
        patterns = [
            f"{script_name}_spec.py",
            f"{self._normalize_name(script_name)}_spec.py",
        ]
        
        # Add known pattern if available
        if script_name in self.naming_patterns['specs']:
            patterns.insert(0, self.naming_patterns['specs'][script_name])
        
        return self._find_file_by_patterns(self.base_dirs.get('specs', ''), patterns)
    
    def find_builder_file(self, script_name: str) -> Optional[str]:
        """
        Find builder file using flexible naming patterns.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the builder file or None if not found
        """
        # Generate all possible name variations
        name_variations = self._generate_name_variations(script_name)
        
        patterns = []
        
        # Add known pattern if available (highest priority)
        if script_name in self.naming_patterns['builders']:
            patterns.append(self.naming_patterns['builders'][script_name])
        
        # Add patterns for all name variations
        for name_var in name_variations:
            patterns.extend([
                f"builder_{name_var}_step.py",
                f"{name_var}_step_builder.py",  # Alternative pattern
                f"builder_{name_var}.py",       # Simplified pattern
            ])
        
        return self._find_file_by_patterns(self.base_dirs.get('builders', ''), patterns)
    
    def find_config_file(self, script_name: str) -> Optional[str]:
        """
        Find config file using flexible naming patterns.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the config file or None if not found
        """
        patterns = [
            f"config_{script_name}_step.py",
            f"config_{self._normalize_name(script_name)}_step.py",
        ]
        
        # Add known pattern if available
        if script_name in self.naming_patterns['configs']:
            patterns.insert(0, self.naming_patterns['configs'][script_name])
        
        return self._find_file_by_patterns(self.base_dirs.get('configs', ''), patterns)
    
    def _find_file_by_patterns(self, directory: str, patterns: List[str]) -> Optional[str]:
        """
        Find file using multiple patterns, return first match.
        
        Args:
            directory: Directory to search in
            patterns: List of filename patterns to try
            
        Returns:
            Full path to the file or None if not found
        """
        if not directory:
            return None
            
        dir_path = Path(directory)
        if not dir_path.exists():
            return None
        
        for pattern in patterns:
            if pattern is None:
                continue
            file_path = dir_path / pattern
            if file_path.exists():
                return str(file_path)
        
        # If no exact match, try fuzzy matching
        return self._fuzzy_find_file(directory, patterns[0])
    
    def _fuzzy_find_file(self, directory: str, target_pattern: str) -> Optional[str]:
        """
        Fuzzy file matching for similar names.
        
        Args:
            directory: Directory to search in
            target_pattern: Target filename pattern
            
        Returns:
            Path to the best matching file or None
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return None
            
        target_base = target_pattern.replace('.py', '').lower()
        
        best_match = None
        best_similarity = 0.0
        
        for file_path in dir_path.glob('*.py'):
            file_base = file_path.stem.lower()
            similarity = self._calculate_similarity(target_base, file_base)
            
            if similarity > 0.8 and similarity > best_similarity:
                best_similarity = similarity
                best_match = str(file_path)
        
        return best_match
    
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
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a script name for pattern matching.
        
        Args:
            name: Script name to normalize
            
        Returns:
            Normalized name
        """
        # Remove common suffixes and prefixes
        normalized = name.replace('_step', '').replace('step_', '')
        normalized = normalized.replace('_script', '').replace('script_', '')
        
        # Handle common abbreviations and variations
        abbreviations = {
            'xgb': 'xgboost',
            'eval': 'evaluation',
            'preprocess': 'preprocessing',
        }
        
        for abbrev, full in abbreviations.items():
            if abbrev in normalized:
                normalized = normalized.replace(abbrev, full)
        
        return normalized
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """
        Generate common naming variations for a script name.
        
        Args:
            name: Original script name
            
        Returns:
            List of possible naming variations
        """
        variations = [name]
        
        # Add normalized version
        normalized = self._normalize_name(name)
        if normalized != name:
            variations.append(normalized)
        
        # Handle specific common variations
        if 'preprocess' in name and 'preprocessing' not in name:
            variations.append(name.replace('preprocess', 'preprocessing'))
        elif 'preprocessing' in name and 'preprocess' not in name:
            variations.append(name.replace('preprocessing', 'preprocess'))
        
        if 'eval' in name and 'evaluation' not in name:
            variations.append(name.replace('eval', 'evaluation'))
        elif 'evaluation' in name and 'eval' not in name:
            variations.append(name.replace('evaluation', 'eval'))
        
        if 'xgb' in name and 'xgboost' not in name:
            variations.append(name.replace('xgb', 'xgboost'))
        elif 'xgboost' in name and 'xgb' not in name:
            variations.append(name.replace('xgboost', 'xgb'))
        
        return list(set(variations))  # Remove duplicates
    
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
        # Check if we have a known mapping
        if script_name in self.naming_patterns.get('spec_constants', {}):
            constants_map = self.naming_patterns['spec_constants'][script_name]
            if job_type in constants_map:
                return constants_map[job_type]
            elif 'default' in constants_map:
                return constants_map['default']
        
        # Generate based on patterns
        # First try the FlexibleFileResolver spec mapping
        spec_file = self.find_spec_file(script_name)
        if spec_file:
            base_name = self.extract_base_name_from_spec(Path(spec_file))
            return f"{base_name.upper()}_{job_type.upper()}_SPEC"
        
        # Fallback to script name
        return f"{script_name.upper()}_{job_type.upper()}_SPEC"
