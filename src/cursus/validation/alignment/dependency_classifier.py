"""
Dependency pattern classification for alignment validation.

Provides classification logic to distinguish between different types
of dependencies (pipeline, external, configuration, environment) for
appropriate validation handling.
"""

from enum import Enum
from typing import Dict, Any


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
