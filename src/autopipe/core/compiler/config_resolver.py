"""
Configuration Resolver for the Pipeline API.

This module provides intelligent matching of DAG nodes to configuration instances
using multiple resolution strategies with enhanced handling for job types and
configuration variants.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import re
import logging
from difflib import SequenceMatcher

from ..base import BasePipelineConfig
from .exceptions import ConfigurationError, AmbiguityError, ResolutionError

logger = logging.getLogger(__name__)


class StepConfigResolver:
    """
    Resolves DAG nodes to configuration instances using intelligent matching.
    
    This class implements multiple resolution strategies to match DAG node
    names to configuration instances from the loaded configuration file.
    """
    
    # Pattern mappings for step type detection
    STEP_TYPE_PATTERNS = {
        r'.*data_load.*': ['CradleDataLoading'],
        r'.*preprocess.*': ['TabularPreprocessing'],
        r'.*train.*': ['XGBoostTraining', 'PyTorchTraining', 'DummyTraining'],
        r'.*eval.*': ['XGBoostModelEval'],
        r'.*model.*': ['XGBoostModel', 'PyTorchModel'],
        r'.*calibrat.*': ['ModelCalibration'],
        r'.*packag.*': ['MIMSPackaging'],
        r'.*payload.*': ['MIMSPayload'],
        r'.*regist.*': ['ModelRegistration'],
        r'.*transform.*': ['BatchTransform'],
        r'.*currency.*': ['CurrencyConversion'],
        r'.*risk.*': ['RiskTableMapping'],
        r'.*hyperparam.*': ['HyperparameterPrep'],
    }
    
    # Job type keywords for matching
    JOB_TYPE_KEYWORDS = {
        'train': ['training', 'train'],
        'calib': ['calibration', 'calib'],
        'eval': ['evaluation', 'eval', 'test'],
        'inference': ['inference', 'infer', 'predict'],
        'validation': ['validation', 'valid'],
    }
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the config resolver.
        
        Args:
            confidence_threshold: Minimum confidence score for automatic resolution
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self._metadata_mapping = {}  # Step name to config mapping from metadata
        self._config_cache = {}      # Cache for parsed node names
    
    def resolve_config_map(
        self, 
        dag_nodes: List[str], 
        available_configs: Dict[str, BasePipelineConfig],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, BasePipelineConfig]:
        """
        Resolve DAG nodes to configurations with enhanced metadata handling.
        
        Resolution strategies (in order of preference):
        1. Direct name matching with exact match
        2. Metadata mapping from config_types
        3. Job type + config type matching with pattern recognition
        4. Semantic similarity matching
        5. Pattern-based matching
        
        Args:
            dag_nodes: List of DAG node names
            available_configs: Available configuration instances
            metadata: Optional metadata from configuration file
            
        Returns:
            Dictionary mapping node names to configuration instances
            
        Raises:
            ConfigurationError: If nodes cannot be resolved
            AmbiguityError: If multiple configs match with similar confidence
        """
        # Extract metadata.config_types mapping if available
        self._metadata_mapping = {}
        if metadata and "config_types" in metadata:
            self._metadata_mapping = metadata["config_types"]
            self.logger.info(f"Using metadata.config_types mapping with {len(self._metadata_mapping)} entries")
        
        # Clear cache for this resolution session
        self._config_cache = {}
        
        # Proceed with node resolution
        resolved_configs = {}
        unresolved_nodes = []
        ambiguous_nodes = []
        
        for node_name in dag_nodes:
            try:
                config, confidence, method = self._resolve_single_node(node_name, available_configs)
                resolved_configs[node_name] = config
                self.logger.info(f"Resolved node '{node_name}' to {type(config).__name__} "
                                f"(job_type='{getattr(config, 'job_type', 'N/A')}') "
                                f"with confidence {confidence:.2f} using {method} matching")
            except ResolutionError as e:
                self.logger.warning(f"Failed to resolve node '{node_name}': {str(e)}")
                unresolved_nodes.append(node_name)
            except AmbiguityError as e:
                self.logger.warning(f"Ambiguity when resolving node '{node_name}': {str(e)}")
                ambiguous_nodes.append((node_name, e.candidates))
        
        # If any nodes are unresolved, raise an error
        if unresolved_nodes:
            available_config_names = list(available_configs.keys())
            raise ConfigurationError(
                f"Failed to resolve {len(unresolved_nodes)} DAG nodes to configurations",
                missing_configs=unresolved_nodes,
                available_configs=available_config_names
            )
        
        # If any nodes are ambiguous, raise a detailed error
        if ambiguous_nodes:
            first_node, candidates = ambiguous_nodes[0]
            candidate_info = []
            
            for candidate in candidates:
                config, confidence, method = candidate
                job_type = getattr(config, 'job_type', None)
                config_type = type(config).__name__
                candidate_info.append(f"  - {config_type} (job_type='{job_type}', confidence={confidence:.2f})")
            
            details = "\n".join(candidate_info)
            raise AmbiguityError(
                f"Multiple configurations match node '{first_node}' with similar confidence\n"
                f"Candidates for node '{first_node}':\n{details}",
                node_name=first_node,
                candidates=[{
                    "config_type": type(c[0]).__name__,
                    "job_type": getattr(c[0], 'job_type', None),
                    "confidence": c[1],
                    "method": c[2]
                } for c in candidates]
            )
        
        return resolved_configs
    
    def _resolve_single_node(
        self,
        node_name: str,
        available_configs: Dict[str, BasePipelineConfig]
    ) -> Tuple[BasePipelineConfig, float, str]:
        """
        Resolve a single DAG node to a configuration using enhanced tiered approach.
        
        This method implements a tiered resolution strategy that prioritizes
        exact matches and makes better use of node name patterns and job types.
        
        Args:
            node_name: DAG node name
            available_configs: Available configuration instances
            
        Returns:
            Tuple of (config, confidence_score, resolution_method)
            
        Raises:
            ResolutionError: If no suitable config found
            AmbiguityError: If multiple configs match with similar confidence
        """
        # Tier 1: Try direct name matching - if successful, return immediately with highest confidence
        direct_match = self._direct_name_matching(node_name, available_configs)
        if direct_match:
            return direct_match, 1.0, 'direct_name'
        
        # Tier 2: Parse node name for information
        parsed_info = self._parse_node_name(node_name)
        
        # Tier 3: Try job type matching if we extracted job type information
        if parsed_info and 'job_type' in parsed_info:
            job_type_matches = self._job_type_matching_enhanced(
                parsed_info['job_type'], 
                available_configs,
                config_type=parsed_info.get('config_type')
            )
            
            # If we found matches, use the best one
            if job_type_matches:
                best_match = max(job_type_matches, key=lambda x: x[1])
                return best_match[0], best_match[1], 'job_type_enhanced'
        
        # Tier 4: Fall back to traditional matching strategies
        candidates = []
        
        # Job type matching (traditional)
        job_type_matches = self._job_type_matching(node_name, available_configs)
        candidates.extend(job_type_matches)
        
        # Semantic matching
        semantic_matches = self._semantic_matching(node_name, available_configs)
        candidates.extend(semantic_matches)
        
        # Pattern matching
        pattern_matches = self._pattern_matching(node_name, available_configs)
        candidates.extend(pattern_matches)
        
        # If no candidates, resolution fails
        if not candidates:
            raise ResolutionError(f"No matching configurations found for node '{node_name}'")
        
        # Sort by confidence and return the best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_match = candidates[0]
        
        # Check if confidence is above threshold
        if best_match[1] >= self.confidence_threshold:
            return best_match
        
        # If multiple matches with similar confidence, report ambiguity
        close_matches = [c for c in candidates if c[1] >= best_match[1] - 0.05]
        if len(close_matches) > 1:
            raise AmbiguityError(
                f"Multiple configurations match node '{node_name}' with similar confidence",
                candidates=close_matches
            )
        
        # Otherwise, return best match even below threshold (with warning)
        self.logger.warning(
            f"Using best match for '{node_name}' with below-threshold confidence {best_match[1]:.2f}"
        )
        return best_match
    
    def _direct_name_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> Optional[BasePipelineConfig]:
        """
        Match node name directly to configuration using enhanced matching.
        
        This method prioritizes exact matches and makes use of the metadata.config_types
        mapping for more precise matching.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            Matching configuration or None
        """
        # First priority: Direct match with config key
        if node_name in configs:
            self.logger.info(f"Found exact key match for node '{node_name}'")
            return configs[node_name]
        
        # Second priority: Check metadata.config_types mapping if available
        if self._metadata_mapping and node_name in self._metadata_mapping:
            config_class_name = self._metadata_mapping[node_name]
            
            # Find configs of the specified class
            for config_name, config in configs.items():
                if type(config).__name__ == config_class_name:
                    # If job type is part of the node name, check for match
                    if '_' in node_name:
                        node_parts = node_name.split('_')
                        if len(node_parts) > 1:
                            job_type = node_parts[-1].lower()
                            if hasattr(config, 'job_type') and getattr(config, 'job_type', '').lower() == job_type:
                                self.logger.info(f"Found metadata mapping match with job type for node '{node_name}'")
                                return config
                    else:
                        self.logger.info(f"Found metadata mapping match for node '{node_name}'")
                        return config
        
        # Case-insensitive match as fallback
        node_lower = node_name.lower()
        for config_name, config in configs.items():
            if config_name.lower() == node_lower:
                self.logger.info(f"Found case-insensitive match for node '{node_name}': {config_name}")
                return config
        
        return None
    
    def _job_type_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Match based on job_type attribute and node naming patterns.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        node_lower = node_name.lower()
        
        # Extract potential job type from node name
        detected_job_type = None
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in node_lower:
                    detected_job_type = job_type
                    break
            if detected_job_type:
                break
        
        if not detected_job_type:
            return matches
        
        # Find configs with matching job_type
        for config_name, config in configs.items():
            if hasattr(config, 'job_type'):
                config_job_type = getattr(config, 'job_type', '').lower()
                
                # Check for job type match
                job_type_keywords = self.JOB_TYPE_KEYWORDS.get(detected_job_type, [])
                if any(keyword in config_job_type for keyword in job_type_keywords):
                    # Calculate confidence based on how well the node name matches the config type
                    config_type_confidence = self._calculate_config_type_confidence(node_name, config)
                    total_confidence = 0.7 + (config_type_confidence * 0.3)  # Job type match + config type match
                    matches.append((config, total_confidence, 'job_type'))
        
        return matches
    
    def _semantic_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Use semantic similarity to match node names to config types.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        
        for config_name, config in configs.items():
            confidence = self._calculate_semantic_similarity(node_name, config)
            if confidence >= 0.5:  # Minimum semantic similarity threshold
                matches.append((config, confidence, 'semantic'))
        
        return matches
    
    def _pattern_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Use regex patterns to match node names to config types.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        node_lower = node_name.lower()
        
        # Find matching patterns
        matching_step_types = []
        for pattern, step_types in self.STEP_TYPE_PATTERNS.items():
            if re.match(pattern, node_lower):
                matching_step_types.extend(step_types)
        
        if not matching_step_types:
            return matches
        
        # Find configs that match the detected step types
        for config_name, config in configs.items():
            config_type = type(config).__name__
            
            # Convert config class name to step type
            step_type = self._config_class_to_step_type(config_type)
            
            if step_type in matching_step_types:
                # Base confidence for pattern match
                confidence = 0.6
                
                # Boost confidence if there are additional matches
                if hasattr(config, 'job_type'):
                    job_type_boost = self._calculate_job_type_boost(node_name, config)
                    confidence += job_type_boost * 0.2
                
                matches.append((config, min(confidence, 0.9), 'pattern'))
        
        return matches
    
    def _calculate_config_type_confidence(
        self,
        node_name: str,
        config: BasePipelineConfig
    ) -> float:
        """
        Calculate confidence based on how well node name matches config type.
        
        Args:
            node_name: DAG node name
            config: Configuration instance
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        config_type = type(config).__name__.lower()
        node_lower = node_name.lower()
        
        # Remove common suffixes for comparison
        config_base = config_type.replace('config', '').replace('step', '')
        
        # Check for substring matches
        if config_base in node_lower or any(part in node_lower for part in config_base.split('_')):
            return 0.8
        
        # Use sequence matching for similarity
        similarity = SequenceMatcher(None, node_lower, config_base).ratio()
        return similarity
    
    def _calculate_semantic_similarity(
        self,
        node_name: str,
        config: BasePipelineConfig
    ) -> float:
        """
        Calculate semantic similarity between node name and config.
        
        Args:
            node_name: DAG node name
            config: Configuration instance
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        config_type = type(config).__name__.lower()
        node_lower = node_name.lower()
        
        # Define semantic mappings
        semantic_mappings = {
            'data': ['cradle', 'load', 'loading'],
            'preprocess': ['preprocessing', 'process', 'clean'],
            'train': ['training', 'fit', 'learn', 'model_fit'],  # Added 'model_fit' as a synonym for training
            'eval': ['evaluation', 'evaluate', 'test', 'assess', 'model_test'],  # Added 'model_test' as a synonym
            'model': ['model', 'create', 'build'],
            'calibrat': ['calibration', 'calibrate', 'adjust'],
            'packag': ['packaging', 'package', 'bundle'],
            'regist': ['registration', 'register', 'deploy'],
        }
        
        max_similarity = 0.0
        
        for semantic_key, synonyms in semantic_mappings.items():
            if semantic_key in config_type:
                for synonym in synonyms:
                    if synonym in node_lower:
                        # Calculate similarity based on how well the synonym matches
                        similarity = SequenceMatcher(None, node_lower, synonym).ratio()
                        max_similarity = max(max_similarity, similarity * 0.8)  # Scale down semantic matches
        
        return max_similarity
    
    def _calculate_job_type_boost(
        self,
        node_name: str,
        config: BasePipelineConfig
    ) -> float:
        """
        Calculate confidence boost based on job type matching.
        
        Args:
            node_name: DAG node name
            config: Configuration instance
            
        Returns:
            Boost score (0.0 to 1.0)
        """
        if not hasattr(config, 'job_type'):
            return 0.0
        
        config_job_type = getattr(config, 'job_type', '').lower()
        node_lower = node_name.lower()
        
        # Check for job type keywords in node name
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            if any(keyword in config_job_type for keyword in keywords):
                if any(keyword in node_lower for keyword in keywords):
                    return 1.0
        
        return 0.0
    
    def _config_class_to_step_type(self, config_class_name: str) -> str:
        """
        Convert configuration class name to step type.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            Step type name
        """
        # Use the same logic as in builder_registry
        step_type = config_class_name
        
        # Remove 'Config' suffix
        if step_type.endswith('Config'):
            step_type = step_type[:-6]
        
        # Remove 'Step' suffix if present
        if step_type.endswith('Step'):
            step_type = step_type[:-4]
        
        # Handle special cases
        if step_type == "CradleDataLoad":
            return "CradleDataLoading"
        elif step_type == "PackageStep" or step_type == "Package":
            return "MIMSPackaging"
        
        return step_type
    
    def _parse_node_name(self, node_name: str) -> Dict[str, str]:
        """
        Parse node name to extract config type and job type information.
        
        Args:
            node_name: DAG node name
            
        Returns:
            Dictionary with extracted information
        """
        # Check if we've already parsed this node name
        if node_name in self._config_cache:
            return self._config_cache[node_name]
        
        result = {}
        
        # Common patterns
        patterns = [
            # Pattern 1: ConfigType_JobType (e.g., CradleDataLoading_training)
            (r'^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$', 'config_first'),
            
            # Pattern 2: JobType_Task (e.g., training_data_load)
            (r'^([a-z]+)_([A-Za-z_]+)$', 'job_first'),
        ]
        
        for pattern, pattern_type in patterns:
            match = re.match(pattern, node_name)
            if match:
                parts = match.groups()
                
                if pattern_type == 'config_first':  # ConfigType_JobType
                    result['config_type'] = parts[0]
                    result['job_type'] = parts[1]
                else:  # JobType_Task
                    result['job_type'] = parts[0]
                    
                    # Try to infer config type from task
                    task_map = {
                        'data_load': 'CradleDataLoading',
                        'preprocess': 'TabularPreprocessing',
                        'train': 'XGBoostTraining',
                        'eval': 'XGBoostModelEval',
                        'calibrat': 'ModelCalibration',
                        'packag': 'Package',
                        'regist': 'Registration',
                        'payload': 'Payload'
                    }
                    
                    for task_pattern, config_type in task_map.items():
                        if task_pattern in parts[1]:
                            result['config_type'] = config_type
                            break
                
                break
        
        # Cache the result
        self._config_cache[node_name] = result
        return result
        
    def _job_type_matching_enhanced(
        self, 
        job_type: str, 
        configs: Dict[str, BasePipelineConfig],
        config_type: Optional[str] = None
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Match configurations based on job type with improved accuracy.
        
        Args:
            job_type: Job type string
            configs: Available configurations
            config_type: Optional config type to filter by
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        
        normalized_job_type = job_type.lower()
        
        # For each config, check if the job_type matches
        for config_name, config in configs.items():
            if hasattr(config, 'job_type'):
                config_job_type = getattr(config, 'job_type', '').lower()
                
                # Skip if job types don't match
                if config_job_type != normalized_job_type:
                    continue
                
                # Start with base confidence for job type match
                base_confidence = 0.8
                
                # If config_type is specified, check for match to boost confidence
                if config_type:
                    config_class_name = type(config).__name__
                    config_type_lower = config_type.lower()
                    class_name_lower = config_class_name.lower()
                    
                    # Different levels of match for config type
                    if config_class_name == config_type:
                        # Exact match
                        base_confidence = 0.9
                    elif config_type_lower in class_name_lower or class_name_lower in config_type_lower:
                        # Partial match
                        base_confidence = 0.85
                
                matches.append((config, base_confidence, 'job_type_enhanced'))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
        
    def preview_resolution(
        self,
        dag_nodes: List[str],
        available_configs: Dict[str, BasePipelineConfig],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Preview resolution candidates for each DAG node with enhanced diagnostics.
        
        Args:
            dag_nodes: List of DAG node names
            available_configs: Available configuration instances
            metadata: Optional metadata from configuration file
            
        Returns:
            Dictionary with resolution preview information including node-to-config mapping
            and diagnostic recommendations
        """
        # Set up metadata if provided
        if metadata and "config_types" in metadata:
            self._metadata_mapping = metadata["config_types"]
        else:
            self._metadata_mapping = {}
        
        # Clear cache for this preview session
        self._config_cache = {}
        
        # Collect resolution information
        node_resolution = {}
        resolution_confidence = {}
        node_config_map = {}
        recommendations = []
        
        for node in dag_nodes:
            try:
                # Try to resolve the node
                config, confidence, method = self._resolve_single_node(node, available_configs)
                
                # Store resolution info
                node_resolution[node] = {
                    'config_type': type(config).__name__,
                    'confidence': confidence,
                    'method': method,
                    'job_type': getattr(config, 'job_type', 'N/A')
                }
                
                resolution_confidence[node] = confidence
                node_config_map[node] = type(config).__name__
                
            except (ResolutionError, AmbiguityError) as e:
                # Store error information
                node_resolution[node] = {
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                
                if isinstance(e, AmbiguityError) and hasattr(e, 'candidates'):
                    # Add recommendations for ambiguous nodes
                    candidate_info = []
                    for candidate in e.candidates:
                        if isinstance(candidate, tuple):
                            config, confidence, method = candidate
                            job_type = getattr(config, 'job_type', 'N/A')
                            candidate_info.append(f"{type(config).__name__} (job_type='{job_type}', confidence={confidence:.2f})")
                    
                    recommendations.append(
                        f"Node '{node}' has multiple matching configurations with similar confidence:\n"
                        f"  {', '.join(candidate_info)}\n"
                        f"Consider renaming the node or configs to resolve ambiguity."
                    )
        
        return {
            'node_resolution': node_resolution,
            'resolution_confidence': resolution_confidence,
            'node_config_map': node_config_map,
            'metadata_mapping': self._metadata_mapping,
            'recommendations': recommendations
        }
