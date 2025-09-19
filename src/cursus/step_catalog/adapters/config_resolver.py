"""
Config resolver adapters for backward compatibility.

This module provides adapters that maintain existing config resolver APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Type

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class StepConfigResolverAdapter:
    """
    Enhanced adapter maintaining backward compatibility with StepConfigResolver.
    
    Replaces: src/cursus/core/compiler/config_resolver.py
    
    This enhanced version includes essential legacy methods needed for production
    while leveraging the unified step catalog for superior discovery capabilities.
    """
    
    # Job type keywords for matching (simplified from legacy)
    JOB_TYPE_KEYWORDS = {
        "training": ["training", "train"],
        "calibration": ["calibration", "calib"],
        "evaluation": ["evaluation", "eval", "test"],
        "inference": ["inference", "infer", "predict"],
    }
    
    # Pattern mappings for step type detection (from legacy)
    STEP_TYPE_PATTERNS = {
        r".*data_load.*": ["CradleDataLoading"],
        r".*preprocess.*": ["TabularPreprocessing"],
        r".*train.*": ["XGBoostTraining", "PyTorchTraining", "DummyTraining"],
        r".*eval.*": ["XGBoostModelEval"],
        r".*model.*": ["XGBoostModel", "PyTorchModel"],
        r".*calibrat.*": ["ModelCalibration"],
        r".*packag.*": ["MIMSPackaging"],
        r".*payload.*": ["MIMSPayload"],
        r".*regist.*": ["ModelRegistration"],
        r".*transform.*": ["BatchTransform"],
        r".*currency.*": ["CurrencyConversion"],
        r".*risk.*": ["RiskTableMapping"],
        r".*hyperparam.*": ["HyperparameterPrep"],
    }
    
    def __init__(self, workspace_root: Optional[Path] = None, confidence_threshold: float = 0.7):
        """Initialize with unified catalog."""
        # PORTABLE: Use package-only discovery by default (works in all deployment scenarios)
        if workspace_root is None:
            self.catalog = StepCatalog(workspace_dirs=None)
        else:
            self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self._metadata_mapping = {}
        self._config_cache = {}
    
    def resolve_config_map(self, dag_nodes: List[str], available_configs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced resolution with direct name matching + catalog fallback."""
        try:
            resolved_configs = {}
            
            for node_name in dag_nodes:
                # 1. Try direct name matching first (handles 99% of cases)
                config = self._direct_name_matching(node_name, available_configs)
                if config is not None:
                    resolved_configs[node_name] = config
                    continue
                
                # 2. Fallback to catalog-based matching
                step_info = self.catalog.get_step_info(node_name)
                if step_info and step_info.config_class:
                    # Find config instance that matches the step's config class
                    for config_name, config_instance in available_configs.items():
                        if type(config_instance).__name__ == step_info.config_class:
                            resolved_configs[node_name] = config_instance
                            break
                else:
                    # 3. Last resort: first available config
                    if available_configs:
                        resolved_configs[node_name] = next(iter(available_configs.values()))
            
            return resolved_configs
            
        except Exception as e:
            self.logger.error(f"Error resolving config map: {e}")
            return {}
    
    def _direct_name_matching(self, node_name: str, configs: Dict[str, Any]) -> Optional[Any]:
        """
        Enhanced direct name matching with metadata support.
        
        Based on legacy implementation that supports metadata.config_types mapping.
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
                    if "_" in node_name:
                        node_parts = node_name.split("_")
                        if len(node_parts) > 1:
                            job_type = node_parts[-1].lower()
                            if (
                                hasattr(config, "job_type")
                                and getattr(config, "job_type", "").lower() == job_type
                            ):
                                self.logger.info(
                                    f"Found metadata mapping match with job type for node '{node_name}'"
                                )
                                return config
                    else:
                        self.logger.info(
                            f"Found metadata mapping match for node '{node_name}'"
                        )
                        return config

        # Case-insensitive match as fallback
        node_lower = node_name.lower()
        for config_name, config in configs.items():
            if config_name.lower() == node_lower:
                self.logger.info(
                    f"Found case-insensitive match for node '{node_name}': {config_name}"
                )
                return config

        return None
    
    def _job_type_matching(self, node_name: str, configs: Dict[str, Any]) -> List[tuple]:
        """
        Match based on job_type attribute and node naming patterns.
        
        Based on legacy implementation from StepConfigResolver.
        
        Args:
            node_name: DAG node name
            configs: Available configurations

        Returns:
            List of (config, confidence, method) tuples
        """
        matches: List[tuple] = []
        node_lower = node_name.lower()

        # Extract potential job type from node name (from legacy JOB_TYPE_KEYWORDS)
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

        # Find configs with matching job_type (legacy logic)
        for config_name, config in configs.items():
            if hasattr(config, "job_type"):
                config_job_type = getattr(config, "job_type", "").lower()

                # Check for job type match (legacy logic)
                job_type_keywords = self.JOB_TYPE_KEYWORDS.get(detected_job_type, [])
                if any(keyword in config_job_type for keyword in job_type_keywords):
                    # Calculate confidence based on how well the node name matches the config type
                    config_type_confidence = self._calculate_config_type_confidence(
                        node_name, config
                    )
                    total_confidence = 0.7 + (
                        config_type_confidence * 0.3
                    )  # Job type match + config type match
                    matches.append((config, total_confidence, "job_type"))

        return matches
    
    def _calculate_config_type_confidence(self, node_name: str, config: Any) -> float:
        """
        Calculate confidence based on how well node name matches config type.
        
        From legacy implementation.

        Args:
            node_name: DAG node name
            config: Configuration instance

        Returns:
            Confidence score (0.0 to 1.0)
        """
        config_type = type(config).__name__.lower()
        node_lower = node_name.lower()

        # Remove common suffixes for comparison
        config_base = config_type.replace("config", "").replace("step", "")

        # Check for substring matches
        if config_base in node_lower or any(
            part in node_lower for part in config_base.split("_")
        ):
            return 0.8

        # Use sequence matching for similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, node_lower, config_base).ratio()
        return similarity
    
    def _semantic_matching(self, node_name: str, configs: Dict[str, Any]) -> List[tuple]:
        """Semantic matching based on keywords."""
        matches = []
        
        # Define semantic keywords
        semantic_map = {
            'data': ['loading', 'load', 'cradle'],
            'preprocess': ['preprocessing', 'tabular', 'process'],
            'train': ['training', 'xgboost', 'pytorch', 'model'],
            'evaluate': ['evaluation', 'eval', 'test'],
            'transform': ['transformation', 'batch']
        }
        
        node_lower = node_name.lower()
        
        for config_key, config_instance in configs.items():
            config_lower = config_key.lower()
            confidence = 0.0
            
            # Check for semantic matches
            for category, keywords in semantic_map.items():
                if any(keyword in node_lower for keyword in keywords):
                    if any(keyword in config_lower for keyword in keywords):
                        confidence = 0.7
                        break
            
            if confidence > 0:
                matches.append((config_instance, confidence, "semantic"))
        
        return matches
    
    def _pattern_matching(self, node_name: str, configs: Dict[str, Any]) -> List[tuple]:
        """
        Use regex patterns to match node names to config types.
        
        Based on legacy implementation from StepConfigResolver.

        Args:
            node_name: DAG node name
            configs: Available configurations

        Returns:
            List of (config, confidence, method) tuples
        """
        matches: List[tuple] = []
        node_lower = node_name.lower()

        # Find matching patterns (from legacy STEP_TYPE_PATTERNS)
        matching_step_types = []
        for pattern, step_types in self.STEP_TYPE_PATTERNS.items():
            import re
            if re.match(pattern, node_lower):
                matching_step_types.extend(step_types)

        if not matching_step_types:
            return matches

        # Find configs that match the detected step types (legacy logic)
        for config_name, config in configs.items():
            config_type = type(config).__name__

            # Convert config class name to step type (legacy logic)
            step_type = self._config_class_to_step_type(config_type)

            if step_type in matching_step_types:
                # Base confidence for pattern match
                confidence = 0.6

                # Boost confidence if there are additional matches (legacy logic)
                if hasattr(config, "job_type"):
                    job_type_boost = self._calculate_job_type_boost(node_name, config)
                    confidence += job_type_boost * 0.2

                matches.append((config, min(confidence, 0.9), "pattern"))

        return matches
    
    def _config_class_to_step_type(self, config_class_name: str) -> str:
        """
        Convert configuration class name to step type using step catalog system.
        
        Enhanced to use step catalog's registry data for accurate mapping.

        Args:
            config_class_name: Configuration class name

        Returns:
            Step type name
        """
        try:
            # First, try to find the step type using the step catalog
            steps = self.catalog.list_available_steps()
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.config_class == config_class_name:
                    # Use the step name from the catalog as the step type
                    return step_name
                
                # Also check registry data for builder_step_name
                if step_info and step_info.registry_data.get('builder_step_name'):
                    builder_step_name = step_info.registry_data['builder_step_name']
                    if f"{builder_step_name}Config" == config_class_name:
                        return builder_step_name
            
            # Fallback to legacy logic if not found in catalog
            step_type = config_class_name

            # Remove 'Config' suffix
            if step_type.endswith("Config"):
                step_type = step_type[:-6]

            # Remove 'Step' suffix if present
            if step_type.endswith("Step"):
                step_type = step_type[:-4]

            # Handle special cases (legacy logic)
            if step_type == "CradleDataLoad":
                return "CradleDataLoading"
            elif step_type == "PackageStep" or step_type == "Package":
                return "MIMSPackaging"

            return step_type
            
        except Exception as e:
            self.logger.debug(f"Error using catalog for config class mapping, falling back to legacy: {e}")
            
            # Pure legacy fallback
            step_type = config_class_name
            if step_type.endswith("Config"):
                step_type = step_type[:-6]
            if step_type.endswith("Step"):
                step_type = step_type[:-4]
            if step_type == "CradleDataLoad":
                return "CradleDataLoading"
            elif step_type == "PackageStep" or step_type == "Package":
                return "MIMSPackaging"
            return step_type
    
    def _calculate_job_type_boost(self, node_name: str, config: Any) -> float:
        """
        Calculate confidence boost based on job type matching.
        
        From legacy implementation.

        Args:
            node_name: DAG node name
            config: Configuration instance

        Returns:
            Boost score (0.0 to 1.0)
        """
        if not hasattr(config, "job_type"):
            return 0.0

        config_job_type = getattr(config, "job_type", "").lower()
        node_lower = node_name.lower()

        # Check for job type keywords in node name (legacy logic)
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            if any(keyword in config_job_type for keyword in keywords):
                if any(keyword in node_lower for keyword in keywords):
                    return 1.0

        return 0.0
    
    def _resolve_single_node(self, node_name: str, configs: Dict[str, Any]) -> tuple:
        """Resolve a single node using all matching strategies."""
        # Try direct matching first
        direct_match = self._direct_name_matching(node_name, configs)
        if direct_match is not None:
            return (direct_match, 1.0, "direct_name")
        
        # Collect all matches from different strategies
        all_matches = []
        
        # Job type matching
        job_matches = self._job_type_matching(node_name, configs)
        all_matches.extend(job_matches)
        
        # Semantic matching
        semantic_matches = self._semantic_matching(node_name, configs)
        all_matches.extend(semantic_matches)
        
        # Pattern matching
        pattern_matches = self._pattern_matching(node_name, configs)
        all_matches.extend(pattern_matches)
        
        if not all_matches:
            try:
                from ...core.compiler.exceptions import ResolutionError
                raise ResolutionError(f"No configuration found for node: {node_name}")
            except ImportError:
                # Fallback if ResolutionError is not available
                raise ValueError(f"No configuration found for node: {node_name}")
        
        # Return the highest confidence match
        best_match = max(all_matches, key=lambda x: x[1])
        return best_match
    
    def resolve_config_for_step(self, step_name: str, configs: Dict[str, Any]) -> Optional[Any]:
        """
        Resolve configuration for a single step (used by generator.py).
        
        Args:
            step_name: Name of the step
            configs: Available configurations
            
        Returns:
            Resolved configuration or None
        """
        try:
            # Try direct name matching first
            config = self._direct_name_matching(step_name, configs)
            if config is not None:
                return config
            
            # Try enhanced resolution
            resolved_tuple = self._resolve_single_node(step_name, configs)
            if resolved_tuple:
                return resolved_tuple[0]  # Return just the config, not the tuple
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error resolving config for step {step_name}: {e}")
            return None
    
    def preview_resolution(self, dag_nodes: List[str], available_configs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced preview resolution with metadata support."""
        try:
            # Extract metadata.config_types mapping if available
            self._metadata_mapping = {}
            if metadata and "config_types" in metadata:
                self._metadata_mapping = metadata["config_types"]
                self.logger.info(f"Using metadata.config_types mapping with {len(self._metadata_mapping)} entries")
            
            node_resolution = {}
            resolution_confidence = {}
            node_config_map = {}
            
            for node in dag_nodes:
                try:
                    # Try to resolve the node
                    config, confidence, method = self._resolve_single_node(node, available_configs)
                    
                    # Store resolution info
                    node_resolution[node] = {
                        "config_type": type(config).__name__,
                        "confidence": confidence,
                        "method": method,
                        "job_type": getattr(config, "job_type", "N/A"),
                    }
                    
                    resolution_confidence[node] = confidence
                    node_config_map[node] = type(config).__name__
                    
                except Exception as e:
                    # Store error information
                    node_resolution[node] = {
                        "error": f"Step not found in catalog: {node}",
                        "error_type": "ResolutionError",
                    }
                    resolution_confidence[node] = 0.0
                    node_config_map[node] = "Unknown"
            
            return {
                "node_resolution": node_resolution,
                "resolution_confidence": resolution_confidence,
                "node_config_map": node_config_map,
                "metadata_mapping": self._metadata_mapping,
                "recommendations": [],
            }
            
        except Exception as e:
            self.logger.error(f"Error previewing resolution: {e}")
            return {"error": str(e)}
    
    def _parse_node_name(self, node_name: str) -> Dict[str, str]:
        """
        Parse node name to extract config type and job type information.
        
        Based on legacy implementation patterns from the original StepConfigResolver.
        
        Args:
            node_name: DAG node name
            
        Returns:
            Dictionary with extracted information
        """
        # Check cache first
        if node_name in self._config_cache:
            from typing import cast, Dict
            return cast(Dict[str, str], self._config_cache[node_name])

        result = {}

        # Common patterns from legacy implementation
        patterns = [
            # Pattern 1: ConfigType_JobType (e.g., CradleDataLoading_training)
            (r"^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$", "config_first"),
            # Pattern 2: JobType_Task (e.g., training_data_load)
            (r"^([a-z]+)_([A-Za-z_]+)$", "job_first"),
        ]

        import re
        for pattern, pattern_type in patterns:
            match = re.match(pattern, node_name)
            if match:
                parts = match.groups()

                if pattern_type == "config_first":  # ConfigType_JobType
                    result["config_type"] = parts[0]
                    result["job_type"] = parts[1]
                else:  # JobType_Task
                    result["job_type"] = parts[0]

                    # Try to infer config type from task (from legacy task_map)
                    task_map = {
                        "data_load": "CradleDataLoading",
                        "preprocess": "TabularPreprocessing",
                        "train": "XGBoostTraining",
                        "eval": "XGBoostModelEval",
                        "calibrat": "ModelCalibration",
                        "packag": "Package",
                        "regist": "Registration",
                        "payload": "Payload",
                    }

                    for task_pattern, config_type in task_map.items():
                        if task_pattern in parts[1]:
                            result["config_type"] = config_type
                            break

                break

        # Cache the result
        self._config_cache[node_name] = result
        return result
    
    def _job_type_matching_enhanced(self, job_type: str, configs: Dict[str, Any], config_type: Optional[str] = None) -> List[tuple]:
        """
        Enhanced job type matching with config type filtering.
        
        Args:
            job_type: Job type string (e.g., "training", "calibration")
            configs: Available configurations
            config_type: Optional config type to filter by
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        normalized_job_type = job_type.lower()
        
        for config_name, config in configs.items():
            if hasattr(config, "job_type"):
                config_job_type = getattr(config, "job_type", "").lower()
                
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
                    elif (config_type_lower in class_name_lower or class_name_lower in config_type_lower):
                        # Partial match
                        base_confidence = 0.85
                
                matches.append((config, base_confidence, "job_type_enhanced"))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
